# from llama_model import VaLlamaForCausalLM
# from vappt_llama_model import VaPPTLlamaForCausalLM
from torch import nn
import torch
from transformers import LlamaTokenizer, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3ForCausalLM
import os
use_auth_token = os.getenv("HUGGING_FACE_TOKEN")
class EncoderWrapperSupervised(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        attention_mask = attention_masks.to(input_ids.device)

        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        all_values = torch.stack(output.all_values, dim=0)
        last_values = all_values[-14:]
        token_values = last_values.mean(dim=0)   # [B, T, H]

        # 更稳健地找最后一个有效 token
        reversed_mask = torch.flip(attention_mask, dims=[1])               # [B, T]
        last_offset = reversed_mask.float().argmax(dim=1)                  # [B]
        last_indices = attention_mask.size(1) - 1 - last_offset           # [B]

        batch_idx = torch.arange(token_values.size(0), device=token_values.device)
        embedding = token_values[batch_idx, last_indices]                  # [B, H]

        return embedding


class EncoderWrapperSupervisedLastToken(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        # attention_masks: [batch, seq_len]，1 为有效 token，0 为 padding
        attention_mask = attention_masks.to(input_ids.device)

        # 调用底层编码器，LlamaForCausalLM 
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 你之前就是这样写的，保持不动
        )

        embedding = output.hidden_states[-1] #[B, T, H]

        lengths = attention_mask.sum(dim=1)          # [B]
        last_indices = (lengths - 1).clamp(min=0)    # [B]，最后一个有效 token 的下标

        batch_size = embedding.size(0)

        # 取每个样本最后一个有效 token 的 embedding
        last_token_embeddings = embedding[
            torch.arange(batch_size, device=embedding.device),
            last_indices
        ]  # [B, H]

        
        return last_token_embeddings  # 不在这里做归一化，在外面再 normalize
    

import torch
import torch.nn as nn


class EncoderWrapperSupervisedAppendN(nn.Module):
    def __init__(
        self,
        encoder,
        residual=False,
        init_from_decode_then_reembed=True,
        use_lm_head_bias=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.residual = residual
        hidden_size = encoder.base_model.model.lm_head.weight.shape[1]
        vocab_size = encoder.base_model.model.lm_head.weight.shape[0]
        self.gate = nn.Linear(hidden_size, vocab_size, bias=False)
        self.ffn = nn.Linear(vocab_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.prefix_embedding = nn.Linear(hidden_size, 10, bias=False) # 词表层前10个token初始化
        self._init_gate_ffn()

    def _init_gate_ffn(self):
        with torch.no_grad():
            input_embed = self.encoder.get_input_embeddings().weight   # [V, H]
            lm_head = self.encoder.lm_head.weight                      # [V, H]
            self.gate.weight.copy_(lm_head)
            self.ffn.weight.copy_(input_embed.transpose(0, 1))
            self.prefix_embedding.weight.copy_(input_embed[:10])  # 用词表前10个token的embedding初始化 prefix_embedding

    @staticmethod
    def _assert_left_padding(attention_mask: torch.Tensor):
        """
        检查每一行是否满足 left padding:
            0 ... 0 1 ... 1
        """
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be 2D: [B, T]")

        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0/1")

        # left padding 要求 mask 单调不下降
        ok = torch.all(attention_mask[:, 1:] >= attention_mask[:, :-1])
        if not ok:
            raise ValueError(
                "This implementation requires LEFT padding: each row of attention_mask "
                "must look like 0...011...1."
            )

        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("Each sample must contain at least one valid token.")

    @staticmethod
    def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        left padding 下:
            mask = [0,0,1,1,1] -> pos = [0,0,0,1,2]
        pad 位置填 0，仅作占位，不参与注意力。
        """
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    @staticmethod
    def _last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        返回每个样本最后一个有效 token 的物理列下标。
        这个函数对任意 padding 都成立。
        """
        return attention_mask.size(1) - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1)

    def _base_transformer(self):
        return self.encoder.base_model.model.model

    def forward(
        self,
        input_ids,
        attention_masks,
        n_steps=4,
        return_all=False,
        query=False,
        dummy_tensor=None
    ):
        """
        严格支持:
            - batch
            - left padding
            - n 次连续 append
            - 整段重跑版

        参数:
            input_ids: [B, T]
            attention_masks: [B, T]
            n_steps: 连续追加多少次
            return_all: 是否返回每一步的中间结果

        返回:
            默认返回第 n 步追加后的最后 hidden, [B, H]

            若 return_all=True，则返回字典，包含:
                - final_hidden
                - initial_last_hidden
                - all_appended_embeds
                - all_appended_hiddens
                - all_physical_new_indices
                - all_semantic_new_position_ids
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        
        if query:
            prefix_embed = self.prefix_embedding.weight  # [10, H]
            input_embeds = self.encoder.get_input_embeddings()(input_ids)  # [B, T, H]

            # 对原始输入进行left padding检查
            self._assert_left_padding(attention_masks)

            # 计算每个样本的有效长度和padding长度
            original_base_lengths = attention_masks.long().sum(dim=1)  # [B]，原始序列的有效长度
            padding_lengths = input_ids.size(1) - original_base_lengths  # [B]
            # base_lengths应该是原始有效长度加上prefix的长度（10）
            base_lengths = original_base_lengths + 10  # [B]，包含prefix的有效长度

            # 为每个样本构造新的embeds和mask
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            hidden_size = input_embeds.size(2)
            device = input_ids.device

            new_embeds = []
            new_masks = []

            for i in range(batch_size):
                pad_len = padding_lengths[i].item()
                valid_len = original_base_lengths[i].item()  # 使用原始有效长度

                # 获取当前样本的padding部分、有效部分
                pad_embeds = input_embeds[i, :pad_len, :]  # [pad_len, H]
                valid_embeds = input_embeds[i, pad_len:pad_len+valid_len, :]  # [valid_len, H]

                # 构造新序列：padding + prefix + valid
                new_embed = torch.cat([
                    pad_embeds,  # [pad_len, H]
                    prefix_embed,  # [10, H]
                    valid_embeds  # [valid_len, H]
                ], dim=0)  # [pad_len + 10 + valid_len, H]

                # 构造新mask：padding(0) + prefix(1) + valid(1)
                new_mask = torch.cat([
                    torch.zeros(pad_len, dtype=attention_masks.dtype, device=device),
                    torch.ones(10, dtype=attention_masks.dtype, device=device),
                    torch.ones(valid_len, dtype=attention_masks.dtype, device=device)
                ], dim=0)  # [pad_len + 10 + valid_len]

                new_embeds.append(new_embed)
                new_masks.append(new_mask)

            # 将列表转换为tensor
            input_embeds = torch.stack(new_embeds, dim=0)  # [B, T+10, H]
            attention_mask = torch.stack(new_masks, dim=0)  # [B, T+10]

            current_position_ids = self._build_position_ids(attention_mask)
            first_out = self._base_transformer()(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=current_position_ids,
                use_cache=False,
                return_dict=True,
            )
            first_hidden = first_out.last_hidden_state  # [B, T+10, H]

            # 定义后续需要的变量
            init_seq_len = input_embeds.size(1)  # 拼接prefix后的实际长度
            batch_idx = torch.arange(batch_size, device=device)
            # 保存当前的embeds和mask用于后续append
            current_embeds = input_embeds  # [B, T+10, H]
            current_attention_mask = attention_mask  # [B, T+10]
        else:
            device = input_ids.device
            attention_mask = attention_masks.to(device)

            self._assert_left_padding(attention_mask)

            batch_size, init_seq_len = input_ids.shape
            batch_idx = torch.arange(batch_size, device=device)

            # 每个样本原始有效长度 Li
            base_lengths = attention_mask.long().sum(dim=1)  # [B]

            # 第一次用原始序列跑，取原始最后一个有效 token 的 hidden
            init_position_ids = self._build_position_ids(attention_mask)

            first_out = self._base_transformer()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=init_position_ids,
                use_cache=False,
                return_dict=True,
            )
            first_hidden = first_out.last_hidden_state  # [B, T, H]

            # 保存当前的embeds和mask用于后续append
            current_embeds = self.encoder.get_input_embeddings()(input_ids)  # [B, T, H]
            current_attention_mask = attention_mask  # [B, T]

        # 显式定位原始最后一个有效 token
        init_last_indices = self._last_valid_indices(attention_mask)  # [B]

        # 在严格 left padding 下，它们都应当等于 T-1
        expected_last = torch.full_like(init_last_indices, init_seq_len - 1)
        if not torch.equal(init_last_indices, expected_last):
            raise RuntimeError(
                "Input is not strictly left-padded to a shared right boundary. "
                "Please ensure the tokenizer uses padding_side='left'."
            )

        last_hidden = first_hidden[batch_idx, init_last_indices]  # h_0, [B, H]
        all_appended_embeds = []
        all_appended_hiddens = [last_hidden]
        all_physical_new_indices = []
        all_semantic_new_position_ids = []

        for step in range(n_steps):
            # 1) 由上一步 hidden 产生新的连续 embedding
            appended_embed = F.normalize(self.encoder.base_model.model.model.norm(last_hidden), p=2, dim=-1)  # [B, H]
            softmax_logit =  F.softmax(self.act_fn(self.gate(appended_embed)), dim=-1)  # [B, V]
            appended_embed = self.ffn(softmax_logit)  # [B, H]

            all_appended_embeds.append(appended_embed)

            # 2) append 到末尾
            current_embeds = torch.cat(
                [current_embeds, appended_embed.unsqueeze(1)], dim=1
            )  # [B, T+step+1, H]

            extra_mask = torch.ones(
                batch_size, 1,
                dtype=current_attention_mask.dtype,
                device=device
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, extra_mask], dim=1
            )  # [B, T+step+1]

            current_position_ids = self._build_position_ids(current_attention_mask)

            # 3) 显式维护“当前新位置”的两个索引
            # 3.1 物理列下标：hidden_states 张量里新位置所在列
            physical_new_indices = torch.full(
                (batch_size,),
                current_embeds.size(1) - 1,
                dtype=torch.long,
                device=device
            )  # [B]

            # 3.2 语义 position_id：样本自己的真实序列位置
            # 原始长度 Li，第 1 次 append 的新位置是 Li，第 2 次是 Li+1 ...
            semantic_new_position_ids = base_lengths + step  # [B]

            # 校验：当前张量这一列上的 position_id，应当等于上面这个语义位置
            gathered_pos = current_position_ids[batch_idx, physical_new_indices]
            if not torch.equal(gathered_pos, semantic_new_position_ids):
                raise RuntimeError(
                    "Mismatch between physical appended indices and semantic position_ids."
                )

            all_physical_new_indices.append(physical_new_indices)
            all_semantic_new_position_ids.append(semantic_new_position_ids)
            # 4) 整段重跑
            step_out = self._base_transformer()(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                use_cache=False,
                return_dict=True,
            )
            step_hidden = step_out.last_hidden_state  # [B, T+step+1, H]

            # 5) 不再写死 -1，而是显式用当前新位置的物理列下标取
            last_hidden = step_hidden[batch_idx, physical_new_indices]  # h_t, [B, H]
            all_appended_hiddens.append(last_hidden)
        final_embeddings = all_appended_hiddens[-1] #torch.mean(torch.stack(all_appended_hiddens, dim=0), dim=0)

        # final_embeddings = torch.stack(all_appended_hiddens, dim=1)  # [B, T, H]

        if return_all:
            return {
                "final_hidden": last_hidden,                             # [B, H]
                "initial_last_hidden": first_hidden[batch_idx, init_last_indices],  # [B, H]
                "all_appended_embeds": all_appended_embeds,             # list of [B, H]
                "all_appended_hiddens": all_appended_hiddens,           # list of [B, H]
                "all_physical_new_indices": all_physical_new_indices,   # list of [B]
                "all_semantic_new_position_ids": all_semantic_new_position_ids,  # list of [B]
            }

        return final_embeddings


class EncoderWrapperSupervisedAppendNLT(nn.Module):
    def __init__(
        self,
        encoder,
        residual=False,
        init_from_decode_then_reembed=True,
        use_lm_head_bias=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.residual = residual
        hidden_size = encoder.base_model.model.lm_head.weight.shape[1]
        vocab_size = encoder.base_model.model.lm_head.weight.shape[0]
        self.gate = nn.Linear(hidden_size, vocab_size, bias=False)
        self.ffn = nn.Linear(vocab_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.global_step = 0
        self._init_gate_ffn()

    def _init_gate_ffn(self):
        with torch.no_grad():
            input_embed = self.encoder.get_input_embeddings().weight   # [V, H]
            lm_head = self.encoder.lm_head.weight                      # [V, H]
            self.gate.weight.copy_(lm_head)
            self.ffn.weight.copy_(input_embed.transpose(0, 1))

    @staticmethod
    def _assert_left_padding(attention_mask: torch.Tensor):
        """
        检查每一行是否满足 left padding:
            0 ... 0 1 ... 1
        """
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be 2D: [B, T]")

        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0/1")

        # left padding 要求 mask 单调不下降
        ok = torch.all(attention_mask[:, 1:] >= attention_mask[:, :-1])
        if not ok:
            raise ValueError(
                "This implementation requires LEFT padding: each row of attention_mask "
                "must look like 0...011...1."
            )

        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("Each sample must contain at least one valid token.")

    @staticmethod
    def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        left padding 下:
            mask = [0,0,1,1,1] -> pos = [0,0,0,1,2]
        pad 位置填 0，仅作占位，不参与注意力。
        """
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    @staticmethod
    def _last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        返回每个样本最后一个有效 token 的物理列下标。
        这个函数对任意 padding 都成立。
        """
        return attention_mask.size(1) - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1)

    def forward(
        self,
        input_ids,
        attention_masks,
        n_steps=3,
        return_all=False,
        query=False,
        dummy_tensor=None
    ):
        """
        严格支持:
            - batch
            - left padding
            - n 次连续 append
            - 整段重跑版

        参数:
            input_ids: [B, T]
            attention_masks: [B, T]
            n_steps: 连续追加多少次
            return_all: 是否返回每一步的中间结果

        返回:
            默认返回第 n 步追加后的最后 hidden, [B, H]

            若 return_all=True，则返回字典，包含:
                - final_hidden
                - initial_last_hidden
                - all_appended_embeds
                - all_appended_hiddens
                - all_physical_new_indices
                - all_semantic_new_position_ids
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        
        self.global_step += 1

        device = input_ids.device
        attention_mask = attention_masks.to(device)

        self._assert_left_padding(attention_mask)

        batch_size, init_seq_len = input_ids.shape
        batch_idx = torch.arange(batch_size, device=device)

        # 每个样本原始有效长度 Li
        base_lengths = attention_mask.long().sum(dim=1)  # [B]

        # 第一次用原始序列跑，取原始最后一个有效 token 的 hidden
        init_position_ids = self._build_position_ids(attention_mask)

        first_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=init_position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        first_hidden = first_out.hidden_states[-1]  # [B, T, H]

        # 显式定位原始最后一个有效 token
        init_last_indices = self._last_valid_indices(attention_mask)  # [B]

        # 在严格 left padding 下，它们都应当等于 T-1
        expected_last = torch.full_like(init_last_indices, init_seq_len - 1)
        if not torch.equal(init_last_indices, expected_last):
            raise RuntimeError(
                "Input is not strictly left-padded to a shared right boundary. "
                "Please ensure the tokenizer uses padding_side='left'."
            )

        last_hidden = first_hidden[batch_idx, init_last_indices]  # h_0, [B, H]

        current_embeds = self.encoder.get_input_embeddings()(input_ids)  # [B, T, H]
        current_attention_mask = attention_mask
        all_appended_embeds = []
        all_appended_hiddens = [last_hidden]
        all_physical_new_indices = []
        all_semantic_new_position_ids = []

        for step in range(n_steps):
            # 1) 由上一步 hidden 产生新的连续 embedding
            appended_embed = F.normalize(self.encoder.base_model.model.model.norm(last_hidden), p=2, dim=-1)  # [B, H]
            softmax_logit =  F.softmax(self.act_fn(self.gate(appended_embed)), dim=-1)  # [B, V]
            appended_embed = self.ffn(softmax_logit)  # [B, H]

            all_appended_embeds.append(appended_embed)

            # 2) append 到末尾
            current_embeds = torch.cat(
                [current_embeds, appended_embed.unsqueeze(1)], dim=1
            )  # [B, T+step+1, H]

            extra_mask = torch.ones(
                batch_size, 1,
                dtype=current_attention_mask.dtype,
                device=device
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, extra_mask], dim=1
            )  # [B, T+step+1]

            current_position_ids = self._build_position_ids(current_attention_mask)

            # 3) 显式维护“当前新位置”的两个索引
            # 3.1 物理列下标：hidden_states 张量里新位置所在列
            physical_new_indices = torch.full(
                (batch_size,),
                current_embeds.size(1) - 1,
                dtype=torch.long,
                device=device
            )  # [B]

            # 3.2 语义 position_id：样本自己的真实序列位置
            # 原始长度 Li，第 1 次 append 的新位置是 Li，第 2 次是 Li+1 ...
            semantic_new_position_ids = base_lengths + step  # [B]

            # 校验：当前张量这一列上的 position_id，应当等于上面这个语义位置
            gathered_pos = current_position_ids[batch_idx, physical_new_indices]
            if not torch.equal(gathered_pos, semantic_new_position_ids):
                raise RuntimeError(
                    "Mismatch between physical appended indices and semantic position_ids."
                )

            all_physical_new_indices.append(physical_new_indices)
            all_semantic_new_position_ids.append(semantic_new_position_ids)
            # 4) 整段重跑
            step_out = self.encoder(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            step_hidden = step_out.hidden_states[-1]  # [B, T+step+1, H]

            # 5) 不再写死 -1，而是显式用当前新位置的物理列下标取
            last_hidden = step_hidden[batch_idx, physical_new_indices]  # h_t, [B, H]
            all_appended_hiddens.append(last_hidden)
        final_embeddings = all_appended_hiddens
        # final_embeddings = torch.mean(torch.stack(all_appended_hiddens, dim=0), dim=0)
        if return_all:
            return {
                "final_hidden": last_hidden,                             # [B, H]
                "initial_last_hidden": first_hidden[batch_idx, init_last_indices],  # [B, H]
                "all_appended_embeds": all_appended_embeds,             # list of [B, H]
                "all_appended_hiddens": all_appended_hiddens,           # list of [B, H]
                "all_physical_new_indices": all_physical_new_indices,   # list of [B]
                "all_semantic_new_position_ids": all_semantic_new_position_ids,  # list of [B]
            }

        return final_embeddings


class EncoderWrapperSupervisedMeanPooling(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        # attention_masks: [batch, seq_len]，1 为有效 token，0 为 padding
        attention_mask = attention_masks.to(input_ids.device)

        # 调用底层编码器，VaLlamaForCausalLM 需要保证返回 all_values
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 你之前就是这样写的，保持不动
        )

        embedding = output.hidden_states[-1] #[B, T, H]

        # 将 attention_mask 扩展到 [B, T, 1]，并转成和 embedding 相同的 dtype
        mask = attention_mask.unsqueeze(-1).type_as(embedding)  # [B, T, 1]

        # 对 padding 位置置零，再对时间维求和
        masked_embeddings = embedding * mask  # [B, T, H]
        sum_embeddings = masked_embeddings.sum(dim=1)  # [B, H]

        # 有效 token 个数，用于做平均
        lengths = mask.sum(dim=1)  # [B, 1]
        lengths = lengths.clamp(min=1.0)  # 避免除以 0

        mean_embeddings = sum_embeddings / lengths  # [B, H]

        # 不在这里做归一化，在外面再 normalize
        return mean_embeddings



class EncoderWrapperVAPPT(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        # attention_masks: [batch, seq_len]，1 为有效 token，0 为 padding
        attention_mask = attention_masks.to(input_ids.device)
        ppt_begin_pos = attention_mask.sum(dim=1, keepdim=True)
        ppt_end_pos = attention_mask.sum(dim=1, keepdim=True) + 56
        # 调用底层编码器，VaLlamaForCausalLM 需要保证返回 all_values
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  
            ppt_begin_pos=ppt_begin_pos,
            ppt_end_pos=ppt_end_pos
        )
        last16 = (output.all_va_outputs.permute(0, 1, 3, 2, 4).reshape(28, attention_masks.size(0), attention_masks.size(1) + 64, 28*128))[-14:]
        value = torch.stack(output.va_value_states, dim=0).permute(0, 1, 3, 2, 4).reshape(32, attention_masks.size(0), attention_masks.size(1) + 64, 32*128)[-14:]
        all_layer_embeddings = []
        for j in range(ppt_begin_pos.shape[0]):
            now_begin = ppt_begin_pos[j][0] + 28
            now_embeddings = []
            value_embeddings = []
            for i in range(14):
                now_embeddings.append(last16[i, j, i + now_begin])
                value_embeddings.append(value[i, j, i + now_begin + 14])
            all_layer_embeddings.append(torch.mean(torch.stack(now_embeddings, dim=0), dim=0) + torch.mean(torch.stack(value_embeddings, dim=0), dim=0))
        return torch.stack(all_layer_embeddings, dim=0)



class Value_Aggregation_Gather(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if "llama" in model_name:
            self.model = VaLlamaForCausalLM.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        else:
            self.model = VaQwen3ForCausalLM.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def model_wrapper(self):
        # 用 EncoderWrapperSupervised 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervised(self.model)
        self.encoder_gpu_train_limit = 16
        self.temperature = 0.05
        self.scale = 1

    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervised
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask,
        positive_attention_mask,
        negative_attention_mask,
    ):
        query_embedding = self._encode_in_chunks(
            query_input_ids, query_attention_mask
        )
        positive_embedding = self._encode_in_chunks(
            positive_input_ids, positive_attention_mask
        )
        negative_embedding = self._encode_in_chunks(
            negative_input_ids, negative_attention_mask
        )

        # E5 / InfoNCE 风格：句向量 L2 归一化，用归一化后的点积当余弦
        query_weight_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
        positive_weight_embedding_norm = F.normalize(positive_embedding, p=2, dim=-1)
        negative_weight_embedding_norm = F.normalize(negative_embedding, p=2, dim=-1)

        return (
            query_weight_embedding_norm,
            positive_weight_embedding_norm,
            negative_weight_embedding_norm,
        )




class VAPPT_Gather(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if "llama" in model_name:
            self.model = VaPPTLlamaForCausalLM.from_pretrained(
                model_name,
                o_layer=o_layer,
                use_auth_token=use_auth_token
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        else:
            self.model = VaPPTQwenForCausalLM.from_pretrained(
                model_name,
                o_layer=o_layer,
                use_auth_token=use_auth_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        self.model.initialize_prompt(64)
        self.o_layer = o_layer

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def model_wrapper(self):
        # 用 EncoderWrapperSupervised 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperVAPPT(self.model)
        self.encoder_gpu_train_limit = 16
        self.temperature = 0.05
        self.scale = 1

    def _encode_in_chunks_vappt(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervised
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask,
        positive_attention_mask,
        negative_attention_mask,
    ):
        query_embedding = self._encode_in_chunks_vappt(
            query_input_ids, query_attention_mask
        )
        positive_embedding = self._encode_in_chunks_vappt(
            positive_input_ids, positive_attention_mask
        )
        negative_embedding = self._encode_in_chunks_vappt(
            negative_input_ids, negative_attention_mask
        )

        # E5 / InfoNCE 风格：句向量 L2 归一化，用归一化后的点积当余弦
        query_weight_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
        positive_weight_embedding_norm = F.normalize(positive_embedding, p=2, dim=-1)
        negative_weight_embedding_norm = F.normalize(negative_embedding, p=2, dim=-1)

        return (
            query_weight_embedding_norm,
            positive_weight_embedding_norm,
            negative_weight_embedding_norm,
        )

class Value_Aggregation_Eval(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if "llama" in model_name:
            self.model = VaLlamaForCausalLM.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        else:
            self.model = VaQwen3ForCausalLM.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        # 给 backbone 注入 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper()

    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervised
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def model_wrapper(self):
        # 用 EncoderWrapperSupervised 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervised(self.model)
        self.encoder_gpu_train_limit = 16
        self.temperature = 0.05
        self.scale = 1


    def forward(
        self,
        input_ids,
        attention_mask
    ):
        current_embedding = self._encode_in_chunks(
            input_ids, attention_mask
        )
        return current_embedding
    


class InforNCE_and_Eigenvalue(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if model_name is None:
            model_name = "Qwen/Qwen3.5-0.8B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        # 给 backbone 注入 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper()
    
    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervisedLastToken
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
                use_reentrant=True
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def model_wrapper(self):
        # 用 EncoderWrapperSupervisedLastToken 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervisedLastToken(self.model)
        self.encoder_gpu_train_limit = 32
        self.temperature = 0.05
        self.scale = 1

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask,
        positive_attention_mask,
        negative_attention_mask,
    ):
        query_embedding = self._encode_in_chunks(
            query_input_ids, query_attention_mask
        )
        positive_embedding = self._encode_in_chunks(
            positive_input_ids, positive_attention_mask
        )
        negative_embedding = self._encode_in_chunks(
            negative_input_ids, negative_attention_mask
        )
        # E5 / InfoNCE 风格：句向量 L2 归一化，用归一化后的点积当余弦
        query_weight_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
        positive_weight_embedding_norm = F.normalize(positive_embedding, p=2, dim=-1)
        negative_weight_embedding_norm = F.normalize(negative_embedding, p=2, dim=-1)

        return (
            query_weight_embedding_norm,
            positive_weight_embedding_norm,
            negative_weight_embedding_norm,
        )


class InforNCE_and_Eigenvalue_Eval(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if model_name is None:
            model_name = "Qwen/Qwen3.5-0.8B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        # 给 backbone 注入 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper()

    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervised
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def model_wrapper(self):
        # 用 EncoderWrapperSupervised 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervisedLastToken(self.model)
        self.encoder_gpu_train_limit = 16
        self.temperature = 0.05
        self.scale = 1


    def forward(
        self,
        input_ids,
        attention_mask
    ):
        current_embedding = self._encode_in_chunks(
            input_ids, attention_mask
        )
        return current_embedding




class Plain_Model_Evaluation(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if model_name is None:
            model_name = "Qwen/Qwen3.5-0.8B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.model_wrapper()
    
    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervisedLastToken
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
                use_reentrant=True
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def model_wrapper(self):
        # 用 EncoderWrapperSupervisedLastToken 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervisedLastToken(self.model)
        self.encoder_gpu_train_limit = 32
        self.temperature = 0.05
        self.scale = 1

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask,
        positive_attention_mask,
        negative_attention_mask,
    ):
        query_embedding = self._encode_in_chunks(
            query_input_ids, query_attention_mask
        )
        positive_embedding = self._encode_in_chunks(
            positive_input_ids, positive_attention_mask
        )
        negative_embedding = self._encode_in_chunks(
            negative_input_ids, negative_attention_mask
        )

        # E5 / InfoNCE 风格：句向量 L2 归一化，用归一化后的点积当余弦
        query_weight_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
        positive_weight_embedding_norm = F.normalize(positive_embedding, p=2, dim=-1)
        negative_weight_embedding_norm = F.normalize(negative_embedding, p=2, dim=-1)

        return (
            query_weight_embedding_norm,
            positive_weight_embedding_norm,
            negative_weight_embedding_norm,
        )



class InforNCE_and_Generative_Hops(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if model_name is None:
            model_name = "Qwen/Qwen3.5-0.8B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        lora_config = LoraConfig(
            r=32,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        # 给 backbone 注入 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper_update()
    
    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]
            # input_ids, attention_masks, n_steps, return_all, query, dummy_tensor
            sub_emb = checkpoint(
                self.model,  
                sub_input_ids,
                sub_attention_mask,
                4,
                False,
                False,
                dummy_tensor,
                use_reentrant=True
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def _encode_in_chunks_query(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]
            # input_ids, attention_masks, n_steps, return_all, query, dummy_tensor
            sub_emb = checkpoint(
                self.model,  
                sub_input_ids,
                sub_attention_mask,
                4,
                False,
                True,
                dummy_tensor,
                use_reentrant=True
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]
    
    def model_wrapper_origin(self):
        # 用 EncoderWrapperSupervisedLastToken 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervisedLastToken(self.model)
        self.encoder_gpu_train_limit = 8
        self.scale = 1

    def model_wrapper_update(self):
        # 用 EncoderWrapperSupervisedLastToken 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervisedAppendN(self.model)
        self.encoder_gpu_train_limit = 8
        self.scale = 1

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask,
        positive_attention_mask,
        negative_attention_mask,
    ):
        query_embedding = self._encode_in_chunks_query(
            query_input_ids, query_attention_mask
        )
        positive_embedding = self._encode_in_chunks(
            positive_input_ids, positive_attention_mask
        )
        negative_embedding = self._encode_in_chunks(
            negative_input_ids, negative_attention_mask
        )
        # E5 / InfoNCE 风格：句向量 L2 归一化，用归一化后的点积当余弦
        query_weight_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
        positive_weight_embedding_norm = F.normalize(positive_embedding, p=2, dim=-1)
        negative_weight_embedding_norm = F.normalize(negative_embedding, p=2, dim=-1)

        return (
            query_weight_embedding_norm,
            positive_weight_embedding_norm,
            negative_weight_embedding_norm,
        )




class InforNCE_and_Generative_Hops_Eval(nn.Module):
    def __init__(self, local_rank=0, model_name=None) -> None:
        super().__init__()
        if model_name is None:
            model_name = "Qwen/Qwen3.5-0.8B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        self.model.config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        lora_config = LoraConfig(
            r=32,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        # 给 backbone 注入 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper_update()
    def load_checkpoint(self, checkpoint_path):
        """加载DeepSpeed保存的模型checkpoint
        
        Args:
            checkpoint_path: DeepSpeed保存的checkpoint路径
        """
        import glob
        import os
        special_tokens = ["[STOP_SEARCH]", "[SUFFICIENT_EVIDENCE]", "[ANSWER_READY]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.encoder.base_model.model.resize_token_embeddings(len(self.tokenizer))
        # 查找mp_rank_00_model_states.pt文件
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "mp_rank_*_model_states.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
        
        # 加载第一个找到的checkpoint文件
        checkpoint_file = checkpoint_files[0]
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
                
        # 加载模型权重
        self.load_state_dict(checkpoint["module"], strict=True)
        print(f"Loaded checkpoint from {checkpoint_file}")
    
    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块；eval 推理不需要 checkpoint。
        """
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]
            sub_emb = self.model(
                sub_input_ids,
                sub_attention_mask,
                4,
                False,
                False,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def _encode_in_chunks_query(self, input_ids, attention_mask):
        """
        手动按 batch 维切块；eval 推理不需要 checkpoint。
        """
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]
            sub_emb = self.model(
                sub_input_ids,
                sub_attention_mask,
                4,
                False,
                True,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]
        
    def model_wrapper_origin(self):
        # 用 EncoderWrapperSupervisedLastToken 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervisedLastToken(self.model)
        self.encoder_gpu_train_limit = 1
        self.scale = 1

    def model_wrapper_update(self):
        # 用 EncoderWrapperSupervisedLastToken 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervisedAppendN(self.model)
        self.encoder_gpu_train_limit = 1
        self.scale = 1

    def forward(
        self,
        input_ids,
        attention_mask,
        is_query=False
    ):
        if is_query:
            current_embedding = self._encode_in_chunks_query(
                input_ids, attention_mask
            )
        else:
            current_embedding = self._encode_in_chunks(
                input_ids, attention_mask
            )
        return current_embedding