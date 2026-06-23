import os

import json
import numpy as np
import torch
import nltk
from collections import Counter
from typing import Optional, Union, Tuple, List, Dict
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM, 
    apply_rotary_pos_emb, 
    repeat_kv, 
    eager_attention_forward
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

# Monkey Patch 
def DecoderLayer_forward_fixed(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[Cache] = None, use_cache: Optional[bool] = False, cache_position: Optional[torch.LongTensor] = None, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, output_attentions: Optional[bool] = False, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, output_attentions=output_attentions, **kwargs)
    if output_attentions and self_attn_weights is not None:
        self_attn_weights = self_attn_weights[..., -1:, :]
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
        return outputs
    return hidden_states

def Model_forward_fixed(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[Cache] = None, inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
    if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None: position_ids = cache_position.unsqueeze(0)
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {"config": self.config, "input_embeds": inputs_embeds, "attention_mask": attention_mask, "cache_position": cache_position, "past_key_values": past_key_values, "position_ids": position_ids}
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers: causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    hidden_states = inputs_embeds
    output_attentions = kwargs.pop("output_attentions", self.config.output_attentions)
    all_attentions = () if output_attentions else None
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    max_layers = getattr(self, "early_exit_layer", self.config.num_hidden_layers)
    for decoder_layer in self.layers[: max_layers]:
        if output_attentions:
            layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask_mapping[decoder_layer.attention_type], position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, output_attentions=True, **kwargs)
            hidden_states = layer_outputs[0]
            all_attentions += (layer_outputs[1],) 
        else:
            hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask_mapping[decoder_layer.attention_type], position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, output_attentions=False, **kwargs)
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None, attentions=all_attentions)

def Qwen3Attention_forward_fixed(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], past_key_values: Optional[Cache] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    output_attentions = kwargs.pop("output_attentions", False)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager": attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
    
    if output_attentions:
        q_last = query_states[:, :, -1:, :]
        k_repeated = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights_last = torch.matmul(q_last, k_repeated.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                causal_mask = attention_mask[:, :, -1:, : key_states.shape[-2]]
                attn_weights_last = attn_weights_last + causal_mask
            elif attention_mask.dim() == 2:
                is_pad = (attention_mask == 0)
                pad_mask = torch.zeros_like(attention_mask, dtype=attn_weights_last.dtype)
                pad_mask.masked_fill_(is_pad, torch.finfo(attn_weights_last.dtype).min)
                pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
                attn_weights_last = attn_weights_last + pad_mask
        attn_weights = torch.nn.functional.softmax(attn_weights_last, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = Qwen3Attention_forward_fixed
transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer.forward = DecoderLayer_forward_fixed
transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward = Model_forward_fixed

class Compressor:
    def __init__(self, model_path: str, heads_file: str, device: str = "cuda:3"):
        self.model_path = model_path
        self.device = torch.device(device) if isinstance(device, str) else device

        self.head_weights = self._load_and_normalize_heads(heads_file, top_n=10)
        self.target_heads_indices = list(self.head_weights.keys())

        print(f"[EffRAGCompressor] Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        self.model = Qwen3ForCausalLM.from_pretrained(
            self.model_path, 
            device_map={"": 3},
            dtype=torch.bfloat16, 
            trust_remote_code=True, 
            # attn_implementation="flash_attention_2" 
        ).eval()
        
        # early exit
        max_needed_layer = max(layer_idx for (layer_idx, head_idx) in self.target_heads_indices)
        self.model.model.early_exit_layer = max_needed_layer + 1

    def _load_and_normalize_heads(self, heads_file: str, top_n: int):
        with open(heads_file, 'r', encoding='utf-8') as f:
            heads_list = json.load(f)
        heads_list.sort(key=lambda x: x['score'], reverse=True)
        top_heads = heads_list[:top_n]
        raw_scores = np.array([h['score'] for h in top_heads])
        normalized_weights = raw_scores / raw_scores.sum()
        head_weights_map = {}
        for idx, h in enumerate(top_heads):
            head_weights_map[(h['layer'], h['head'])] = normalized_weights[idx]
        return head_weights_map

    def compress(self, query: str, docs: List[Dict[str, str]], threshold: float = 0.2) -> List[Dict[str, str]]:
        if not docs:
            return []

        input_ids = []
        id_map = []
        sent_lengths = {} 
        current_sent_metadata = {} 
        doc_titles = {} 
        
        system_header = "Context information is below.\n---------------------\n"
        header_tokens = self.tokenizer.encode(system_header, add_special_tokens=False)
        input_ids.extend(header_tokens)
        id_map.extend([-1] * len(header_tokens))
        
        global_sent_id = 0
        
        for doc_idx, doc in enumerate(docs):
            raw_text = doc.get('text', '')
            
            # Assume the format is like "Title. Context...", we split it by the first ". " as the boundary
            parts = raw_text.split('. ', 1)
            if len(parts) == 2:
                title = parts[0] + "." # Add the period back to the title if it's missing
                context = parts[1].strip()
            else:
                # Handle cases where there's no period in the input
                title = ""
                context = raw_text
                
            # Store the title in the dictionary for later use
            doc_titles[doc_idx] = title
            
            # Tokenize the context
            full_context = f"{context} "
            sentences = nltk.sent_tokenize(full_context)
            
            for sent_text in sentences:
                chunk_tokens = self.tokenizer.encode(" " + sent_text, add_special_tokens=False)
                chunk_len = len(chunk_tokens)
                if chunk_len == 0: continue

                sent_lengths[global_sent_id] = chunk_len
                input_ids.extend(chunk_tokens)
                id_map.extend([global_sent_id] * chunk_len)
                
                current_sent_metadata[global_sent_id] = {
                    "text": sent_text,
                    "doc_idx": doc_idx
                }
                global_sent_id += 1

        system_footer = "\n---------------------\nAnswer the question based on the given context. Only give me the answer and do not output any other words."
        user_part = f"Query: {query}\nAnswer: "
        suffix_tokens = self.tokenizer.encode(system_footer + user_part, add_special_tokens=False)
        input_ids.extend(suffix_tokens)
        id_map.extend([-2] * len(suffix_tokens))
        
        input_tensor = torch.tensor([input_ids]).to(self.device)
        mask_tensor = torch.ones_like(input_tensor).to(self.device)
        id_map_array = np.array(id_map)

        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_tensor, 
                attention_mask=mask_tensor, 
                output_attentions=True
            )
            
        actual_seq_len = len(input_ids)
        k_val = max(1, actual_seq_len // 8)
        vote_scores = Counter()
        
        for (layer_idx, head_idx) in self.target_heads_indices:
            head_importance = self.head_weights[(layer_idx, head_idx)]
            attn_vector = outputs.attentions[layer_idx][0, head_idx, -1, :] 
            curr_k = min(k_val, attn_vector.shape[0])
            
            _, top_indices = torch.topk(attn_vector, k=curr_k)
            top_indices = top_indices.cpu().numpy()
            
            hit_ids = id_map_array[top_indices]
            for sent_id in hit_ids:
                if sent_id >= 0: 
                    vote_scores[sent_id] += 1.0 * head_importance
                    
        del outputs, input_tensor, mask_tensor
        torch.cuda.empty_cache()

        normalized_scores = {}
        for sent_id, raw_weighted_sum in vote_scores.items():
            length = sent_lengths.get(int(sent_id), 1)
            normalized_scores[int(sent_id)] = raw_weighted_sum / length

        doc_sentences_map = {i: [] for i in range(len(docs))}
        for sent_id, score in normalized_scores.items():
            if score >= threshold:
                meta = current_sent_metadata[sent_id]
                doc_idx = meta['doc_idx']
                doc_sentences_map[doc_idx].append({
                    "sentence_id": sent_id,
                    "text": meta['text']
                })

        compressed_docs = []
        for doc_idx, original_doc in enumerate(docs):
            hits = doc_sentences_map[doc_idx]
            hits.sort(key=lambda x: x['sentence_id'])
            
            merged_context = " ".join([h['text'] for h in hits]).strip()
            title = doc_titles.get(doc_idx, "")
            
            if merged_context:
                final_text = f"{title} {merged_context}".strip() if title else merged_context
                compressed_docs.append({
                    "id": original_doc.get("id", ""),
                    "text": final_text
                })
            else:
                compressed_docs.append({
                    "id": original_doc.get("id", ""),
                    "text": ""
                })

        return compressed_docs