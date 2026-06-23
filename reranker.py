import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional

class Reranker:
    """
    用于多跳问答的重排序器，基于 Qwen3-Reranker-4B。
    对给定的子问题和文档列表计算相关性得分，返回按得分升序（最不相关在前）的文档子集。
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        instruction: Optional[str] = None,
        max_length: int = 500,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda:3"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2"
            attn_implementation="eager"
        ).to(self.device).eval()

        # 针对多跳子问题的指令，可自定义
        if instruction is None:
            instruction = "Given a sub-question of a multi-hop QA task, retrieve relevant passages that help answer the sub-question."
        self.instruction = instruction

        self.max_length = max_length
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # 对话模板的 prefix 和 suffix
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_instruction(self, query: str, doc: str) -> str:
        """将指令、查询和文档拼接成模型输入格式。"""
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """对输入文本进行分词、拼接 prefix/suffix、填充并移至设备。"""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def _compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """计算每个文档的 yes/no 概率，返回 yes 的概率作为相关性得分。"""
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits

        true_scores = logits[:, self.token_true_id]
        false_scores = logits[:, self.token_false_id]
        stacked = torch.stack([false_scores, true_scores], dim=1)
        probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = probs[:, 1].exp().tolist()  # yes 的概率
        return scores

    def rerank(
        self,
        question: str,
        documents: List[Dict[str, str]],
        t: float = 1.0,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        对文档进行重排序，并返回指定比例的文档。

        参数:
            question: 子问题（字符串）
            documents: 文档列表，每个元素为 {"id": str, "text": str}
            t: 保留比例，0 < t <= 1。t=1 返回所有文档，否则返回前 topk * t 个最相关的文档。
            batch_size: 批处理大小。None 表示并行处理所有文档，1 表示逐个处理（用于性能测试）。

        返回:
            按得分升序（最不相关在前，最相关在最后）排列的文档列表。
        """
        if not documents:
            return []

        if batch_size == 1:
            scores = []
            for i, doc in enumerate(documents):
                pair = self._format_instruction(question, doc["text"])
                inputs = self._process_inputs([pair])
                score = self._compute_logits(inputs)[0]
                scores.append(score)
        else:
            pairs = [
                self._format_instruction(question, doc["text"])
                for doc in documents
            ]
            inputs = self._process_inputs(pairs)
            scores = self._compute_logits(inputs)

        # 将得分附加到文档上（避免修改原始对象）
        scored_docs = [
            {**doc, "score": score}
            for doc, score in zip(documents, scores)
        ]

        # 按得分升序排序（低分在前，高分在后）
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # 按比例保留最相关的部分（即列表末尾的文档）
        topk = len(scored_docs)
        keep = int(topk * t)
        keep = max(1, keep) if t > 0 else 0  # 至少保留一个（若 t>0）
        result = scored_docs[:keep] if keep > 0 else []

        # 移除内部 score 字段（可选）
        # for doc in result:
        #     doc.pop("score", None)

        return result
    
    def rerank_topk(
        self,
        question: str,
        documents: List[Dict[str, str]],
        top_k: int = 20,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        对文档进行重排序，并返回指定比例的文档。

        参数:
            question: 子问题（字符串）
            documents: 文档列表，每个元素为 {"id": str, "text": str}
            top_k: 保留的最大文档数，默认是 20 。
            batch_size: 批处理大小。None 表示并行处理所有文档，1 表示逐个处理（用于性能测试）。

        返回:
            按得分升序（最不相关在前，最相关在最后）排列的文档列表。
        """
        if not documents:
            return []

        if batch_size == 1:
            scores = []
            for i, doc in enumerate(documents):
                pair = self._format_instruction(question, doc["text"])
                inputs = self._process_inputs([pair])
                score = self._compute_logits(inputs)[0]
                scores.append(score)
        else:
            pairs = [
                self._format_instruction(question, doc["text"])
                for doc in documents
            ]
            inputs = self._process_inputs(pairs)
            scores = self._compute_logits(inputs)

        # 将得分附加到文档上（避免修改原始对象）
        scored_docs = [
            {**doc, "score": score}
            for doc, score in zip(documents, scores)
        ]

        # 按得分升序排序（低分在前，高分在后）
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # 按比例保留最相关的部分（即列表末尾的文档）
        
        keep = top_k
        result = scored_docs[:keep] if keep > 0 else []

        # 移除内部 score 字段（可选）
        # for doc in result:
        #     doc.pop("score", None)

        return result