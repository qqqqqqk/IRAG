import json
import os
import re
from collections import defaultdict
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, HfArgumentParser
from tqdm import tqdm
def _normalize_text(text: str) -> str:
    """
    Normalize text for loose title/entity matching.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\([^)]*\)", "", text)  # remove parenthetical disambiguation
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ordered_unique(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d and d[key] is not None else default


def _lexical_overlap_score(a: str, b: str) -> int:
    """
    A very simple lexical overlap score used only as a fallback.
    """
    sa = set(_normalize_text(a).split())
    sb = set(_normalize_text(b).split())
    if not sa or not sb:
        return 0
    return len(sa & sb)


def _build_2wiki_context(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Support both:
    1. Hotpot-style dict:
       context = {"title": [...], "sentences": [[...], [...], ...]}
    2. Original list-style:
       context = [[title, [sent1, sent2, ...]], ...]
    """
    context = item.get("context", {})
    paragraphs: List[Dict[str, Any]] = []

    if isinstance(context, dict):
        titles = _safe_get(context, "title", [])
        sentences = _safe_get(context, "sentences", [])
        for idx, title in enumerate(titles):
            sents = sentences[idx] if idx < len(sentences) else []
            if isinstance(sents, list):
                text = " ".join(sents)
            else:
                text = str(sents)
            paragraphs.append(
                {
                    "idx": idx,
                    "title": title,
                    "sentences": sents if isinstance(sents, list) else [str(sents)],
                    "paragraph_text": text,
                }
            )
    elif isinstance(context, list):
        for idx, elem in enumerate(context):
            if isinstance(elem, (list, tuple)) and len(elem) == 2:
                title, sents = elem
                if isinstance(sents, list):
                    text = " ".join(sents)
                else:
                    text = str(sents)
                paragraphs.append(
                    {
                        "idx": idx,
                        "title": title,
                        "sentences": sents if isinstance(sents, list) else [str(sents)],
                        "paragraph_text": text,
                    }
                )
    else:
        raise ValueError("Unsupported 2Wiki context format.")

    return paragraphs


def _resolve_entity_to_context_idx(
    entity: str,
    context_titles: List[str],
    support_titles: Optional[List[str]] = None,
) -> Optional[int]:
    """
    Resolve an entity string from evidences to a context paragraph index.

    Matching priority:
    1. exact normalized title match
    2. exact match against support title, then map to context
    3. containment-based loose match
    """
    if not entity:
        return None

    norm_entity = _normalize_text(entity)
    if not norm_entity:
        return None

    norm_context = [_normalize_text(t) for t in context_titles]

    # 1. Exact normalized match in context titles
    exact_hits = [i for i, t in enumerate(norm_context) if t == norm_entity]
    if len(exact_hits) == 1:
        return exact_hits[0]

    # 2. Exact normalized match in supporting titles, then map back to context
    if support_titles is not None:
        norm_support = [_normalize_text(t) for t in support_titles]
        matched_support = [t for t, nt in zip(support_titles, norm_support) if nt == norm_entity]
        if len(matched_support) == 1:
            target = _normalize_text(matched_support[0])
            hits = [i for i, t in enumerate(norm_context) if t == target]
            if len(hits) == 1:
                return hits[0]

    # 3. Loose containment match
    loose_hits = []
    for i, t in enumerate(norm_context):
        if norm_entity in t or t in norm_entity:
            loose_hits.append(i)
    if len(loose_hits) == 1:
        return loose_hits[0]

    return None


def _infer_chain_from_edges(
    support_indices: List[int],
    edges: List[Tuple[int, int]],
    question: str,
    titles: List[str],
) -> Optional[List[int]]:
    """
    Try to infer a single chain order from directed edges over support docs.
    """
    support_set = set(support_indices)

    adj = defaultdict(list)
    indeg = {i: 0 for i in support_indices}
    outdeg = {i: 0 for i in support_indices}

    for u, v in edges:
        if u in support_set and v in support_set and u != v:
            if v not in adj[u]:
                adj[u].append(v)
                indeg[v] += 1
                outdeg[u] += 1

    # Case 1: graph suggests a unique start node
    starts = [i for i in support_indices if indeg[i] == 0]
    if len(starts) == 1:
        order = []
        cur = starts[0]
        visited = set()

        while cur not in visited:
            order.append(cur)
            visited.add(cur)
            next_nodes = [x for x in adj[cur] if x not in visited]
            if len(next_nodes) == 1:
                cur = next_nodes[0]
            else:
                break

        if set(order) == support_set and len(order) == len(support_indices):
            return order

    # Case 2: lexical fallback for choosing a start
    scored = sorted(
        support_indices,
        key=lambda i: _lexical_overlap_score(question, titles[i]),
        reverse=True,
    )
    if scored:
        start = scored[0]
        order = [start]
        visited = {start}
        cur = start

        while True:
            next_nodes = [x for x in adj[cur] if x not in visited]
            if len(next_nodes) == 1:
                cur = next_nodes[0]
                order.append(cur)
                visited.add(cur)
            else:
                break

        # If chain inference is partial, append the remaining support docs
        # by descending lexical overlap as a weak fallback.
        remaining = [i for i in support_indices if i not in visited]
        remaining = sorted(
            remaining,
            key=lambda i: _lexical_overlap_score(question, titles[i]),
            reverse=True,
        )
        order.extend(remaining)

        if len(order) == len(support_indices):
            return order

    return None


def process_2wiki_item(
    item: Dict[str, Any],
    comparison_mode: str = "all_permutations",
) -> Dict[str, Any]:
    """
    Convert one 2Wiki item into a unified training example.

    Returns:
        {
            "id": ...,
            "query": ...,
            "answer": ...,
            "question_type": ...,
            "candidates": [...],
            "candidate_texts": [...],
            "positive_set": [...],
            "hop_order": [... ] or None,
            "valid_orders": [[...], [...]] or None,
            "order_source": "path_induced" | "permutation" | "unordered"
        }

    Notes:
    - For comparison / bridge_comparison questions, unique order is often not reliable.
    - In that case, this function returns all permutations of support docs by default.
    """
    paragraphs = _build_2wiki_context(item)
    titles = [p["title"] for p in paragraphs]
    texts = [p["paragraph_text"] for p in paragraphs]

    sf = item.get("supporting_facts", {})
    support_titles = _ordered_unique(_safe_get(sf, "title", []))

    title_to_indices = defaultdict(list)
    for idx, title in enumerate(titles):
        title_to_indices[_normalize_text(title)].append(idx)

    positive_set = []
    for title in support_titles:
        hits = title_to_indices.get(_normalize_text(title), [])
        if len(hits) >= 1:
            positive_set.append(hits[0])
    positive_set = _ordered_unique(positive_set)

    question_type = str(item.get("type", "")).lower()
    question = item.get("question", "")

    # Comparison-like questions: do not force a single order.
    if "comparison" in question_type:
        if comparison_mode == "all_permutations" and 1 <= len(positive_set) <= 6:
            valid_orders = [list(p) for p in permutations(positive_set)]
        else:
            valid_orders = None

        return {
            "id": item.get("id"),
            "query": question,
            "answer": item.get("answer"),
            "question_type": question_type,
            "candidates": titles,
            "candidate_texts": texts,
            "positive_set": positive_set,
            "hop_order": None,
            "valid_orders": valid_orders,
            "order_source": "permutation" if valid_orders is not None else "unordered",
        }

    # Try to induce order from evidences
    evidences = item.get("evidences", [])
    edges: List[Tuple[int, int]] = []

    for ev in evidences:
        if not isinstance(ev, (list, tuple)) or len(ev) < 3:
            continue

        subj, _, obj = ev[0], ev[1], ev[2]
        u = _resolve_entity_to_context_idx(subj, titles, support_titles)
        v = _resolve_entity_to_context_idx(obj, titles, support_titles)

        if u is not None and v is not None and u != v:
            edges.append((u, v))

    hop_order = None
    order_source = "unordered"

    if positive_set:
        inferred = _infer_chain_from_edges(
            support_indices=positive_set,
            edges=edges,
            question=question,
            titles=titles,
        )
        if inferred is not None:
            hop_order = inferred
            order_source = "path_induced"

    return {
        "id": item.get("id"),
        "query": question,
        "answer": item.get("answer"),
        "question_type": question_type,
        "candidates": titles,
        "candidate_texts": texts,
        "positive_set": positive_set,
        "hop_order": hop_order,
        "valid_orders": [hop_order] if hop_order is not None else None,
        "order_source": order_source,
    }


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON list file or a JSONL file.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_nonempty = None
        for line in f:
            if line.strip():
                first_nonempty = line.lstrip()
                break

    if first_nonempty is None:
        return []

    # JSON list/object
    if first_nonempty.startswith("[") or first_nonempty.startswith("{"):
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict):
                    # some files may wrap data under a key
                    for key in ["data", "examples", "records"]:
                        if key in obj and isinstance(obj[key], list):
                            return obj[key]
                    return [obj]
            except json.JSONDecodeError:
                pass

    # JSONL
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _convert_musique_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one MuSiQue item into the same unified format.

    Expected fields:
    - paragraphs: list of {"idx", "title", "paragraph_text"}
    - question_decomposition: list of {"id", "question", "answer", "paragraph_support_idx"}
    """
    paragraphs = item.get("paragraphs", [])
    decomps = item.get("question_decomposition", [])

    # sort paragraphs by idx if possible
    def _para_key(p: Dict[str, Any]) -> int:
        return int(p.get("idx", 10**9))

    paragraphs = sorted(paragraphs, key=_para_key)

    titles = [p.get("title", "") for p in paragraphs]
    texts = [p.get("paragraph_text", "") for p in paragraphs]

    hop_order = []
    for step in decomps:
        if "paragraph_support_idx" in step and step["paragraph_support_idx"] is not None:
            hop_order.append(int(step["paragraph_support_idx"]))

    hop_order = _ordered_unique(hop_order)

    answer_aliases = item.get("answer_aliases", [])
    if answer_aliases is None:
        answer_aliases = []

    return {
        "id": item.get("id"),
        "query": item.get("question"),
        "answer": item.get("answer"),
        "answer_aliases": answer_aliases,
        "answerable": bool(item.get("answerable", True)),
        "candidates": titles,
        "candidate_texts": texts,
        "positive_set": hop_order[:],
        "hop_order": hop_order[:],
        "valid_orders": [hop_order[:]] if hop_order else None,
        "order_source": "gold_decomposition" if hop_order else "unordered",
        "question_decomposition": decomps,
    }


def read_musique(
    path: Optional[str] = None,
    split: str = "train",
    hf_repo: str = "bdsaglam/musique",
    answerable_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Read MuSiQue either from a local file or from Hugging Face datasets,
    and convert every item to the unified format.

    Usage:
        data = read_musique(path="musique_ans_v1.0_train.jsonl")
        data = read_musique(split="train", hf_repo="bdsaglam/musique")

    Returns:
        List[Dict[str, Any]]
    """
    if path is not None:
        raw_records = _load_json_or_jsonl(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is not installed. Please `pip install datasets`, "
                "or provide a local `path`."
            ) from exc

        ds = load_dataset(hf_repo, split=split)
        raw_records = [dict(x) for x in ds]

    output = []
    for item in raw_records:
        ex = _convert_musique_item(item)
        if answerable_only and not ex.get("answerable", True):
            continue
        output.append(ex)
    return output


def read_2wiki_and_process(
    path: Optional[str] = None,
    split: str = "train",
    hf_repo: str = "framolfese/2WikiMultihopQA",
    comparison_mode: str = "all_permutations",
) -> List[Dict[str, Any]]:
    """
    Read 2Wiki either from a local file or from Hugging Face datasets,
    then process every item into the unified format.

    Usage:
        data = read_2wiki_and_process(path="train.json")
        data = read_2wiki_and_process(split="train", hf_repo="framolfese/2WikiMultihopQA")
    """
    if path is not None:
        raw_records = _load_json_or_jsonl(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is not installed. Please `pip install datasets`, "
                "or provide a local `path`."
            ) from exc

        ds = load_dataset(hf_repo, split=split)
        raw_records = [dict(x) for x in ds]

    return [process_2wiki_item(item, comparison_mode=comparison_mode) for item in raw_records]


from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset


class MultiHopRetrievalDataset(Dataset):
    """
    Dataset over unified multi-hop retrieval examples.

    Each example is expected to look like:
    {
        "id": str,
        "query": str,
        "candidates": List[str],          # titles
        "candidate_texts": List[str],     # paragraph texts
        "positive_set": List[int],
        "hop_order": Optional[List[int]],
        "valid_orders": Optional[List[List[int]]],
        "order_source": str
    }
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        use_title: bool = True,
        use_text: bool = True,
        title_sep: str = "[SEP]",
    ) -> None:
        self.examples = examples
        self.use_title = use_title
        self.use_text = use_text
        self.title_sep = title_sep

        if not self.use_title and not self.use_text:
            raise ValueError("At least one of use_title or use_text must be True.")

    def __len__(self) -> int:
        return len(self.examples)

    def _build_doc_text(self, ex: Dict[str, Any], idx: int) -> str:
        title = ex["candidates"][idx] if idx < len(ex["candidates"]) else ""
        text = ex["candidate_texts"][idx] if idx < len(ex["candidate_texts"]) else ""

        if self.use_title and self.use_text:
            if title and text:
                return f"{title}{self.title_sep}{text}"
            return title or text
        elif self.use_title:
            return title
        else:
            return text

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ex = self.examples[index]

        doc_texts = [self._build_doc_text(ex, i) for i in range(len(ex["candidates"]))]

        hop_order = ex.get("hop_order")
        valid_orders = ex.get("valid_orders")
        positive_set = ex.get("positive_set", [])

        return {
            "id": ex.get("id", str(index)),
            "query": ex["query"],
            "doc_texts": doc_texts,
            "doc_titles": ex["candidates"],
            "positive_set": positive_set,
            "hop_order": hop_order,
            "valid_orders": valid_orders,
            "order_source": ex.get("order_source", "unordered"),
            "raw_example": ex,
        }
    

def get_tokenizer(model_name: str = "bert-base-uncased", use_auth_token: Optional[str] = None) -> AutoTokenizer:
    """
    初始化tokenizer并配置padding。

    参数:
        model_name: 模型名称或路径
        use_auth_token: Hugging Face认证token

    返回:
        配置好的tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=use_auth_token,
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # 退一步，新增一个 pad_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

# 初始化tokenizer
tokenizer = get_tokenizer(model_name="bert-base-uncased")

def load_and_combine_datasets(
    musique_path: Optional[str] = None,
    wiki_path: Optional[str] = None,
    split: str = "train",
    answerable_only: bool = True,
    comparison_mode: str = "all_permutations",
) -> List[Dict[str, Any]]:
    """
    加载并混合MuSiQue和2Wiki数据集。
    
    参数:
        musique_path: MuSiQue数据集的本地路径，如果为None则从Hugging Face加载
        wiki_path: 2Wiki数据集的本地路径，如果为None则从Hugging Face加载
        split: 数据集分割（train/dev/test）
        answerable_only: 是否只包含可回答的样本
        comparison_mode: 2Wiki比较问题的处理模式
        
    返回:
        混合后的数据集列表
    """
    combined_data = []
    
    # 加载MuSiQue数据
    musique_data = read_musique(
        path=musique_path,
        split=split,
        answerable_only=answerable_only,
    )
    
    # 为MuSiQue数据添加来源标记
    for item in musique_data:
        item["source"] = "musique"
        combined_data.append(item)
    
    # 加载2Wiki数据
    wiki_data = read_2wiki_and_process(
        path=wiki_path,
        split=split,
        comparison_mode=comparison_mode,
    )
    
    # 为2Wiki数据添加来源标记
    for item in wiki_data:
        item["source"] = "2wiki"
        combined_data.append(item)
    
    return combined_data

# # 读取并混合MuSiQue和2Wiki数据集
# combined_data = load_and_combine_datasets(
#     split="train",
#     answerable_only=True,
# )

# print(f"Total samples: {len(combined_data)}")
# print(f"MuSiQue samples: {sum(1 for x in combined_data if x['source'] == 'musique')}")
# print(f"2Wiki samples: {sum(1 for x in combined_data if x['source'] == '2wiki')}")

# 打印第一个样本的信息
# if combined_data:
#     print("\nFirst sample info:")
#     print(f"Source: {combined_data[0]['source']}")
#     print(f"Query: {combined_data[0]['query']}")
#     print(f"Hop order: {combined_data[0].get('hop_order', [])}")
#     print(f"Positive set: {combined_data[0].get('positive_set', [])}")

def construct_multihop_step_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    max_retrieved_docs: int = 10,
    use_title: bool = True,
    use_text: bool = True,
    title_sep: str = "[SEP]",
    num_hard_negatives: int = 3,
) -> List[Dict[str, Any]]:
    """
    构造multihop数据集，每个推理步骤作为一个样本。
    
    对于每个样本，输入是：
    - 第一个step: 原始query
    - 后续step: 原始query + 前续order的document内容
    
    输出是：
    - 当前步骤的ground truth文档
    
    每个样本还包含hard negatives，来自paragraphs中除正例外的其他paragraph
    
    参数:
        examples: 原始数据集，每个样本包含hop_order等信息
        tokenizer: 用于文本编码的tokenizer
        max_length: 最大序列长度
        max_retrieved_docs: 每个步骤检索的最大文档数
        use_title: 是否使用文档标题
        use_text: 是否使用文档文本
        title_sep: 标题和文本之间的分隔符
        num_hard_negatives: 每个样本包含的hard negative数量
        
    返回:
        List[Dict[str, Any]]: 每个步骤的样本列表
    """
    step_samples = []
    
    for ex in tqdm(examples):
        query = ex["query"]
        hop_order = ex.get("hop_order", [])
        candidates = ex["candidates"]
        candidate_texts = ex["candidate_texts"]
        
        if not hop_order:
            continue
        
        # 为每个hop步骤创建一个样本
        for step_idx, gold_doc_idx in enumerate(hop_order):
            # 构建当前步骤的文档文本
            def _build_doc_text(idx: int) -> str:
                title = candidates[idx] if idx < len(candidates) else ""
                text = candidate_texts[idx] if idx < len(candidate_texts) else ""
                
                if use_title and use_text:
                    if title and text:
                        return f"{title}{title_sep}{text}"
                    return title or text
                elif use_title:
                    return title
                else:
                    return text
            
            # 获取当前步骤的文档文本
            current_doc_text = _build_doc_text(gold_doc_idx)
            
            # 构建输入：第一个step是query本身，后续step是query+前续order的document内容
            if step_idx == 0:
                # 第一个step的输入是query本身
                input_text = query
            else:
                # 后续step的输入是query+前续order的document内容
                prev_docs = []
                for prev_idx in hop_order[:step_idx]:
                    prev_docs.append(_build_doc_text(prev_idx))
                input_text = f"{query} {' '.join(prev_docs)}"
            
            # Tokenize输入和输出
            input_encodings = tokenizer(
                input_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            output_encodings = tokenizer(
                current_doc_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            # 选择hard negatives: 从paragraphs中除正例外的其他paragraph
            # 获取所有非正例的文档索引
            all_indices = set(range(len(candidates)))
            positive_indices = set(hop_order[:step_idx+1])  # 包括当前步骤的正例
            negative_indices = list(all_indices - positive_indices)
            
            # 如果有足够的negative，随机选择num_hard_negatives个
            import random
            if len(negative_indices) > num_hard_negatives:
                selected_negatives = random.sample(negative_indices, num_hard_negatives)
            else:
                selected_negatives = negative_indices
            
            # 构建hard negative文档文本
            hard_negative_texts = [_build_doc_text(idx) for idx in selected_negatives]
            hard_negative_titles = [candidates[idx] for idx in selected_negatives]
            
            # Tokenize hard negatives
            hard_negative_encodings = []
            for neg_text in hard_negative_texts:
                neg_encoding = tokenizer(
                    neg_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                hard_negative_encodings.append({
                    "input_ids": neg_encoding["input_ids"].squeeze(0),
                    "attention_mask": neg_encoding["attention_mask"].squeeze(0),
                })
            # 构建样本
            step_sample = {
                "id": f'{ex.get("id", "unknown")}_step{step_idx}',
                "original_id": ex.get("id", "unknown"),
                "step": step_idx,
                "query": query,
                "input_text": input_text,
                "output_text": current_doc_text,
                "input_ids": input_encodings["input_ids"].squeeze(0),
                "attention_mask": input_encodings["attention_mask"].squeeze(0),
                "labels": output_encodings["input_ids"].squeeze(0),
                "gold_doc_idx": gold_doc_idx,
                "hop_order": hop_order,
                "num_hops": len(hop_order),
                "hard_negatives": hard_negative_texts,
                "hard_negative_titles": hard_negative_titles,
                "hard_negative_indices": selected_negatives,
                "hard_negative_encodings": hard_negative_encodings,
            }
            
            step_samples.append(step_sample)
    
    return step_samples

class MultiHopStepDataset(Dataset):
    """
    PyTorch Dataset for multi-hop step-level training.

    每个样本包含一个推理步骤的数据，输入是query+当前文档，输出是ground truth文档。
    """
    def __init__(self, step_samples: List[Dict[str, Any]]):
        """
        初始化数据集。

        参数:
            step_samples: construct_multihop_step_dataset返回的步骤样本列表
        """
        self.step_samples = step_samples

    def __len__(self) -> int:
        return len(self.step_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取一个样本。

        返回:
            包含input_ids, attention_mask, labels等字段的字典
        """
        sample = self.step_samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
            "id": sample["id"],
            "original_id": sample["original_id"],
            "step": sample["step"],
            "query": sample["query"],
            "gold_doc_idx": sample["gold_doc_idx"],
            "hop_order": sample["hop_order"],
            "num_hops": sample["num_hops"],
        }


# # 使用示例：从combined_data构造multihop步骤数据集
# multihop_samples = construct_multihop_step_dataset(
#     examples=combined_data,
#     tokenizer=tokenizer,
#     max_length=512,
# )

# print(f"Total step samples: {len(multihop_samples)}")
# if multihop_samples:
#     print(f"First sample keys: {multihop_samples[0].keys()}")

# # 创建PyTorch Dataset
# multihop_dataset = MultiHopStepDataset(multihop_samples)
# print(f"Dataset size: {len(multihop_dataset)}")
# print(f"First dataset sample keys: {multihop_dataset[0].keys()}")