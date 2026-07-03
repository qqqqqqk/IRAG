
"""
IRCOT-style Evaluation Pipeline with Custom Retrieval Model
==========================================================
This script implements an IRCOT-style pipeline for multi-hop QA evaluation,
but replaces the retrieval model with the custom model trained in multihop_contrastive_train.py.

Based on the IRCOT paper, this evaluation supports the following datasets:
1. HotpotQA (fullwiki and distractor)
2. 2WikiMultiHopQA
3. MuSiQue

The pipeline is separated into two main components:
1. Indexing: Building document indices for efficient retrieval
2. IRCOT Pipeline: Multi-hop reasoning with iterative retrieval

Usage:
    python ircot_evaluation.py \
        --model_path /path/to/checkpoint \
        --dataset_name hotpotqa \
        --split dev \
        --output_dir ./results
"""

import os
import json
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# Import from existing modules
from models import InforNCE_and_Generative_Hops_Eval
from dataset_loading import (
    _normalize_text,
    _ordered_unique,
    _safe_get,
    _build_2wiki_context,
    _resolve_entity_to_context_idx,
    _infer_chain_from_edges,
)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""
    model_path: str
    dataset_name: str  # "hotpotqa", "2wiki" or "musique"
    split: str = "dev"
    output_dir: str = "./results"
    batch_size: int = 8
    max_retrieved_docs: int = 10
    max_hops: int = 3
    temperature: float = 0.05
    use_title: bool = True
    use_text: bool = True
    title_sep: str = "[SEP]"
    max_length: int = 512
    hf_repo: Optional[str] = None
    local_data_path: Optional[str] = None
    # HotpotQA specific
    hotpotqa_mode: str = "distractor"  # "distractor" or "fullwiki"
    # Indexing configuration
    use_indexing: bool = True  # Whether to use indexing for efficient retrieval
    index_path: Optional[str] = None  # Path to save/load index
    # Tokenizer configuration
    tokenizer_path: Optional[str] = None  # Path to tokenizer (e.g., original Qwen3-0.6B model)


class DocumentIndexer:
    """Document indexer for efficient retrieval in multi-hop QA."""
    
    def __init__(self, config: EvaluationConfig, model, tokenizer, device, logger=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.index = None
        self.documents = None
        self.document_embeddings = None
        self.logger = logger or logging.getLogger(__name__)
        
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build document index from a list of documents.
        
        Args:
            documents: List of documents, each with "title" and "text" keys
        """
        self.documents = documents
        self._compute_embeddings()
        self.logger.info(f"Built index for {len(documents)} documents")
        
    def _compute_embeddings(self) -> None:
        """Compute embeddings for all documents in the index."""
        if not self.documents:
            return
            
        doc_texts = [
            self._build_doc_text(doc.get("title", ""), doc.get("text", ""))
            for doc in self.documents
        ]
        
        all_embeddings = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(doc_texts), batch_size), desc="Computing document embeddings"):
                batch_texts = doc_texts[i:i+batch_size]
                doc_encodings = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                
                doc_input_ids = doc_encodings["input_ids"].to(self.device)
                doc_attention_mask = doc_encodings["attention_mask"].to(self.device)
                
                doc_embeddings = self.model(
                    input_ids=doc_input_ids,
                    attention_mask=doc_attention_mask,
                    is_query=False
                )
                doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
                all_embeddings.append(doc_embeddings.cpu())
        
        self.document_embeddings = torch.cat(all_embeddings, dim=0)
        
    def _build_doc_text(self, title: str, text: str) -> str:
        """Build document text from title and content."""
        if self.config.use_title and self.config.use_text:
            if title and text:
                return f"{title}{self.config.title_sep}{text}"
            return title or text
        elif self.config.use_title:
            return title
        else:
            return text
    
    def save_index(self, path: str) -> None:
        """Save the document index to disk.
        
        Args:
            path: Path to save the index
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "documents": self.documents,
            "document_embeddings": self.document_embeddings,
        }, path)
        
    def load_index(self, path: str) -> None:
        """Load the document index from disk.
        
        Args:
            path: Path to load the index from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
            
        data = torch.load(path)
        self.documents = data["documents"]
        self.document_embeddings = data["document_embeddings"]
        
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[int], List[float]]:
        """Retrieve top-k documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve (default: config.max_retrieved_docs)
            
        Returns:
            Tuple of (document_indices, scores)
        """
        if self.document_embeddings is None:
            raise ValueError("Index not built. Call build_index() or load_index() first.")
            
        if top_k is None:
            top_k = self.config.max_retrieved_docs
            
        # Encode query
        query_encodings = self.tokenizer(
            query,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            query_input_ids = query_encodings["input_ids"].to(self.device)
            query_attention_mask = query_encodings["attention_mask"].to(self.device)
            
            query_embedding = self.model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                is_query=True
            )
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            
            # Compute similarity scores
            scores = (query_embedding @ self.document_embeddings.T.to(self.device)).squeeze(0)
            
            # Get top-k
            top_k = min(top_k, len(scores))
            top_scores, top_indices = torch.topk(scores, k=top_k)
            
        return top_indices.tolist(), top_scores.tolist()


class MultiHopQAEvaluator:
    """Evaluator for multi-hop QA using IRCOT-style pipeline."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        self.load_model()
        self.load_dataset()
        
        # Initialize document indexer
        self.indexer = DocumentIndexer(config, self.model, self.tokenizer, self.device, self.logger)
        
        # Build or load index if configured
        if self.config.use_indexing:
            self._setup_index()

    def setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        log_file = os.path.join(
            self.config.output_dir,
            f"ircot_eval_{self.config.dataset_name}_{self.config.split}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Evaluation config: {self.config}")

    def load_model(self):
        """Load the trained retrieval model."""
        self.logger.info(f"Loading model from {self.config.model_path}")

        # Initialize tokenizer from original model path if tokenizer_path is provided
        tokenizer_path = self.config.tokenizer_path if self.config.tokenizer_path else self.config.model_path
        self.logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Initialize model with original model path (not checkpoint path)
        base_model_path = self.config.tokenizer_path if self.config.tokenizer_path else self.config.model_path
        self.logger.info(f"Initializing model from {base_model_path}")
        self.model = InforNCE_and_Generative_Hops_Eval(
            local_rank=0,
            model_name=base_model_path
        )
        
        # Load trained weights from checkpoint
        self.logger.info(f"Loading trained weights from {self.config.model_path}")
        self.model.load_checkpoint(self.config.model_path)
        
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model loaded successfully")

    def load_dataset(self):
        """Load the evaluation dataset."""
        self.logger.info(f"Loading {self.config.dataset_name} dataset ({self.config.split})")

        if self.config.dataset_name == "hotpotqa":
            self.dataset = self._load_hotpotqa_dataset()
        elif self.config.dataset_name == "2wiki":
            self.dataset = self._load_2wiki_dataset()
        elif self.config.dataset_name == "musique":
            self.dataset = self._load_musique_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")

        self.logger.info(f"Loaded {len(self.dataset)} examples")

    def _setup_index(self):
        """Setup document index for efficient retrieval."""
        # Collect all unique documents from the dataset
        all_documents = []
        seen_docs = set()
        
        for example in self.dataset:
            for title, text in zip(example["candidates"], example["candidate_texts"]):
                doc_key = title
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    all_documents.append({"title": title, "text": text})
        
        self.logger.info(f"Building index for {len(all_documents)} unique documents")
        
        # Try to load existing index if available
        if self.config.index_path and os.path.exists(self.config.index_path):
            try:
                self.indexer.load_index(self.config.index_path)
                self.logger.info(f"Loaded existing index from {self.config.index_path}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load index from {self.config.index_path}: {e}")
        else:
            # Build new index
            self.indexer.build_index(all_documents)
        
        # Save index if path is specified
        if self.config.index_path and (not os.path.exists(self.config.index_path)):
            try:
                self.indexer.save_index(self.config.index_path)
                self.logger.info(f"Saved index to {self.config.index_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save index to {self.config.index_path}: {e}")

    def _load_hotpotqa_dataset(self) -> List[Dict[str, Any]]:
        """Load HotpotQA dataset (both distractor and fullwiki modes)."""
        if self.config.local_data_path:
            with open(self.config.local_data_path, "r") as f:
                data = json.load(f) if self.config.local_data_path.endswith('.json') else [json.loads(line) for line in f]
        else:
            if self.config.hotpotqa_mode == "distractor":
                hf_repo = self.config.hf_repo or "hotpotqa/hotpotqa-distractor"
            else:  # fullwiki
                hf_repo = self.config.hf_repo or "hotpotqa/hotpotqa-fullwiki"
            ds1 = load_dataset(hf_repo, split="train")
            ds2 = load_dataset(hf_repo, split="validation")
            data1 = [dict(x) for x in ds1]
            data2 = [dict(x) for x in ds2]
            data = data1 + data2

        # Process each item
        processed_data = []
        for item in tqdm(data, desc="Processing HotpotQA data"):
            processed_item = self._process_hotpotqa_item(item)
            processed_data.append(processed_item)

        return processed_data

    def _load_2wiki_dataset(self) -> List[Dict[str, Any]]:
        """Load 2Wiki dataset."""
        if self.config.local_data_path:
            with open(self.config.local_data_path, "r") as f:
                data = json.load(f) if self.config.local_data_path.endswith('.json') else [json.loads(line) for line in f]
        else:
            hf_repo = self.config.hf_repo or "framolfese/2WikiMultihopQA"
            ds1 = load_dataset(hf_repo, split="train")
            ds2 = load_dataset(hf_repo, split="validation")
            data1 = [dict(x) for x in ds1]
            data2 = [dict(x) for x in ds2]
            data = data1 + data2

        # Process each item
        processed_data = []
        for item in tqdm(data, desc="Processing 2Wiki data"):
            processed_item = self._process_2wiki_item(item)
            processed_data.append(processed_item)

        return processed_data

    def _load_musique_dataset(self) -> List[Dict[str, Any]]:
        """Load MuSiQue dataset."""
        if self.config.local_data_path:
            with open(self.config.local_data_path, "r") as f:
                data = [json.loads(line) for line in f]
        else:
            hf_repo = self.config.hf_repo or "dgslibisey/MuSiQue"
            ds1 = load_dataset(hf_repo, split="train")
            ds2 = load_dataset(hf_repo, split="validation")
            data1 = [dict(x) for x in ds1]
            data2 = [dict(x) for x in ds2]
            data = data1 + data2

        # Process each item
        processed_data = []
        for item in tqdm(data, desc="Processing MuSiQue data"):
            processed_item = self._process_musique_item(item)
            processed_data.append(processed_item)

        return processed_data

    def _process_hotpotqa_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a HotpotQA dataset item."""
        # Extract context paragraphs
        context = item.get("context", [])
        titles = []
        texts = []
        
        for para in context:
            if isinstance(para, list) and len(para) >= 2:
                title = para[0]
                sentences = para[1] if isinstance(para[1], list) else [para[1]]
                titles.append(title)
                texts.append(" ".join(sentences))
        
        # Extract supporting facts
        supporting_facts = item.get("supporting_facts", [])
        support_titles = _ordered_unique([sf[0] for sf in supporting_facts])
        
        # Map support titles to indices
        title_to_indices = defaultdict(list)
        for idx, title in enumerate(titles):
            title_to_indices[title].append(idx)
        
        # Get positive set (indices of supporting paragraphs)
        positive_set = []
        for title in support_titles:
            hits = title_to_indices.get(title, [])
            if len(hits) >= 1:
                positive_set.append(hits[0])
        positive_set = _ordered_unique(positive_set)
        
        # Get question type
        question_type = str(item.get("type", "")).lower()
        question = item.get("question", "")
        
        # For multi-hop questions, try to infer hop order
        hop_order = None
        order_source = "unordered"
        
        if question_type == "multi-hop" and len(positive_set) > 1:
            # Try to infer order from supporting facts
            # Use the order in which supporting facts appear in the question
            hop_order = positive_set
            order_source = "support_facts_order"
        
        return {
            "id": item.get("_id", ""),
            "query": question,
            "answer": item.get("answer", ""),
            "question_type": question_type,
            "candidates": titles,
            "candidate_texts": texts,
            "positive_set": positive_set,
            "hop_order": hop_order,
            "valid_orders": [hop_order] if hop_order is not None else None,
            "order_source": order_source,
            "level": item.get("level", ""),  # "easy" or "hard"
        }

    def _process_2wiki_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a 2Wiki dataset item."""
        paragraphs = _build_2wiki_context(item)
        titles = [p["title"] for p in paragraphs]
        texts = [p["paragraph_text"] for p in paragraphs]

        sf = item.get("supporting_facts", {})
        support_titles = _ordered_unique(_safe_get(sf, "title", []))

        title_to_indices = defaultdict(list)
        for idx, title in enumerate(titles):
            title_to_indices[title].append(idx)

        positive_set = []
        for title in support_titles:
            hits = title_to_indices.get(title, [])
            if len(hits) >= 1:
                positive_set.append(hits[0])
        positive_set = _ordered_unique(positive_set)

        question_type = str(item.get("type", "")).lower()
        question = item.get("question", "")

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

    def _process_musique_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a MuSiQue dataset item."""
        paragraphs = item.get("paragraphs", [])
        decomps = item.get("question_decomposition", [])

        # Sort paragraphs by idx if possible
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

    def _build_doc_text(self, title: str, text: str) -> str:
        """Build document text from title and content."""
        if self.config.use_title and self.config.use_text:
            if title and text:
                return f"{title}{self.config.title_sep}{text}"
            return title or text
        elif self.config.use_title:
            return title
        else:
            return text

    def retrieve_documents(
        self,
        query: str,
        candidate_texts: List[str],
        candidate_titles: List[str],
        retrieved_docs: List[str] = None,
        hop_idx: int = 0
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query text
            candidate_texts: List of candidate document texts
            candidate_titles: List of candidate document titles
            retrieved_docs: Previously retrieved documents (for multi-hop)
            hop_idx: Current hop index

        Returns:
            Tuple of (top_indices, scores)
        """
        # Build query text (include previously retrieved docs for multi-hop)
        if hop_idx > 0 and retrieved_docs:
            query_text = f"{query} {' '.join(retrieved_docs)}"
        else:
            query_text = query

        # Use indexer if available
        if self.config.use_indexing and self.indexer.document_embeddings is not None:
            # Retrieve from global index
            global_indices, scores = self.indexer.retrieve(query_text, top_k=10)
            
            # Map global indices back to candidate indices
            # title_to_idx = {title: idx for idx, title in enumerate(candidate_titles)}
            # candidate_indices = []
            # candidate_scores = []
            # for g_idx, score in zip(global_indices, scores):
            #     if g_idx < len(self.indexer.documents):
            #         doc_title = self.indexer.documents[g_idx]["title"]
            #         c_idx = title_to_idx.get(doc_title)
            #         if c_idx is not None:
            #             candidate_indices.append(c_idx)
            #             candidate_scores.append(score)
            #     if len(candidate_indices) >= self.config.max_retrieved_docs:
            #        break
            return global_indices, scores 
        
        # Fallback to original method if indexer not available
        # Build candidate texts
        doc_texts = [
            self._build_doc_text(title, text)
            for title, text in zip(candidate_titles, candidate_texts)
        ]

        # Tokenize query
        query_encodings = self.tokenizer(
            query_text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize documents in batches
        all_scores = []
        batch_size = self.config.batch_size
        with torch.no_grad():
            # Encode query
            query_input_ids = query_encodings["input_ids"].to(self.device)
            query_attention_mask = query_encodings["attention_mask"].to(self.device)

            query_embedding = self.model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                is_query=True
            )
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)

            # Encode documents in batches
            for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding documents"):
                batch_texts = doc_texts[i:i+batch_size]
                doc_encodings = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                doc_input_ids = doc_encodings["input_ids"].to(self.device)
                doc_attention_mask = doc_encodings["attention_mask"].to(self.device)

                doc_embeddings = self.model(
                    input_ids=doc_input_ids,
                    attention_mask=doc_attention_mask,
                    is_query=False
                )
                doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)

                # Compute similarity scores
                scores = (query_embedding @ doc_embeddings.T).squeeze(0)

                all_scores.append(scores.cpu())
        # Concatenate all scores
        all_scores = torch.cat(all_scores, dim=0)

        # Get top-k indices
        top_k = min(self.config.max_retrieved_docs, len(all_scores))
        top_scores, top_indices = torch.topk(all_scores, k=top_k)

        return top_indices.tolist(), top_scores.tolist()

    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single example using IRCOT-style multi-hop retrieval.

        Args:
            example: A single dataset example

        Returns:
            Dictionary containing evaluation results
        """
        query = example["query"]
        candidate_texts = example["candidate_texts"]
        candidate_titles = example["candidates"]
        gold_docs = example.get("positive_set", [])
        hop_order = example.get("hop_order", [])

        # Multi-hop retrieval
        retrieved_docs = []
        retrieved_indices = []
        retrieved_scores = []
        retrieved_texts = []
        for hop_idx in range(self.config.max_hops):
            # Retrieve documents for current hop
            top_indices, scores = self.retrieve_documents(
                query=query,
                candidate_texts=candidate_texts,
                candidate_titles=candidate_titles,
                retrieved_docs=retrieved_texts,
                hop_idx=hop_idx
            )
            # Store results
            retrieved_indices.append(top_indices)
            retrieved_scores.append(scores)

            # Build retrieved texts for next hop
            hop_retrieved_texts = []
            for idx in top_indices:
                doc_text = self._build_doc_text(
                    self.indexer.documents[idx]["title"],
                    self.indexer.documents[idx]["text"]
                )
                hop_retrieved_texts.extend(doc_text)
                retrieved_texts.extend(doc_text)

            retrieved_docs.extend(hop_retrieved_texts)

        # Compute metrics
        metrics = self._compute_metrics(
            retrieved_indices=retrieved_indices,
            gold_docs=gold_docs,
            hop_order=hop_order,
            candidate_texts=candidate_texts,
            candidate_titles=candidate_titles,
            total_candidates=len(candidate_texts)
        )

        return {
            "id": example.get("id"),
            "query": query,
            "gold_docs": gold_docs,
            "hop_order": hop_order,
            "retrieved_indices": retrieved_indices,
            "retrieved_scores": retrieved_scores,
            "metrics": metrics,
        }

    def _compute_metrics(
        self,
        retrieved_indices: List[List[int]],
        gold_docs: List[int],
        hop_order: Optional[List[int]],
        candidate_texts: List[str],
        candidate_titles: List[str],
        total_candidates: int
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.

        Args:
            retrieved_indices: List of retrieved document indices for each hop (global indices from self.indexer.documents)
            gold_docs: Ground truth document indices (local indices in candidates)
            hop_order: Ground truth hop order (local indices in candidates)
            candidate_texts: List of candidate document texts
            candidate_titles: List of candidate document titles
            total_candidates: Total number of candidate documents

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Create mapping from candidate title to local index
        title_to_local_idx = {title: idx for idx, title in enumerate(candidate_titles)}

        # Create mapping from global index to local index
        global_to_local_idx = {}
        for local_idx, (title, text) in enumerate(zip(candidate_titles, candidate_texts)):
            # Find the global index for this document
            for global_idx, doc in enumerate(self.indexer.documents):
                if doc["title"] == title and doc["text"] == text:
                    global_to_local_idx[global_idx] = local_idx
                    break

        # Flatten retrieved indices across all hops and convert to local indices
        all_retrieved_local = []
        for hop_indices in retrieved_indices:
            for global_idx in hop_indices:
                if global_idx in global_to_local_idx:
                    local_idx = global_to_local_idx[global_idx]
                    all_retrieved_local.append(local_idx)

        # Remove duplicates while preserving order
        unique_retrieved_local = []
        seen = set()
        for idx in all_retrieved_local:
            if idx not in seen:
                seen.add(idx)
                unique_retrieved_local.append(idx)

        # Recall@k metrics
        for k in [1, 3, 5, 10]:
            if k <= len(unique_retrieved_local):
                retrieved_k = set(unique_retrieved_local[:k])
                gold_set = set(gold_docs)
                recall = len(retrieved_k & gold_set) / len(gold_set) if gold_set else 0.0
                metrics[f"recall@{k}"] = recall

        # Precision@k metrics
        for k in [1, 3, 5, 10]:
            if k <= len(unique_retrieved_local):
                retrieved_k = set(unique_retrieved_local[:k])
                gold_set = set(gold_docs)
                precision = len(retrieved_k & gold_set) / k if k > 0 else 0.0
                metrics[f"precision@{k}"] = precision

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for gold_doc in gold_docs:
            for rank, idx in enumerate(unique_retrieved_local, start=1):
                if idx == gold_doc:
                    mrr += 1.0 / rank
                    break
        metrics["mrr"] = mrr / len(gold_docs) if gold_docs else 0.0

        # Hop order accuracy (if available)
        if hop_order and len(hop_order) > 1:
            correct_order = 0
            total_pairs = 0
            for i in range(len(hop_order) - 1):
                for j in range(i + 1, len(hop_order)):
                    total_pairs += 1
                    doc_i = hop_order[i]
                    doc_j = hop_order[j]

                    # Check if the order is preserved in retrieval
                    rank_i = unique_retrieved_local.index(doc_i) if doc_i in unique_retrieved_local else -1
                    rank_j = unique_retrieved_local.index(doc_j) if doc_j in unique_retrieved_local else -1

                    if rank_i >= 0 and rank_j >= 0 and rank_i < rank_j:
                        correct_order += 1

            metrics["order_accuracy"] = correct_order / total_pairs if total_pairs > 0 else 0.0

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the entire dataset.

        Returns:
            Dictionary of aggregated metrics
        """
        self.logger.info("Starting evaluation...")

        all_results = []
        all_metrics = []

        for example in tqdm(self.dataset, desc="Evaluating examples"):
            result = self.evaluate_example(example)
            all_results.append(result)
            all_metrics.append(result["metrics"])
            print()
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)

        # Save results
        self._save_results(all_results, aggregated_metrics)

        self.logger.info("Evaluation completed")
        self.logger.info(f"Aggregated metrics: {aggregated_metrics}")

        return aggregated_metrics

    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all examples."""
        aggregated = {}

        # Compute mean and std for each metric
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregated[f"{metric_name}_std"] = (
                    sum((v - aggregated[f"{metric_name}_mean"]) ** 2 for v in values) / len(values)
                ) ** 0.5

        return aggregated

    def _save_results(
        self,
        all_results: List[Dict[str, Any]],
        aggregated_metrics: Dict[str, float]
    ):
        """Save evaluation results to files."""
        # Save detailed results
        results_file = os.path.join(
            self.config.output_dir,
            f"detailed_results_{self.config.dataset_name}_{self.config.split}.json"
        )
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        self.logger.info(f"Detailed results saved to {results_file}")

        # Save aggregated metrics
        metrics_file = os.path.join(
            self.config.output_dir,
            f"metrics_{self.config.dataset_name}_{self.config.split}.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(aggregated_metrics, f, indent=2)
        self.logger.info(f"Aggregated metrics saved to {metrics_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IRCOT-style evaluation with custom retrieval model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer (e.g., original Qwen3-0.6B model). If not provided, will use model_path"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["hotpotqa", "2wiki", "musique"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--hotpotqa_mode",
        type=str,
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="HotpotQA mode (distractor or fullwiki)"
    )
    parser.add_argument(
        "--use_indexing",
        action="store_true",
        default=True,
        help="Whether to use document indexing for efficient retrieval"
    )
    parser.add_argument(
        "--no_indexing",
        action="store_false",
        dest="use_indexing",
        help="Disable document indexing"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="Path to save/load document index"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for document encoding"
    )
    parser.add_argument(
        "--max_retrieved_docs",
        type=int,
        default=3,
        help="Maximum number of documents to retrieve per hop"
    )
    parser.add_argument(
        "--max_hops",
        type=int,
        default=3,
        help="Maximum number of hops to perform"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature for similarity computation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="Hugging Face repository name (if loading from HF)"
    )
    parser.add_argument(
        "--local_data_path",
        type=str,
        default=None,
        help="Path to local data file"
    )

    args = parser.parse_args()

    # Create evaluation config
    config = EvaluationConfig(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_retrieved_docs=args.max_retrieved_docs,
        max_hops=args.max_hops,
        temperature=args.temperature,
        max_length=args.max_length,
        hf_repo=args.hf_repo,
        local_data_path=args.local_data_path,
        hotpotqa_mode=args.hotpotqa_mode,
        use_indexing=args.use_indexing,
        index_path=args.index_path,
        tokenizer_path=args.tokenizer_path,
    )

    # Create evaluator and run evaluation
    evaluator = MultiHopQAEvaluator(config)
    metrics = evaluator.evaluate()

    print("" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("="*50 + "")


if __name__ == "__main__":
    main()
