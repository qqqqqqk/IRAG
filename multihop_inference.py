#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import os
import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from prompt_templates import PROMPT_TEMPLATES
from utils.model_utils import load_model, BaseModel, calculate_entropy_from_logprobs
from compressor import Compressor

import sys
from retrievers import Retriever

logger = logging.getLogger(__name__)


def format_context(qa_history: List[Dict[str, str]], original_question: str) -> str:
    context_parts = [f"original question: {original_question}"]
    
    for i, qa in enumerate(qa_history):
        context_parts.append(f"follow up sub-question: {qa['sub_question']}")
        context_parts.append(f"intermediate answer: {qa['answer']}")
    
    return "\n".join(context_parts)


def parse_context_judge_response(response: str) -> Tuple[bool, Optional[str]]:
    response = response.strip()
    
    yes_match = re.search(r"[Yy]es,?\s*(?:the\s+)?answer\s+(?:for\s+the\s+original\s+question\s+)?is\s+(.+)", response)
    if yes_match:
        return True, yes_match.group(1).strip().strip('[]')
    
    no_match = re.search(r"[Nn]o.*?\bask\b[\s:]*(.+)", response)
    if no_match:
        return False, no_match.group(1).strip().strip('[]')
    
    if response.lower().startswith("yes"):
        return True, response
    elif response.lower().startswith("no"):
        return False, response
    
    return False, response


def calculate_mean_entropy(token_logprobs: List[Dict[int, float]], use_topk: int=50) -> float:
    """Calculate mean entropy of token logprobs
    
    Args:
        token_logprobs: token logprobs list
        
    Returns:
        float: Mean entropy
    """
    if not token_logprobs:
        return 0.0
    
    entropies = []
    for logprobs in token_logprobs:
        entropy = calculate_entropy_from_logprobs(logprobs, use_top_k=use_topk)
        entropies.append(entropy)
    
    return sum(entropies) / len(entropies) if entropies else 0.0


def process_token_logprobs(token_texts: List[str], token_logprobs: List[Dict[int, float]], 
                          token_ids: List[int], model_output: str) -> Tuple[List[str], List[Dict[int, float]], List[int]]:
    if not token_texts:
        return token_texts, token_logprobs, token_ids
    
    joined_tokens = ''.join(token_texts)

    if joined_tokens == model_output:
        return token_texts, token_logprobs, token_ids

    start_marker = '\n\n'
    end_marker = '<|im_end|>'
    
    start_idx = joined_tokens.rfind(start_marker)
    end_idx = joined_tokens.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        start_idx += len(start_marker)
        
        new_token_texts = []
        new_token_logprobs = []
        new_token_ids = []
        
        current_pos = 0
        for i, token in enumerate(token_texts):
            token_len = len(token)
            token_start = current_pos
            token_end = current_pos + token_len
            
            if token_end > start_idx and token_start < end_idx:
                new_token_texts.append(token)
                if i < len(token_logprobs):
                    new_token_logprobs.append(token_logprobs[i])
                if i < len(token_ids):
                    new_token_ids.append(token_ids[i])
            
            current_pos = token_end
        
        return new_token_texts, new_token_logprobs, new_token_ids
    return token_texts, token_logprobs, token_ids


def parse_sub_q_answer_response(response: str) -> tuple[str, bool]:
    response = response.strip()

    if "No relevant info" in response or "need to optimize sub-question" in response:
        return response, False

    match = re.search(r"answer\s+is\s+(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('[]'), True

    return response, False


def parse_force_answer_response(response: str) -> str:
    response = response.strip()
    
    match = re.search(r"answer\s+is\s+(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('[]')

    return response


def process_single_sample(
    model: BaseModel,
    sample: Dict[str, Any],
    retriever: Optional[Retriever],
    compressor: Optional[Compressor] = None,
    max_hops: int = 8,
    topk: int = 3,
    compress_threshold: float = 0.2,
    system_message: str = "You are a helpful assistant.",
) -> Dict[str, Any]:
    """Process a single sample with multiihop inference."""

    sample_id = sample["question_id"]
    original_question = sample["question_text"]
    gold_answer = sample["answers_objects"][0]["spans"]

    qa_history = []
    
    context_judge_prompt = PROMPT_TEMPLATES["context_judge"]
    sub_q_judge_prompt = PROMPT_TEMPLATES["answer_without_context"]
    sub_q_answer_prompt = PROMPT_TEMPLATES["sub_q_answer"]
    force_answer_prompt = PROMPT_TEMPLATES["force_answer"]
    
    final_answer = None
    
    for hop_idx in range(max_hops):
        if qa_history:  
            context = format_context(qa_history, original_question)
        else:
            context = " "
        judge_prompt = context_judge_prompt.format(
            context=context,
            original_question=original_question
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": judge_prompt}
        ]

        formatted_prompt = model.format_prompt(messages)
        context_judge_output = model.generate(formatted_prompt, max_tokens=128) 

        context_judge_response = context_judge_output["text"]
        can_answer, content = parse_context_judge_response(context_judge_response)

        if can_answer:
            final_answer = content
            break

        sub_question = content
        if not sub_question:
            sub_question = original_question

        sub_q_judge_prompt_filled = sub_q_judge_prompt.format(question=sub_question)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sub_q_judge_prompt_filled}
        ]
        
        formatted_prompt = model.format_prompt(messages)
        sub_q_judge_output = model.generate(
            formatted_prompt,
            max_tokens=128,
            return_logprobs=True,
            logprobs_top_k=50
        )

        sub_q_judge_response = sub_q_judge_output["text"]

        token_logprobs = sub_q_judge_output.get("token_logprobs", [])
        token_ids = sub_q_judge_output.get("token_ids", [])
        token_texts = sub_q_judge_output.get("token_texts", [])
        
        token_texts, token_logprobs, token_ids = process_token_logprobs(
            token_texts, token_logprobs, token_ids, sub_q_judge_response
        )
        mean_entropy = calculate_mean_entropy(token_logprobs)
        response_lower = sub_q_judge_response.strip().lower()
        is_unknown = "unknown" in response_lower
        knows_answer = mean_entropy <= 0.1 and not is_unknown

        internal_answer = sub_q_judge_response.strip()
        
        if knows_answer and internal_answer:
            sub_answer = internal_answer
        else:
            if retriever is not None:
                try:
                    retrieved_docs = retriever.search(sub_question, top_k=topk)
                    if retrieved_docs and len(retrieved_docs) > 0:
                        docs_for_query = retrieved_docs[0] if isinstance(retrieved_docs[0], list) else retrieved_docs
                        
                        if compressor is not None:
                            compressed_docs = compressor.compress(
                                query=sub_question,
                                docs=docs_for_query[:topk],
                                threshold=compress_threshold
                            )
                            compressed_docs = [doc for doc in compressed_docs if doc['text'].strip()]
                            docs_for_query = compressed_docs
                        
                        retrieved_context = "\n\n".join([
                            f"[Doc {i+1}] : {doc['text']}"
                            for i, doc in enumerate(docs_for_query[:topk])
                        ])
                    else:
                        raise ValueError("No retrieved documents found.")
                except Exception as e:
                    raise ValueError(f"Retrieval failed: {e}")
            else:
                raise ValueError("Retriever not available.")

            sub_q_answer_prompt_filled = sub_q_answer_prompt.format(
                context=retrieved_context,
                sub_question=sub_question
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": sub_q_answer_prompt_filled}
            ]
            
            formatted_prompt = model.format_prompt(messages)
            sub_q_answer_output = model.generate(formatted_prompt, max_tokens=128)

            sub_q_answer_response = sub_q_answer_output["text"]
            sub_answer, answer_success = parse_sub_q_answer_response(sub_q_answer_response)
            if not answer_success:
                sub_answer = "No relevant info, need to optimize sub-question"

        qa_history.append({
            "sub_question": sub_question,
            "answer": sub_answer
        })
    
    if final_answer is None:
        context = format_context(qa_history, original_question)
        final_answer_prompt = force_answer_prompt.format(
            context=context,
            original_question=original_question
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": final_answer_prompt}
        ]
        
        formatted_prompt = model.format_prompt(messages)

        final_output = model.generate(formatted_prompt, max_tokens=128)
        final_response = final_output["text"]
        final_answer = parse_force_answer_response(final_response)
    

    result = {
        "sample_id": sample_id,
        "original_question": original_question,
        "gold_answer": gold_answer,
        "predicted_answer": final_answer,
    }
    
    return result


def process_dataset(
    model: BaseModel,
    dataset: List[Dict[str, Any]],
    retriever: Optional[Retriever],
    compressor: Optional[Compressor],
    output_path: str,
    max_hops: int = 8,
    topk: int = 5,
    compress_threshold: float = 0.2,
    system_message: str = "You are a helpful assistant.",
    use_progress_bar: bool = True,
) -> List[Dict[str, Any]]:
    """process dataset"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # check if processed
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    processed_ids.add(result.get('sample_id'))
                except:
                    continue
        logger.info(f"processed already {len(processed_ids)} samples")
    
    results = []
    iterator = tqdm(dataset, desc="Processing samples") if use_progress_bar else dataset
    
    with open(output_path, 'a', encoding='utf-8') as fout:
        for sample in iterator:
            sample_id = sample["question_id"]
            
            if sample_id in processed_ids:
                continue
            
            try:
                result = process_single_sample(
                    model=model,
                    sample=sample,
                    retriever=retriever,
                    compressor=compressor,
                    max_hops=max_hops,
                    topk=topk,
                    compress_threshold=compress_threshold,
                    system_message=system_message,
                )
                
                results.append(result)
                fout.write(json.dumps(result, ensure_ascii=True) + '\n')
                fout.flush()
                
            except Exception as e:
                logger.error(f"process sample {sample_id} failed: {e}")
    
    return results


def run_multihop_inference(
    model_name: str,
    compress_model_name: str,
    dataset_path: str,
    output_path: str,
    passage_path: str,
    embedding_path: str,
    heads_json: str,
    max_hops: int = 8,
    topk: int = 5,
    compress_threshold: float = 0.2,
    backend: str = "vllm",
    system_message: str = "You are a helpful assistant.",
    **model_kwargs
) -> List[Dict[str, Any]]:
    logger.info(f"loading model: {model_name}")
    model = load_model(model_name, backend=backend, **model_kwargs)
    logger.info("model loaded successfully")
    
    logger.info(f"loading retriever...")
    retriever = Retriever(
        passage_path=passage_path,
        passage_embedding_path=embedding_path,
        index_path_dir=embedding_path,
        model_type="e5-large-v2",
    )
    logger.info("retriever loaded successfully")
    
    logger.info(f"loading compressor...")
    compressor = Compressor(
        model_path=compress_model_name,
        heads_file=heads_json
    )
    logger.info("compressor loaded successfully")
    
    logger.info(f"loading dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    logger.info(f"dataset loaded successfully, total {len(dataset)} samples")
    
    results = process_dataset(
        model=model,
        dataset=dataset,
        retriever=retriever,
        compressor=compressor,
        output_path=output_path,
        max_hops=max_hops,
        topk=topk,
        compress_threshold=compress_threshold,
        system_message=system_message,
    )
    
    logger.info(f"processed successfully, total {len(results)} samples")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-hop QA inference')
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument('--output_path', type=str, required=True, help='output path')
    parser.add_argument('--passage_path', type=str, required=True, help='passage path')
    parser.add_argument('--embedding_path', type=str, required=True, help='embedding path')
    parser.add_argument('--heads_json', type=str, required=True, help='heads json path')
    parser.add_argument('--max_hops', type=int, default=4, help='max hops number')
    parser.add_argument('--topk', type=int, default=10, help='retrieve top-k number')
    parser.add_argument('--compress_threshold', type=float, default=0.4, help='compress threshold')
    parser.add_argument('--backend', type=str, default='vllm', help='model backend')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_multihop_inference(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        passage_path=args.passage_path,
        embedding_path=args.embedding_path,
        heads_json=args.heads_json,
        max_hops=args.max_hops,
        topk=args.topk,
        compress_threshold=args.compress_threshold,
        backend=args.backend,
    )
