#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import os
import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import argparse

from prompt_templates import PROMPT_TEMPLATES
from utils.model_utils import load_model, BaseModel, calculate_entropy_from_logprobs
from compressor import Compressor

import sys
from retrievers import Retriever
from reranker import Reranker

from utils.model_utils import load_model
from utils.config_utils import ConfigManager
from multihop_inference import process_dataset

logger = logging.getLogger(__name__)

def format_context(qa_history: List[Dict[str, str]], original_question: str) -> str:
    context_parts = [f"original question: {original_question}"]
    retrieved_contexts = "\n\n".join([f"[Doc {i+1}] : {doc['text']}" for i, doc in enumerate(qa['sub_question'])])
    for i, qa in enumerate(qa_history):
        context_parts.append(f"follow up sub-question: {qa['sub_question']}")
        context_parts.append(f"retrieved documents: {retrieved_contexts}")
        context_parts.append(f"intermediate answer: {qa['sub_answer']}")
    
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
    reranker: Optional[Reranker] = None,
    max_hops: int = 8,
    topk: int = 80,
    rerank_topk: int = 20,
    system_message: str = "You are a helpful assistant.",
) -> Dict[str, Any]:
    """Process a single sample with multiihop inference."""

    sample_id = sample["question_id"]
    original_question = sample["question_text"]
    # gold_answer = sample["answers_objects"][0]["spans"]

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
        context_judge_output = model.generate(formatted_prompt, max_tokens=10240) 

        context_judge_response = context_judge_output["text"]
        can_answer, content = parse_context_judge_response(context_judge_response)

        if can_answer:
            final_answer = content
            break

        sub_question = content
        if not sub_question:
            sub_question = original_question

        if retriever is not None:
            
            retrieved_docs = retriever.search(sub_question, top_k=topk)
            if retrieved_docs and len(retrieved_docs) > 0:
                docs_for_query = retrieved_docs[0] if isinstance(retrieved_docs[0], list) else retrieved_docs
                
                if reranker is not None:
                    reranked_docs = reranker.rerank_topk(
                        question=sub_question,
                        documents=docs_for_query[:topk],
                        top_k=rerank_topk,
                        batch_size=1 
                    )
                    reranked_docs = [doc for doc in reranked_docs if doc['text'].strip()]
                    docs_for_query = reranked_docs
                
                retrieved_context = "\n\n".join([
                    f"[Doc {i+1}] : {doc['text']}"
                    for i, doc in enumerate(docs_for_query[:topk])
                ])
            else:
                raise ValueError("No retrieved documents found.")
            # except Exception as e:
            #     raise ValueError(f"Retrieval failed: {e}")
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
        sub_q_answer_output = model.generate(formatted_prompt, max_tokens=10240)

        sub_q_answer_response = sub_q_answer_output["text"]
        sub_answer, answer_success = parse_sub_q_answer_response(sub_q_answer_response)
        if not answer_success:
            sub_answer = "No relevant info, need to optimize sub-question"

        # TODO: 只保存 retrieval docs
        qa_history.append({
            "sub_question": sub_question,
            "retrieved_docs": docs_for_query[:topk],
            "sub_answer": sub_answer
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

        final_output = model.generate(formatted_prompt, max_tokens=10240)
        final_response = final_output["text"]
        final_answer = parse_force_answer_response(final_response)
    

    result = {
        "sample_id": sample_id,
        "original_question": original_question,
        # "gold_answer": gold_answer,
        "qa_history": qa_history,
        "predicted_answer": final_answer,
    }
    
    return result

def process_single_sample_custom(
    model: BaseModel,
    sample: Dict[str, Any],
    retriever: Optional[Retriever],
    reranker: Optional[Reranker] = None,
    max_hops: int = 8,
    topk: int = 80,
    rerank_topk: int = 20,
    system_message: str = "You are a helpful assistant.",
) -> Dict[str, Any]:
    """Process a single sample with multiihop inference."""

    sample_id = sample["question_id"]
    original_question = sample["question_text"]
    # gold_answer = sample["answers_objects"][0]["spans"]

    qa_history = []
    
    context_judge_prompt = PROMPT_TEMPLATES["context_judge"]
    sub_q_judge_prompt = PROMPT_TEMPLATES["answer_without_context"]
    sub_q_answer_prompt = PROMPT_TEMPLATES["sub_q_answer"]
    force_answer_prompt = PROMPT_TEMPLATES["force_answer"]
    
    final_answer = None
    
    for hop_idx in range(max_hops):
        if qa_history:  
            context = "\n\n".join([
                    f"[Doc {i+1}] : {doc['text']}"
                    for i, doc in enumerate(qa) for qa in enumerate(qa_history)
                ])
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
        context_judge_output = model.generate(formatted_prompt, max_tokens=10240) 

        context_judge_response = context_judge_output["text"]
        can_answer, content = parse_context_judge_response(context_judge_response)

        if can_answer:
            final_answer = content
            break

        if retriever is not None:
            
            retrieved_docs = retriever.search(original_question, top_k=topk)
            if retrieved_docs and len(retrieved_docs) > 0:
                docs_for_query = retrieved_docs[0] if isinstance(retrieved_docs[0], list) else retrieved_docs
                
                if reranker is not None:
                    reranked_docs = reranker.rerank_topk(
                        question=original_question,
                        documents=docs_for_query[:topk],
                        top_k=rerank_topk,
                        batch_size=1 
                    )
                    reranked_docs = [doc for doc in reranked_docs if doc['text'].strip()]
                    docs_for_query = reranked_docs
                
                retrieved_context = "\n\n".join([
                    f"[Doc {i+1}] : {doc['text']}"
                    for i, doc in enumerate(docs_for_query[:topk])
                ])
            else:
                raise ValueError("No retrieved documents found.")
            # except Exception as e:
            #     raise ValueError(f"Retrieval failed: {e}")
        else:
            raise ValueError("Retriever not available.")

        qa_history.append({
            "retrieved_docs": docs_for_query[:topk],
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

        final_output = model.generate(formatted_prompt, max_tokens=10240)
        final_response = final_output["text"]
        final_answer = parse_force_answer_response(final_response)
    

    result = {
        "sample_id": sample_id,
        "original_question": original_question,
        # "gold_answer": gold_answer,
        "qa_history": qa_history,
        "predicted_answer": final_answer,
    }
    
    return result

def parse_batch_args():
    parser = argparse.ArgumentParser(description='MiniMemOS Inference')
    parser.add_argument('--config', type=str, default='multihop_config.yaml',
                       help='base config file path')
    parser.add_argument('--model_name', type=str, required=True,
                       help='model name or path')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='dataset name')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='dataset path')
    parser.add_argument('--backend', type=str, default="vllm",
                       help='model backend type')
    parser.add_argument('--passage_path', type=str, default=None,
                       help='passage path')
    parser.add_argument('--embedding_path', type=str, default=None,
                       help='embedding path')
    parser.add_argument('--retriever_model_type', type=str, default=None)
    parser.add_argument("--retriever_model_path", type=str, default=None)
    parser.add_argument('--reranker_model_name', type=str, required=True)
    
    return parser.parse_args()

def main():
    args = parse_batch_args()

    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()

    config_manager.config['model']['name'] = args.model_name
    
    if args.backend is not None:
        config_manager.config['model']['backend'] = args.backend
    
    if args.dataset_path is not None:
        config_manager.config['dataset']['path'] = args.dataset_path
    
    if args.passage_path is not None:
        config_manager.config['retrieval']['passage_path'] = args.passage_path
    
    if args.embedding_path is not None:
        config_manager.config['retrieval']['embedding_path'] = args.embedding_path
    
    if 'compression' not in config_manager.config:
        config_manager.config['compression'] = {}
    
    config_manager.config['compression']['reranker_model_name'] = args.reranker_model_name
    
    if args.retriever_model_type is not None:
        config_manager.config['retrieval']["model_type"] = args.retriever_model_type

    if args.retriever_model_path is not None:
        config_manager.config['retrieval']["model_path"] = args.retriever_model_path

    if args.dataset_name is not None:
        config_manager.config['dataset']["name"] = args.dataset_name

    config_manager.setup_logging()
    
    logger.info("=" * 80)
    logger.info("Batch Multihop Inference Task")
    logger.info(f"model name: {config_manager.config['model']['name']}")
    logger.info(f"compress model name: {config_manager.config['compression']['compress_model_name']}")
    logger.info(f"dataset path: {config_manager.config['dataset']['path']}")
    logger.info("=" * 80)

    logger.info(f"loading model: {args.model_name} ...")
    try:
        model_kwargs = {
            k: v for k, v in config_manager.config['model'].items() 
            if k not in ['name', 'backend']
        }
        
        model = load_model(
            model_name=config_manager.config['model']['name'],
            backend=config_manager.config['model']['backend'],
            **model_kwargs
        )
        logger.info("✓ model loaded successfully")
    except Exception as e:
        logger.error(f"✗ model loading failed: {e}")
        return
    
    logger.info("loading retriever...")
    try:
        retriever = Retriever(
            passage_path=config_manager.config['retrieval']['passage_path'],
            passage_embedding_path=config_manager.config['retrieval']['embedding_path'],
            index_path_dir=config_manager.config['retrieval']['embedding_path'],
            model_type=config_manager.config['retrieval'].get('model_type', 'e5-large-v2'),
            model_path=config_manager.config['retrieval'].get('model_path', "/data/lzb/models/e5-large-v2")
        )
        logger.info("✓ retriever loaded successfully")
    except Exception as e:
        logger.error(f"✗ retriever loading failed: {e}")
        return

    logger.info("loading compressor...")
    try:
        reranker = Reranker(
            model_name=args.reranker_model_name
        )
        logger.info("✓ reranker loaded successfully")
    except Exception as e:
        # logger.error(f"✗ compressor loading failed: {e}")
        logger.error(f"✗ reranker loading failed: {e}")
        return
    
    dataset_path = config_manager.config['dataset']['path']
    logger.info(f"loading dataset: {dataset_path} ...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        logger.info(f"✓ dataset loaded successfully, total {len(dataset)} samples")
    except Exception as e:
        logger.error(f"✗ dataset loading failed: {e}")
        return

    output_path = f"./results_minimemos/{args.dataset_name}/{args.model_name}"
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, f"result.jsonl")
    with open(output_file_path, "r") as f:
        start_idx = len(f.readlines())
    output_file = open(output_file_path, "a")            
    for sample in tqdm(dataset[start_idx:], initial=start_idx, total=len(dataset)):
        if args.retreiver_model_type == "custom":
            result = process_single_sample_custom(
                model=model,
                sample=sample,
                retriever=retriever,
                reranker=reranker,
            )
        else:
            result = process_single_sample(
                model=model,
                sample=sample,
                retriever=retriever,
                reranker=reranker,
            )
        output_file.write(json.dumps(result, ensure_ascii=False)+ "\n")
        output_file.flush()
    output_file.close()
                    
    logger.info("\n" + "=" * 80)
    logger.info("Task Completed!")
    
    logger.info("Task finished!")
    
    logger.info("=" * 80)

if __name__ == '__main__':
    main()