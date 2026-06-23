#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import os
import logging
import argparse
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm

from prompt_templates import PROMPT_TEMPLATES
from utils.model_utils import load_model
from utils.config_utils import ConfigManager
from multihop_inference import process_dataset

import sys
from retrievers import Retriever
from compressor import Compressor
from reranker import Reranker

logger = logging.getLogger(__name__)


def parse_batch_args():
    parser = argparse.ArgumentParser(description='Batch Multihop Inference')
    parser.add_argument('--config', type=str, default='multihop_config.yaml',
                       help='base config file path')
    parser.add_argument('--model_name', type=str, required=True,
                       help='model name or path')
    # parser.add_argument('--compress_model_name', type=str, required=True,
    #                    help='compress model name or path')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='dataset name')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='dataset path')
    parser.add_argument('--max_hops', type=str, default="5",
                       help='max hops list')
    parser.add_argument('--topk', type=str, default="10,20,40,80",
                       help='top-k list')
    parser.add_argument('--rerank_threshold', type=str, default="0.2",
                       help='rerank threshold list')
    parser.add_argument('--backend', type=str, default="vllm",
                       help='model backend type')
    parser.add_argument('--passage_path', type=str, default=None,
                       help='passage path')
    parser.add_argument('--embedding_path', type=str, default=None,
                       help='embedding path')
    parser.add_argument('--heads_json', type=str, default=None,
                       help='heads json path')
    parser.add_argument('--retriever_model_type', type=str, default=None)
    parser.add_argument("--retriever_model_path", type=str, default=None)
    parser.add_argument('--reranker_model_name', type=str, required=True)
    
    return parser.parse_args()


def get_multihop_output_path(config: Dict[str, Any], max_hops: int, topk: int, rerank_threshold: float) -> str:
    """generate multihop output path from config"""
    output_config = config.get('output', {})
    directory_pattern = output_config.get('directory', './results_multihop/{dataset_name}/{model_name}')
    filename_pattern = output_config.get('filename_pattern', f'multihop_hops{max_hops}_topk{topk}_rerank{rerank_threshold}.jsonl')
    
    replacements = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model_name': config['model']['name'].replace('/', '_'),
        'model_backend': config['model']['backend'],
        'dataset_name': config.get('dataset', {}).get('name', 'musique'),
        'max_hops': max_hops,
        'topk': topk,
        'rerank_threshold': rerank_threshold,
    }
    
    filename = filename_pattern
    directory = directory_pattern
    for key, value in replacements.items():
        placeholder = f"{{{key}}}"
        filename = filename.replace(placeholder, str(value))
        directory = directory.replace(placeholder, str(value))
    
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def process_one_combination(
    model,
    retriever: Retriever,
    reranker: Reranker,
    config_manager: ConfigManager,
    dataset: List[Dict[str, Any]],
    max_hops: int,
    topk: int,
    rerank_threshold: float
) -> bool:
    """process one combination of max_hops, topk, rerank_threshold"""
    config = config_manager.config
    
    logger.info("=" * 60)
    logger.info(f"processing combination: max_hops={max_hops}, topk={topk}, rerank_threshold={rerank_threshold}")
    logger.info(f"model name: {config['model']['name']}")
    logger.info(f"reranker model name: {config['compression']['reranker_model_name']}")
    logger.info(f"heads json: {config['compression']['heads_json']}")
    logger.info("=" * 60)
    
    system_message = config['prompt'].get('system_message', 'You are a helpful assistant.')
    
    output_path = get_multihop_output_path(config, max_hops, topk, rerank_threshold)
    logger.info(f"output path:{output_path}")
    
    try:
        results = process_dataset(
            model=model,
            dataset=dataset,
            retriever=retriever,
            reranker=reranker,
            output_path=output_path,
            max_hops=max_hops,
            topk=topk,
            rerank_threshold=rerank_threshold,
            system_message=system_message,
            use_progress_bar=config['processing']['use_progress_bar'],
        )
        
        logger.info(f"✓ combination completed: max_hops={max_hops}, topk={topk}, rerank_threshold={rerank_threshold}, processed {len(results)} samples")
        return True
        
    except Exception as e:
        logger.error(f"✗ combination failed: max_hops={max_hops}, topk={topk}, rerank_threshold={rerank_threshold}, error: {e}")
        import traceback
        traceback.print_exc()
        return False


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
    
    if args.heads_json is not None:
        config_manager.config['compression']['heads_json'] = args.heads_json

    if args.max_hops is not None:
        max_hops_list = [int(h.strip()) for h in args.max_hops.split(',')]
    else:
        max_hops_config = config_manager.config['multihop']['max_hops']
        max_hops_list = [int(h.strip()) for h in max_hops_config.split(',')]
    
    if args.topk is not None:
        topk_list = [int(k.strip()) for k in args.topk.split(',')]
    else:
        topk_config = config_manager.config['retrieval']['topk']
        topk_list = [int(k.strip()) for k in topk_config.split(',')]
    
    if args.rerank_threshold is not None:
        rerank_threshold_list = [float(t.strip()) for t in args.rerank_threshold.split(',')]
    else:
        rerank_threshold_config = config_manager.config['compression'].get('rerank_threshold', '0.2')
        rerank_threshold_list = [float(t.strip()) for t in rerank_threshold_config.split(',')]
    
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
    logger.info(f"heads json: {config_manager.config['compression']['heads_json']}")
    logger.info(f"dataset path: {config_manager.config['dataset']['path']}")
    logger.info(f"max_hops list: {max_hops_list}")
    logger.info(f"topk list: {topk_list}")
    logger.info(f"rerank_threshold list: {rerank_threshold_list}")
    logger.info(f"total combinations to process: {len(max_hops_list) * len(topk_list) * len(rerank_threshold_list)}")
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
        # compressor = Compressor(
        #     model_path=config_manager.config['compression']['compress_model_name'],
        #     heads_file=config_manager.config['compression']['heads_json']
        # )
        # logger.info("✓ compressor loaded successfully")
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
    
    total_combinations = len(max_hops_list) * len(topk_list) * len(rerank_threshold_list)
    fail = []
    
    for i, max_hops in enumerate(max_hops_list, 1):
        for j, topk in enumerate(topk_list, 1):
            for k, rerank_threshold in enumerate(rerank_threshold_list, 1):
                current = (i - 1) * len(topk_list) * len(rerank_threshold_list) + (j - 1) * len(rerank_threshold_list) + k
                logger.info(f"\n[{current}/{total_combinations}] processing...")
                
                success = process_one_combination(
                    model=model,
                    retriever=retriever,
                    reranker=reranker,
                    config_manager=config_manager,
                    dataset=dataset,
                    max_hops=max_hops,
                    topk=topk,
                    rerank_threshold=rerank_threshold,
                )
                
                if not success:
                    fail.append((max_hops, topk, rerank_threshold))
    
    logger.info("\n" + "=" * 80)
    logger.info("Task Completed!")
    if fail:
        for f in fail:
            max_h, tk, rt = f
            logger.info(f"Failed combination: max_hops={max_h}, topk={tk}, rerank_threshold={rt}")
    else:
        logger.info("All combinations completed successfully!")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
