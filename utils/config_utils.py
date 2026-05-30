# config_utils.py
import yaml
import os
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import argparse

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self.default_config = self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "model": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "backend": "vllm",
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "dtype": "float16",
                "trust_remote_code": True,
            },
            "generation": {
                "max_tokens": 128,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "do_sample": True,
            },
            "dataset": {
                "name": "musique",
                "path": "datasets/musique/correct10_subQ_1ans_dev.jsonl",
                "split": "200_an_empirical",
                "cache_dir": "./datasets/musique",
            },
            "experiment":{
                "type": "0",
                "hops": "0",
                "cache_dir": "./empirical/context/entropy_train",
            },
            "retrieval": {
                "passage_path": "datacorpus/musique/corpus_renumbered.jsonl",
                "embedding_path": "datacorpus/musique/contriever_re",
                "topk": 3,
                "model_type": "contriever",
            },
            "multihop": {
                "max_hops": 8,
            },
            "prompt": {
                "template_name": "answer",
                "system_message": "You are a helpful assistant.",
            },
            "processing": {
                "use_progress_bar": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,
            }
        }
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = self.config_path
            
        if config_path is None or not os.path.exists(config_path):
            logger.warning(f"config file not found: {config_path}, use default config")
            self.config = self.default_config
            return self.config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            self.config = self._deep_merge(self.default_config, loaded_config)
            logger.info(f"config file loaded: {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"load config file failed: {e}")
            self.config = self.default_config
            return self.config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        arg_dict = vars(args)
        
        arg_mapping = {
            'model_name': ('model', 'name'),
            'model_backend': ('model', 'backend'),
            'tensor_parallel_size': ('model', 'tensor_parallel_size'),
            'gpu_memory_utilization': ('model', 'gpu_memory_utilization'),
            'prompt_template': ('prompt', 'template_name'),
            'dataset_name': ('dataset', 'name'),
            'dataset_split': ('dataset', 'split'),
            'batch_size': ('processing', 'batch_size'),
            'max_workers': ('processing', 'max_workers'),
            'temperature': ('generation', 'temperature'),
            'output_file': ('output', 'filename'),
            'experiment_type': ('experiment', 'type'),
            'experiment_hops': ('experiment', 'hops'),
        }
        
        for arg_name, config_path in arg_mapping.items():
            if arg_name in arg_dict and arg_dict[arg_name] is not None:
                if isinstance(config_path, tuple):
                    current = self.config
                    for key in config_path[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[config_path[-1]] = arg_dict[arg_name]
                else:
                    self.config[config_path] = arg_dict[arg_name]
    
    def get_output_path(self, **kwargs) -> str:
        output_config = self.config.get('output', {})
        directory_pattern = output_config.get('directory', 'results_empirical')

        filename_pattern = output_config.get('filename_pattern', 'results.jsonl')
        
        replacements = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_name': self.config['model']['name'].replace('/', '_'),
            'model_backend': self.config['model']['backend'],
            'prompt_template': self.config['prompt']['template_name'],
            'dataset_name': self.config['dataset']['name'],
            'dataset_split': self.config['dataset']['split'],
            'experiment_type': self.config['experiment']['type'],
            'experiment_hops': self.config['experiment']['hops'],
            **kwargs
        }

        filename = filename_pattern
        directory = directory_pattern
        for key, value in replacements.items():
            placeholder = f"{{{key}}}"
            filename = filename.replace(placeholder, str(value))
            directory = directory.replace(placeholder, str(value))
        
        os.makedirs(directory, exist_ok=True)
        
        return os.path.join(directory, filename)
    
    def setup_logging(self) -> None:
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handlers = [logging.StreamHandler()]
        log_file = log_config.get('file')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=handlers
        )
    
    def save_config(self, save_path: Optional[str] = None) -> None:
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"configs/config_{timestamp}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Config saved to: {save_path}")


def parse_args_with_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Model Inference Parameters')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_name', type=str, 
                       help='Model name or path')
    parser.add_argument('--model_backend', type=str, 
                       choices=['vllm', 'transformers'],
                       help='Model backend')
    
    # vLLM parameters
    parser.add_argument('--tensor_parallel_size', type=int,
                       help='vLLM tensor parallel size')
    parser.add_argument('--gpu_memory_utilization', type=float,
                       help='GPU memory utilization')
    
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str,
                       help='Dataset name')
    parser.add_argument('--dataset_split', type=str,
                       help='Dataset split')

    # empirical experiment parameters
    parser.add_argument('--experiment_type', type=str,
                       help='Experiment type (0,1,2,3)')
    parser.add_argument('--experiment_hops', type=str,
                       help='Experiment hops')
    
    # Prompt parameters
    parser.add_argument('--prompt_template', type=str,
                       help='Prompt template name')
    
    # Generation parameters
    parser.add_argument('--max_tokens', type=int,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float,
                       help='Generation temperature')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for generation')
    
    # Output parameters
    parser.add_argument('--output_file', type=str,
                       help='Output file path (overrides auto-generated path)')
    
    # Experiment parameters
    parser.add_argument('--save_config', action='store_true',
                       help='Save configuration to file')
    
    return parser.parse_args()