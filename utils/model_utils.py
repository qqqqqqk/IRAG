# model_utils.py
from typing import List, Optional, Dict, Any
import torch
import os
import math
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

def calculate_entropy_from_logprobs(
    logprobs: Dict[int, float],
    use_top_k: int = None, 
    base: int = math.e
) -> float:
    if not logprobs:
        return 0.0
    log_prob_values = list(logprobs.values())
    probs = [math.exp(lp) for lp in log_prob_values]

    if use_top_k is not None and use_top_k < len(probs):
        probs = probs[:use_top_k]

    total = sum(probs)
    if total ==0:
        return 0.0
    probs = [p / total for p in probs]

    entropy = 0.0
    for p in probs:
        if p > 0:
            if base == 2:
                entropy -= p * math.log2(p)
            elif base == 10:
                entropy -= p * math.log10(p)
            else: 
                entropy -= p * math.log(p)

    return entropy


class BaseModel:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # HuggingFace token
        self.hf_token = kwargs.get('hf_token', os.getenv('HF_TOKEN', ""))
        
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
        
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class VLLMModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.dtype = kwargs.get('dtype', 'fp16')
        self.trust_remote_code = kwargs.get('trust_remote_code', True)
        self.max_model_len = kwargs.get('max_model_len', None)
        self.max_logprobs = kwargs.get('max_logprobs', 20)
        
        logger.info(f"Loading vLLM model: {model_name}")
        llm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'dtype': self.dtype,
            'trust_remote_code': self.trust_remote_code,
            'max_logprobs': self.max_logprobs,
        }
        
        if self.max_model_len is not None:
            llm_kwargs['max_model_len'] = self.max_model_len
        
        self.llm = LLM(**llm_kwargs)
        
        tokenizer_kwargs = {
            'trust_remote_code': self.trust_remote_code,
            'fix_mistral_regex': True
        }
        if self.hf_token:
            tokenizer_kwargs['token'] = self.hf_token
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **tokenizer_kwargs
        )
        
        logger.info(f"Model loaded successfully on device: {self.device}")
    
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True 
            )
            return text
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using fallback formatting")
            raise
    
    def generate(self,
                prompt: str,
                max_tokens: int = 256,
                temperature: float = 0.6,#0.1
                top_p: float = 0.95,
                top_k: int = 20,
                repetition_penalty: float = 1.05,
                logprobs_top_k: int = 100,
                return_logprobs: bool = False,
                **kwargs) -> Dict[str, Any]:

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            logprobs=logprobs_top_k if return_logprobs else None,
            stop=kwargs.get('stop', None),
            **{k: v for k, v in kwargs.items() if k not in ['stop']}
        )

        try:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            output = outputs[0].outputs[0]

            generated_text = output.text.strip()
            
            token_logprobs = None
            token_ids = None
            token_texts = None
            
            if return_logprobs and output.logprobs:
                token_logprobs = []
                token_ids = []
                token_texts = []
                
                for i, token_id in enumerate(output.token_ids):
                    token_ids.append(token_id)

                    logprobs = output.logprobs[i]
                    serializable_logprobs = {}
                    for tid, lp_obj in logprobs.items():
                        serializable_logprobs[tid] = lp_obj.logprob
                    token_logprobs.append(serializable_logprobs)

                    token_text = self.tokenizer.decode([token_id])
                    token_texts.append(token_text)
            
            # QwQ-32B 
            is_qwq = 'QwQ-32B' in self.model_name or 'qwq-32b' in self.model_name.lower()
            
            if is_qwq:
                full_original_text = output.text
                cleaned_text = full_original_text
                if '</think>' in cleaned_text:
                    cleaned_text = cleaned_text.split('</think>', 1)[1]
                if '\n\n' in cleaned_text:
                    cleaned_text = cleaned_text.rsplit('\n\n', 1)[1]
                cleaned_text = re.sub(r'^.*?answer:\s*', '', cleaned_text, flags=re.IGNORECASE)
                cleaned_text = cleaned_text.strip()
                generated_text = cleaned_text
                
                if return_logprobs and token_texts:
                    full_token_text = ''.join(token_texts)
                    start_pos = full_token_text.find(cleaned_text)
                    if start_pos != -1:
                        cumulative_text = ""
                        start_idx = None
                        end_idx = None
                        
                        for i, token_text in enumerate(token_texts):
                            cumulative_text += token_text
                            if start_idx is None and len(cumulative_text) > start_pos:
                                start_idx = i
                            if start_idx is not None and cumulative_text == full_token_text[:start_pos + len(cleaned_text)]:
                                end_idx = i + 1
                                break
                        
                        if start_idx is not None and end_idx is not None and start_idx < end_idx:
                            token_logprobs = token_logprobs[start_idx:end_idx]
                            token_ids = token_ids[start_idx:end_idx]
                            token_texts = token_texts[start_idx:end_idx]
                        else:
                            logger.warning(
                                f"Could not find exact token boundaries for cleaned text. Keeping all tokens."
                            )
                    else:
                        logger.warning(
                            f"Cleaned text not found in full token text. Keeping all tokens."
                        )

            result = {"text": generated_text}
            
            if return_logprobs:
                result["token_logprobs"] = token_logprobs
                result["token_ids"] = token_ids
                result["token_texts"] = token_texts

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise