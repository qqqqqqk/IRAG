from mteb import MTEB
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, LlamaConfig, AutoModelForCausalLM
import torch
from torch import nn
from transformers import BertLMHeadModel, BertTokenizer, RobertaTokenizer, RobertaForMaskedLM, RobertaModel, BertModel
from tqdm import tqdm
import json
from transformers import LlamaTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from models import Value_Aggregation_Eval, InforNCE_and_Eigenvalue_Eval, InforNCE_and_Generative_Hops_Eval
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F
import os
#import deepspeed
from datasets import Dataset
from arguments_va_eval import ModelArguments, DataTrainingArguments, TrainingArguments
# from mteb.models.text_formatting_utils import corpus_to_texts
# from mteb.encoder_interface import PromptType
import torch
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import font_manager
import math
from collections import Counter, defaultdict
# Add the TTC font to the font manager
font_path = '/usr/share/fonts/wqy-microhei.ttc'
font_manager.fontManager.addfont(font_path)

# Set the font for matplotlib
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Use the font name from the TTC file


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


use_auth_token = os.getenv("HUGGING_FACE_TOKEN")
class MyModel():
    def __init__(self, model, tokenizer, data_args, each_task, model2=None) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.model = model
        self.gpu_count = torch.cuda.device_count() 
        self.max_length = data_args.max_length
        self.batch_size = 4
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()
        self.whole_length = 0
        self.global_count = 0
        self.lm_head = self.model.lm_head
        self.is_query = False
        if model2 is not None:
            self.model = model2
            

    def _build_bm25_index(self, tokenized_corpus):
        num_docs = len(tokenized_corpus)
        doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        avgdl = float(doc_lens.mean()) if num_docs > 0 else 0.0

        term_freqs = []
        doc_freqs = defaultdict(int)

        for doc in tokenized_corpus:
            tf = Counter(doc)
            term_freqs.append(tf)
            for term in tf:
                doc_freqs[str(term)] += 1

        idf = {}
        for term, df in doc_freqs.items():
            idf[term] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

        return term_freqs, idf, doc_lens, avgdl

    def _bm25_score_query(self, query_sentence, idf, doc_lens, avgdl, k1=1.5, b=0.75):
        all_term = query_sentence[0]
        now_idf = torch.zeros([1, self.vocab_size]).cuda()
        now_tf = torch.zeros([1, self.vocab_size]).cuda()
        for term in all_term:
            now_idf[0, term] = idf[str(term.item())]
            now_tf[0, term] += 1.0
        norm = k1 * (1.0 - b + b * len(all_term) / (avgdl + 1e-12))
        sparse_embedding = now_idf * (now_tf * (k1 + 1.0)) / (now_tf + norm)
        final_embedding = sparse_embedding @ self.lm_head.weight[:151643]
        return final_embedding

    def compute_bm25(self, sentences, k1=1.5, b=0.75):
        """
        直接使用 self.tokenizer 对输入 sentences 分词。
        对于每个 sentence：
        1. 用 self.tokenizer(...)[\"input_ids\"] 切成 token ids
        2. 把 token id 转成字符串，作为 BM25 的 term
        """
        num_sentences = len(sentences)
        if num_sentences == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        tokenized_corpus = [
            [tid for tid in self.tokenizer(
                s,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]]
            for s in sentences
        ]
        term_freqs, idf, doc_lens, avgdl = self._build_bm25_index(tokenized_corpus)

        return term_freqs, idf, doc_lens, avgdl
    
    def encode(self, sentences, **kwargs):
        raw_sentences = list(sentences)

        term_freqs, idf, doc_lens, avgdl = self.compute_bm25(raw_sentences)
        self.last_term_freqs = term_freqs
        self.idf = idf
        self.last_doc_lens = doc_lens
        self.last_avgdl = avgdl
        self.global_count += 1
        with torch.no_grad():
            output_embedding = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches"):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                output_embedding.append(self.bm25_embedding(sentences_batch).cpu())
            return torch.cat(output_embedding, dim=0)
    
    

    def bm25_embedding(self, sentences_batch):
        whole_embeddings = []
        for every_sentence in sentences_batch:
            inputs = self.tokenizer([every_sentence], add_special_tokens=False, padding=False, max_length=self.max_length, return_tensors='pt')
            now_embedding = self._bm25_score_query(inputs['input_ids'], self.idf, inputs["attention_mask"].sum(dim=1), self.last_avgdl)
            whole_embeddings.append(now_embedding)
        return torch.cat(whole_embeddings, dim=0)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args.model_name = 'Qwen/Qwen3-8B'
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=use_auth_token,
    add_eos_token=True)
if 'llama' in model_args.model_name.lower(): 
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_args.model_name, use_auth_token=use_auth_token).cuda()
for each_task in ['SICK-R']:#['Banking77Classification', 'EmotionClassification','ArguAna', 'SciFact', 'STS17', 'NFCorpus', 'SICK-R', 'STSBenchmark', 'MedrxivClusteringS2S', 'StackOverflowDupQuestions', 'TwentyNewsgroupsClustering', 'BiorxivClusteringS2S', 'SciDocsRR', 'SprintDuplicateQuestions']:
    mymodel = MyModel(model, tokenizer, data_args, each_task)
    evaluation = MTEB(tasks=[each_task])
    results = evaluation.run(mymodel, output_folder=f"5_2_qwen3_8b_bm25_transform", eval_splits=["test"])