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


task_to_prompt = {
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim:",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question:",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim:",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query:",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia:",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question:",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question:",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper:",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query:",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question:",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim:",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question:",
    "ArguAna": "Given a claim, find documents that refute the claim:",
    "CQADupstackTexRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWebmastersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackEnglishRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGisRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackMathematicaRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackStatsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackPhysicsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackProgrammersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackAndroidRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWordpressRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question:",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum:",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum:",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers:",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history:",
    "Banking77Classification": "Given a online banking query, find the corresponding intents:",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise:",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral:",
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or notcounterfactual:",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset:",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents:",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios:",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation:",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation:",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic:",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment:",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category:",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles:",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles:",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles:",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts:",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles:",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts:",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts:",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles:",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts:",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles:",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs:",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet:",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet:",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum:",
    "STS12": "Retrieve semantically similar text:",
    "STS13": "Retrieve semantically similar text:",
    "STS14": "Retrieve semantically similar text:",
    "STS15": "Retrieve semantically similar text:",
    "STS16": "Retrieve semantically similar text:",
    "STS17": "Retrieve semantically similar text:",
    "STS22": "Retrieve semantically similar text:",
    "BIOSSES": "Retrieve semantically similar text:",
    "SICK-R": "Retrieve semantically similar text:",
    "STSBenchmark": "Retrieve semantically similar text:",
    "SummEval": "Given a news summary, retrieve other semantically similar summaries:"
}
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
        self.prompt = task_to_prompt[each_task]
        self.whole_length = 0
        self.global_count = 0
        self.lm_head = self.model.model.encoder.base_model.lm_head
        self.is_query = False
        # self.A_pinv = torch.linalg.pinv(self.lm_head.weight.T) 
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

        # for doc_idx, tf in enumerate(term_freqs):
        #     dl = doc_lens[doc_idx]
        #     norm = k1 * (1.0 - b + b * dl / (avgdl + 1e-12))
        #     score = 0.0

        #     for term in query_tokens:
        #         if term not in tf:
        #             continue
        #         freq = tf[term]
        #         score += idf.get(term, 0.0) * (freq * (k1 + 1.0)) / (freq + norm)

        #     scores[doc_idx] = score

        # return scores

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
        # if kwargs["prompt_name"] in ["ArguAna", "SciFact", "NFCorpus"]:
        #     if self.global_count == 0:
        #         sentences = [self.prompt + '\n' + text for text in sentences]
        #         self.is_query = True
        # else:
        #     sentences = [self.prompt + '\n' + text for text in sentences]
        self.global_count += 1
        with torch.no_grad():
            output_embedding = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches"):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                output_embedding.append(self.bm25_embedding(sentences_batch).cpu())
            return torch.cat(output_embedding, dim=0)
    
    
    def value_aggregation(self, sentences_batch, is_query=False):
        inputs = self.tokenizer(sentences_batch, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
        inputs = {k:v.cuda() for k,v in inputs.items()}
        embeddings = self.model(inputs['input_ids'], inputs['attention_mask'], is_query=is_query)
        return embeddings.to(torch.float32)


    def n_steps_embedding(self, sentences_batch, is_query=True):
        sentences_batch[0] = "<query>: What is the nationality of the author of the Harry Potter?"
        # sentences_batch[0] = "长沙的特色菜？"
        inputs = self.tokenizer(sentences_batch, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
        inputs = {k:v.cuda() for k,v in inputs.items()}
        n_step_embeddings = self.model(inputs['input_ids'], inputs['attention_mask'], is_query)
        logits = self.lm_head(n_step_embeddings)
        v, i = torch.topk(logits, k=100)
        tokens = [self.tokenizer.decode(index).replace(' ', '_') for index in i[0]]
        import pdb
        pdb.set_trace()
        return n_step_embeddings

    def qwen3_embedding_eval(self, sentences_batch):
        # Tokenize the input texts
        batch_dict = tokenizer(
            sentences_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings

    def bm25_embedding(self, sentences_batch):
        whole_embeddings = []
        for every_sentence in sentences_batch:
            inputs = self.tokenizer([every_sentence], add_special_tokens=False, padding=False, max_length=self.max_length, return_tensors='pt')
            now_embedding = self._bm25_score_query(inputs['input_ids'], self.idf, inputs["attention_mask"].sum(dim=1), self.last_avgdl)
            whole_embeddings.append(now_embedding)
        return torch.cat(whole_embeddings, dim=0)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=use_auth_token,
    add_eos_token=True)
special_tokens = ["[STOP_SEARCH]", "[SUFFICIENT_EVIDENCE]", "[ANSWER_READY]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
tokenizer.padding_side = "left"
if 'llama' in model_args.model_name.lower(): 
    tokenizer.pad_token = tokenizer.eos_token
model = InforNCE_and_Generative_Hops_Eval(training_args.local_rank, model_args.model_name) # InforNCE_and_Eigenvalue_Eval(training_args.local_rank, model_args.model_name)
model.model.encoder.base_model.model.resize_token_embeddings(len(tokenizer))
checkpoint_path = model_args.checkpoint_path
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['module'], strict=True)
for each_task in ['SciFact']:#['Banking77Classification', 'EmotionClassification','ArguAna', 'SciFact', 'STS17', 'NFCorpus', 'SICK-R', 'STSBenchmark', 'MedrxivClusteringS2S', 'StackOverflowDupQuestions', 'TwentyNewsgroupsClustering', 'BiorxivClusteringS2S', 'SciDocsRR', 'SprintDuplicateQuestions']:
    mymodel = MyModel(model, tokenizer, data_args, each_task)
    evaluation = MTEB(tasks=[each_task])
    results = evaluation.run(mymodel, output_folder=f"5_2_qwen3_0.6b_bm25_transform", eval_splits=["test"])
# model2 = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
# for each_task in ['SICK-R']:
#     mymodel = MyModel(model, tokenizer, data_args, each_task)
#     evaluation = MTEB(tasks=[each_task])
#     results = evaluation.run(mymodel, output_folder=f"3_28_qwen3_0.6b_step200_lambda_0_test", eval_splits=["test"])