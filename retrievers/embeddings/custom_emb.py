from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, LlamaConfig, AutoModelForCausalLM
import torch
from .embedding_model.models import InforNCE_and_Generative_Hops_Eval
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
import os

from .dense_embedding import DenseEmbedding

class CustomEmbedding(DenseEmbedding):
    def __init__(
        self,
        model_name_or_path: str,
    ):
        super().__init__(model_name_or_path, embedding_vector_size=2048)
        base_model_path, checkpoint_path = model_name_or_path.split("&")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
        special_tokens = ["[STOP_SEARCH]", "[SUFFICIENT_EVIDENCE]", "[ANSWER_READY]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.tokenizer.padding_side = "left"
        
        self.model = InforNCE_and_Generative_Hops_Eval(3, base_model_path)
        self.model.model.encoder.base_model.model.resize_token_embeddings(len(self.tokenizer))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['module'], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.model.to(torch.bfloat16)

    def instantiate(self):
        return

    def embed_batch(self, queries: list[str], is_query=False):

        sentence = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        sentence = sentence.to(self.device)
        output = self.model(**sentence, is_query=is_query)
        
        return output