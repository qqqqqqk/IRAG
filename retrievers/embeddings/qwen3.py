import torch
from torch._tensor import Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .dense_embedding import DenseEmbedding

class Qwen3Embedding(DenseEmbedding):
    def __init__(
        self,
        model_name_or_path: str,
    ):
        super().__init__(model_name_or_path, embedding_vector_size=2560)
        self.model = SentenceTransformer(model_name_or_path, device=self.device)

    # def instantiate(self):
    #     self.model = SentenceTransformer(self.model_name_or_path, device=self.device)
    #     self.model.eval()
        # if self.fp16 and self.device != "cpu":
        #     self.model = self.model.half()

    def instantiate(self):
        return

    def embed_batch(self, queries: list[str], is_query=False):
        if is_query:
            embeddings = self.model.encode(queries, prompt_name="query", show_progress_bar=False)
        else:
            embeddings = self.model.encode(queries, show_progress_bar=False)
        
        return embeddings