import os
import pickle
from typing import List, Tuple
from typing import Literal
from tqdm import tqdm

import faiss
import numpy as np

from .base import BaseIndex

IndexType = Literal[
    "Flat",
    "HNSW64",
    "IVF100,Flat",
    "PQ16",
    "IVF100,PQ16",
    "LSH",
]


class FaissIndex(BaseIndex):
    def __init__(
        self,
        dim: int = 768,
        index_type: IndexType = "Flat",
        max_search_batch_size: int = 2048,
        max_index_batch_size: int = int(1e6),
    ):
        super().__init__()
        self.index_fname = "index.faiss"
        self.index_meta_fname = "index_meta.faiss"

        self.index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
        self.idx2db = []

        self.max_search_batch_size = max_search_batch_size
        self.max_index_batch_size = max_index_batch_size

    def search(self, query_vectors: np.array, top_k: int = 20) -> List[Tuple[List[object], List[float]]]:
        query_vectors = np.asarray(query_vectors, dtype=np.float32)  # 更稳健的类型转换
        result = []
        batches = (len(query_vectors) - 1) // self.max_search_batch_size + 1
        for idx in range(batches):
            start_idx = idx * self.max_search_batch_size
            end_idx = min((idx + 1) * self.max_search_batch_size, len(query_vectors))
            q = query_vectors[start_idx:end_idx]
            scores, indexes = self.index.search(q, top_k)
            # convert index to passage id
            db_ids = [[str(self.idx2db[i]) for i in query_indexes] for query_indexes in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, self.index_fname)
        meta_file = os.path.join(dir_path, self.index_meta_fname)
        print(f"Serializing index to {index_file}, meta data to {meta_file}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        faiss.write_index(self.index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump(self.idx2db, f)

    def deserialize(self, dir_path):
        index_file = os.path.join(dir_path, self.index_fname)
        meta_file = os.path.join(dir_path, self.index_meta_fname)
        print(f"Loading index from {index_file}, meta data from {meta_file}")
        self.index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            self.idx2db = pickle.load(f)
        assert len(self.idx2db) == self.index.ntotal, "Deserialized idx2db should match faiss index size"

    def exist_index(self, dir_path):
        index_file = os.path.join(dir_path, self.index_fname)
        meta_file = os.path.join(dir_path, self.index_meta_fname)
        return os.path.exists(index_file) and os.path.exists(meta_file)

    def load_data(self, passage_embeddings: List[str]):
        embeddings = np.array([])
        ids = []
        id_counter = 0
        for fpath in tqdm(passage_embeddings, desc="Load embeddings"):
            with open(fpath, "rb") as fin:
                data = pickle.load(fin)
                # 兼容两种格式：
                # 1. 旧格式: (cur_ids, cur_embeddings) 元组
                # 2. 新格式: 只有 cur_embeddings (numpy.ndarray)
                if isinstance(data, tuple) and len(data) == 2:
                    cur_ids, cur_embeddings = data
                    ids.extend(cur_ids)
                else:
                    # 新格式：只有 embeddings，自动生成 ids
                    cur_embeddings = data
                    num_embeddings = cur_embeddings.shape[0] if len(cur_embeddings.shape) == 2 else 1
                    cur_ids = list(range(id_counter, id_counter + num_embeddings))
                    ids.extend(cur_ids)
                    id_counter += num_embeddings
                embeddings = np.vstack((embeddings, cur_embeddings)) if embeddings.size else cur_embeddings
        embeddings = embeddings.astype("float32")
        self.idx2db = ids
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        print(f"Total data indexed {len(self.idx2db)}")
