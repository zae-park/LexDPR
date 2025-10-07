
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from .encoder import TextEncoder

class DPRetriever:
    def __init__(self, query_encoder: TextEncoder, passage_encoder: TextEncoder, metric: str = "dot"):
        self.qe = query_encoder
        self.pe = passage_encoder
        self.metric = metric
        self.index = None
        self.passage_ids: List[str] = []

    def _similarity(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
            P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        return Q @ P.T  # (nq, np)

    def build_index(self, passage_embeddings: np.ndarray, passage_ids: List[str], factory: str = "Flat"):
        d = passage_embeddings.shape[1]
        if factory == "Flat":
            self.index = faiss.IndexFlatIP(d)
        else:
            self.index = faiss.index_factory(d, factory)
        # Normalize for cosine similarity if needed
        if self.metric == "cosine":
            norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True) + 1e-8
            passage_embeddings = passage_embeddings / norms
        self.index.add(passage_embeddings.astype(np.float32))
        self.passage_ids = passage_ids

    def search(self, query_embeddings: np.ndarray, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        Q = query_embeddings.astype(np.float32)
        if self.metric == "cosine":
            Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        scores, idx = self.index.search(Q, top_k)
        results: List[List[Tuple[str, float]]] = []
        for row_scores, row_idx in zip(scores, idx):
            items = [(self.passage_ids[i], float(s)) for i, s in zip(row_idx, row_scores)]
            results.append(items)
        return results
