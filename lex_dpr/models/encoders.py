# lex_dpr/models/encoders.py
from typing import Iterable
from sentence_transformers import SentenceTransformer
import numpy as np
from .templates import TemplateMode, tq, tp

class BiEncoder:
    def __init__(self, name_or_path: str, template: TemplateMode = TemplateMode.BGE,
                 normalize: bool = True, max_seq_length: Optional[int] = None,
                 trust_remote_code: bool = True):
        self.model = SentenceTransformer(name_or_path, trust_remote_code=trust_remote_code)
        if max_seq_length:
            # Sentence-Transformers가 내부 tokenizer 길이를 이 값으로 사용
            self.model.max_seq_length = int(max_seq_length)
        self.template = template
        self.normalize = normalize

    def encode_queries(self, queries: Iterable[str], batch_size=64) -> np.ndarray:
        texts = [tq(q, self.template) for q in queries]
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=self.normalize, show_progress_bar=False)

    def encode_passages(self, passages: Iterable[str], batch_size=64) -> np.ndarray:
        texts = [tp(p, self.template) for p in passages]
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=self.normalize, show_progress_bar=False)
