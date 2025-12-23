"""
LexDPR: Legal Document Retriever & Reranker

사용 예시:
    from lex_dpr import BiEncoder
    
    # 모델 로드
    encoder = BiEncoder("path/to/model")
    
    # 질의 임베딩
    queries = ["법률 질의 텍스트"]
    query_embeddings = encoder.encode_queries(queries)
    
    # 패시지 임베딩
    passages = ["법률 문서 패시지"]
    passage_embeddings = encoder.encode_passages(passages)
"""

from .models.encoders import BiEncoder
from .models.templates import TemplateMode

__all__ = [
    "BiEncoder",
    "TemplateMode",
]

__version__ = "0.1.0"

