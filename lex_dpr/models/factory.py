# lex_dpr/models/factory.py
from .encoders import BiEncoder
from .templates import TemplateMode

ALIASES = {
    "bge-m3": "BAAI/bge-m3",
    "bge-m3-ko": "dragonkue/BGE-m3-ko",   # ko 가중치 alias가 없다면 기본 m3 사용
    # 작은 모델 옵션 (테스트/메모리 제약용)
    "ko-simcse": "jhgan/ko-sroberta-multitask",  # ~110M, 한국어 전용, 추천
    "multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # ~117M, 다국어
    "multilingual-e5-small": "intfloat/multilingual-e5-small",  # ~118M, 다국어
}

def get_bi_encoder(
    name: str, 
    template: str = "bge", 
    normalize: bool = True, 
    max_len: int | None = None,
    query_max_len: int | None = None,
    passage_max_len: int | None = None,
) -> BiEncoder:
    real = ALIASES.get(name, name)
    mode = TemplateMode(template) if template in ("bge","none") else TemplateMode.BGE
    return BiEncoder(
        real, 
        template=mode, 
        normalize=normalize, 
        max_seq_length=max_len,
        query_max_seq_length=query_max_len,
        passage_max_seq_length=passage_max_len,
    )
