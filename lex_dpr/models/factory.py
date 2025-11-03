# lex_dpr/models/factory.py
from .encoders import BiEncoder
from .templates import TemplateMode

ALIASES = {
    "bge-m3": "BAAI/bge-m3",
    "bge-m3-ko": "BAAI/bge-m3"   # ko 가중치 alias가 없다면 기본 m3 사용
}

def get_bi_encoder(name: str, template: str = "bge", normalize: bool = True, max_len: int | None = None) -> BiEncoder:
    real = ALIASES.get(name, name)
    mode = TemplateMode(template) if template in ("bge","none") else TemplateMode.BGE
    return BiEncoder(real, template=mode, normalize=normalize, max_seq_length=max_len)
