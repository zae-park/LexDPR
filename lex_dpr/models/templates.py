# lex_dpr/models/templates.py
from enum import Enum

class TemplateMode(str, Enum):
    NONE = "none"
    BGE = "bge"

BGE_Q = "Represent this sentence for searching relevant passages: {q}"
BGE_P = "Represent this sentence for retrieving relevant passages: {p}"

def tq(q: str, mode: TemplateMode) -> str:
    return BGE_Q.format(q=q) if mode == TemplateMode.BGE else q

def tp(p: str, mode: TemplateMode) -> str:
    return BGE_P.format(p=p) if mode == TemplateMode.BGE else p
