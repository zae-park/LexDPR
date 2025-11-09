# lex_dpr/models/tokenizer.py
from transformers import AutoTokenizer

def get_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    max_len = getattr(tok, "model_max_length", 512)
    # HF에 1e30 같은 placeholder가 올라올 때 대비
    if max_len is None or max_len > 4096:
        max_len = 512
    return tok, int(max_len)
