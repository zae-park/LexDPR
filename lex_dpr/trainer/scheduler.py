# lex_dpr/models/schedulers.py
from transformers import get_cosine_schedule_with_warmup

def build_warmup_cosine(optimizer, total_steps: int, warmup_ratio: float = 0.1, min_warmup: int = 10):
    warmup = max(min_warmup, int(total_steps * warmup_ratio))
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )
