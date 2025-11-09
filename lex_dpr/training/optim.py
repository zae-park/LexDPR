# lex_dpr/models/optim.py
from torch.optim import AdamW

def build_optimizer(params, lr: float, weight_decay: float = 0.01):
    no_decay = ["bias","LayerNorm.weight"]
    grouped = [
        {"params":[p for n,p in params if not any(nd in n for nd in no_decay)], "weight_decay":weight_decay},
        {"params":[p for n,p in params if any(nd in n for nd in no_decay)], "weight_decay":0.0},
    ]
    return AdamW(grouped, lr=lr)