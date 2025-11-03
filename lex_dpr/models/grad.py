# lex_dpr/models/grad.py
import torch

def clip_gradients(model, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
