# lex_dpr/models/distill.py
import torch, torch.nn.functional as F

def mse_distill_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor):
    return F.mse_loss(student_emb, teacher_emb)
