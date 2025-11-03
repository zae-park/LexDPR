# lex_dpr/models/losses.py
from sentence_transformers import SentenceTransformer, losses

def build_mnr_loss(model: SentenceTransformer, temperature: float = 0.05):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return losses.MultipleNegativesRankingLoss(model, scale=temperature)
