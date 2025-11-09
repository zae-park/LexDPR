# lex_dpr/models/losses.py
from sentence_transformers import SentenceTransformer, losses

def build_mnr_loss(model: SentenceTransformer, temperature: float = 0.05):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return losses.MultipleNegativesRankingLoss(model, scale=temperature)


def clip_gradients(model, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)