# lex_dpr/models/losses.py
from sentence_transformers import SentenceTransformer, losses
import torch
import torch.nn.functional as F
from typing import List, Dict

def build_mnr_loss(model: SentenceTransformer, temperature: float = 0.05):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return losses.MultipleNegativesRankingLoss(model, scale=temperature)


class MixedNegativesRankingLoss(torch.nn.Module):
    """
    In-batch negatives와 hard negatives를 섞어서 사용하는 loss 함수
    
    sentence-transformers의 MultipleNegativesRankingLoss를 확장하여
    hard negative를 포함시킨 경우를 처리합니다.
    
    Note: 현재는 데이터셋에서 hard negative를 포함시키고, 기본 loss를 사용합니다.
    실제 비율 조절은 데이터셋 레벨에서 처리됩니다.
    
    Args:
        model: SentenceTransformer 모델
        temperature: Temperature scaling
        hard_negative_ratio: Hard negative 비율 (0.0 = in-batch만, 1.0 = hard negative만)
                            현재는 데이터셋에서 이미 샘플링된 hard negative를 사용합니다.
    """
    
    def __init__(self, model: SentenceTransformer, temperature: float = 0.05, hard_negative_ratio: float = 0.0):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        # 기본 loss 사용 (데이터셋에서 hard negative가 이미 포함됨)
        self.base_loss = losses.MultipleNegativesRankingLoss(model, scale=temperature)
    
    def forward(self, sentence_features: List[Dict[str, torch.Tensor]], labels: torch.Tensor):
        """
        sentence_features: 각 샘플의 sentence_embedding을 포함한 dict 리스트
        - 기본 모드: [{"sentence_embedding": query_emb}, {"sentence_embedding": pos_emb}]
        - Hard negative 모드: [{"sentence_embedding": query_emb}, {"sentence_embedding": pos_emb}, 
                                {"sentence_embedding": neg1_emb}, ...]
        
        Note: MultipleNegativesRankingLoss는 기본적으로 (query, positive) 쌍만 처리하므로,
        hard negative가 포함된 경우는 기본 loss가 자동으로 무시합니다.
        실제 비율 조절은 데이터셋에서 hard negative를 포함시킬 때 처리됩니다.
        """
        # 기본 loss 사용 (hard negative는 데이터셋에서 이미 포함됨)
        return self.base_loss(sentence_features, labels)


def build_mixed_negatives_loss(
    model: SentenceTransformer, 
    temperature: float = 0.05,
    hard_negative_ratio: float = 0.0
):
    """
    In-batch negatives와 hard negatives를 섞어서 사용하는 loss 생성
    
    Args:
        model: SentenceTransformer 모델
        temperature: Temperature scaling
        hard_negative_ratio: Hard negative 비율 (0.0 = in-batch만, 1.0 = hard negative만)
    
    Returns:
        MixedNegativesRankingLoss 인스턴스 또는 MultipleNegativesRankingLoss (hard_negative_ratio=0인 경우)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    
    if hard_negative_ratio <= 0:
        # Hard negative를 사용하지 않으면 기본 loss 사용
        return losses.MultipleNegativesRankingLoss(model, scale=temperature)
    else:
        return MixedNegativesRankingLoss(model, temperature=temperature, hard_negative_ratio=hard_negative_ratio)


def clip_gradients(model, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)