# lex_dpr/trainer/gradient_clipping.py
"""
Gradient Clipping 기능 구현

sentence-transformers의 fit() 메서드에 gradient clipping을 적용하기 위한
Loss 함수 래퍼
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from sentence_transformers import losses

logger = logging.getLogger("lex_dpr.trainer.gradient_clipping")


class GradientClippingHook:
    """
    모델의 backward hook을 사용하여 gradient clipping을 수행하는 클래스
    
    sentence-transformers의 fit() 메서드는 내부적으로 학습 루프를 관리하므로,
    모델의 backward hook을 사용하여 backward 후에 gradient clipping을 수행합니다.
    """
    
    def __init__(self, model: torch.nn.Module, max_norm: float = 1.0):
        """
        Args:
            model: PyTorch 모델
            max_norm: Gradient clipping의 최대 노름 값
        """
        self.model = model
        self.max_norm = max_norm
        self.clipped_count = 0
        self.total_count = 0
        self.last_norm = 0.0
        self.hook_handle = None
    
    def register_hook(self):
        """모델의 backward hook 등록 (비활성화됨 - Accelerate 호환성 문제)"""
        # Accelerate를 사용하는 경우, backward hook 내에서 gradient를 직접 수정하면 안 됨
        # sentence-transformers의 fit()은 내부적으로 Accelerate를 사용하므로
        # hook 방식은 Accelerate의 backward와 충돌함
        
        # Hook을 등록하지 않음 (비활성화)
        self.hook_handle = None
        logger.warning("⚠️  Gradient clipping hook이 Accelerate와 호환되지 않아 비활성화됩니다.")
        logger.warning("   sentence-transformers의 fit()은 내부적으로 Accelerate를 사용하므로,")
        logger.warning("   backward hook에서 gradient clipping을 수행할 수 없습니다.")
        logger.warning("   gradient_clip_norm 설정은 현재 무시됩니다.")
        logger.warning("   대안: optimizer에 max_grad_norm을 설정하거나,")
        logger.warning("   sentence-transformers의 Trainer를 확장하여 gradient clipping을 구현하세요.")
    
    def remove_hook(self):
        """Hook 제거"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            self._clipping_done = False  # 플래그 리셋
            logger.info("Gradient clipping hook 제거됨")
    
    def get_stats(self) -> dict:
        """Gradient clipping 통계 반환"""
        return {
            "total_backwards": self.total_count,
            "clipped_backwards": self.clipped_count,
            "clipping_ratio": self.clipped_count / max(1, self.total_count),
            "last_norm": self.last_norm,
        }


def apply_gradient_clipping_to_model(
    model: torch.nn.Module,
    max_norm: float = 1.0,
) -> Optional[GradientClippingHook]:
    """
    모델에 gradient clipping hook 적용
    
    Args:
        model: PyTorch 모델
        max_norm: Gradient clipping의 최대 노름 값 (0 이하면 비활성화)
    
    Returns:
        GradientClippingHook 인스턴스 (max_norm > 0인 경우) 또는 None (그 외)
    """
    if max_norm <= 0:
        logger.info("Gradient clipping 비활성화됨 (max_norm <= 0)")
        return None
    
    hook = GradientClippingHook(model, max_norm=max_norm)
    hook.register_hook()
    
    return hook

