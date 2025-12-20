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
        """모델의 backward hook 등록"""
        # backward 완료 후 한 번만 실행되도록 하기 위한 플래그
        self._clipping_done = False
        
        def backward_hook(grad):
            """Backward 후 gradient clipping 수행"""
            self.total_count += 1
            
            # Gradient clipping 수행 (한 번만 실행)
            # 모든 파라미터의 gradient가 계산된 후에만 실행
            if self.max_norm > 0 and not self._clipping_done:
                try:
                    # 학습 가능한 파라미터만 선택 (gradient가 있고 requires_grad=True인 것만)
                    parameters = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    
                    # 모든 학습 가능한 파라미터의 gradient가 계산되었는지 확인
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    all_grads_computed = len(parameters) == len(trainable_params) and len(parameters) > 0
                    
                    if all_grads_computed and parameters:
                        # Gradient norm 계산 및 clipping
                        # clip_grad_norm_()은 requires_grad=True이고 grad가 None이 아닌 파라미터만 처리
                        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
                        self.last_norm = total_norm.item()
                        self._clipping_done = True
                        
                        # Clipping이 발생했는지 확인 (norm이 max_norm보다 큰 경우)
                        if total_norm > self.max_norm:
                            self.clipped_count += 1
                            if self.clipped_count % 100 == 1:  # 로그 스팸 방지
                                logger.debug(
                                    f"Gradient clipping 적용: norm={total_norm:.4f} > max_norm={self.max_norm:.4f}"
                                )
                except RuntimeError as e:
                    # "element 0 of tensors does not require grad" 에러 방지
                    error_msg = str(e).lower()
                    if "does not require grad" in error_msg or "grad_fn" in error_msg:
                        # 일부 파라미터의 gradient가 아직 계산되지 않았거나
                        # gradient가 detach되었을 수 있음
                        # 다음 hook 호출에서 다시 시도
                        pass
                    else:
                        raise
                except Exception as e:
                    # 다른 예외는 로깅만 하고 계속 진행
                    logger.warning(f"Gradient clipping 중 예외 발생 (무시됨): {e}")
            
            return grad  # gradient를 그대로 반환
        
        # Gradient를 요구하는 파라미터 찾기
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            logger.warning("학습 가능한 파라미터가 없어 gradient clipping hook을 등록할 수 없습니다.")
            return
        
        # 마지막 학습 가능한 파라미터에 hook 등록
        # backward pass가 완료된 후에 clipping 수행 (모든 gradient가 계산된 후)
        last_trainable_param = trainable_params[-1]
        self.hook_handle = last_trainable_param.register_hook(backward_hook)
        logger.info(f"Gradient clipping hook 등록됨 (max_norm={self.max_norm}, 학습 가능한 파라미터: {len(trainable_params)}개)")
    
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

