# lex_dpr/trainer/early_stopping.py
"""
Early Stopping 기능 구현

Validation 메트릭을 모니터링하여 학습을 조기 종료하고,
최고 성능 모델을 자동으로 저장합니다.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger("lex_dpr.trainer.early_stopping")


class EarlyStoppingCallback:
    """
    Early Stopping 콜백
    
    Validation 메트릭을 모니터링하여:
    - 지정된 patience 동안 개선이 없으면 학습 종료
    - 최고 성능 모델을 자동으로 저장
    """
    
    def __init__(
        self,
        model,
        out_dir: str,
        metric_key: str = "cosine_ndcg@10",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max",
        restore_best_weights: bool = True,
    ):
        """
        Args:
            model: SentenceTransformer 모델
            out_dir: 체크포인트 저장 디렉토리
            metric_key: 모니터링할 메트릭 키 (예: "cosine_ndcg@10", "cosine_mrr@10")
            patience: 개선이 없을 때 기다릴 평가 횟수
            min_delta: 개선으로 간주할 최소 변화량
            mode: "max" (값이 클수록 좋음) 또는 "min" (값이 작을수록 좋음)
            restore_best_weights: 조기 종료 시 최고 성능 가중치로 복원할지 여부
        """
        self.model = model
        self.out_dir = out_dir
        self.metric_key = metric_key
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_step = -1
        self.patience_counter = 0
        self.best_model_path = os.path.join(out_dir, "bi_encoder_best")
        self.should_stop = False
        
        logger.info(f"Early Stopping 초기화:")
        logger.info(f"  - 모니터링 메트릭: {metric_key}")
        logger.info(f"  - Patience: {patience}")
        logger.info(f"  - Mode: {mode}")
        logger.info(f"  - 최소 변화량: {min_delta}")
    
    def __call__(self, metrics: Dict[str, float], step: int, epoch: int) -> bool:
        """
        평가 결과를 받아 early stopping 여부를 결정
        
        Args:
            metrics: 평가 메트릭 딕셔너리
            step: 현재 스텝
            epoch: 현재 에포크
        
        Returns:
            True면 학습을 중단해야 함, False면 계속 진행
        """
        if not metrics:
            return False
        
        # 메트릭 키 찾기 (정확한 키 또는 유사한 키)
        metric_value = None
        for key in [self.metric_key, f"val_{self.metric_key}", f"cosine_{self.metric_key}"]:
            if key in metrics:
                metric_value = float(metrics[key])
                break
        
        if metric_value is None:
            logger.warning(f"Early Stopping: 메트릭 '{self.metric_key}'를 찾을 수 없습니다. 사용 가능한 키: {list(metrics.keys())}")
            return False
        
        # 개선 여부 확인
        is_better = False
        if self.mode == "max":
            if metric_value > self.best_score + self.min_delta:
                is_better = True
        else:  # mode == "min"
            if metric_value < self.best_score - self.min_delta:
                is_better = True
        
        if is_better:
            # 개선됨: 최고 성능 업데이트 및 모델 저장
            self.best_score = metric_value
            self.best_step = step
            self.patience_counter = 0
            
            # 최고 성능 모델 저장
            os.makedirs(self.out_dir, exist_ok=True)
            self.model.save(self.best_model_path)
            logger.info(
                f"✅ 최고 성능 모델 저장 (step={step}, {self.metric_key}={metric_value:.4f})"
            )
        else:
            # 개선 없음: patience 카운터 증가
            self.patience_counter += 1
            logger.info(
                f"Early Stopping: 개선 없음 ({self.patience_counter}/{self.patience}) "
                f"(현재: {metric_value:.4f}, 최고: {self.best_score:.4f})"
            )
        
        # Early stopping 조건 확인
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.warning(
                f"🛑 Early Stopping: {self.patience}번의 평가 동안 개선이 없어 학습을 종료합니다. "
                f"(최고 성능: {self.best_score:.4f} at step {self.best_step})"
            )
            
            # 최고 성능 가중치로 복원
            if self.restore_best_weights:
                logger.info(f"최고 성능 모델 로드 중: {self.best_model_path}")
                try:
                    from sentence_transformers import SentenceTransformer
                    best_model = SentenceTransformer(self.best_model_path)
                    # 모델 가중치 복사
                    self.model.load_state_dict(best_model.state_dict())
                    logger.info("✅ 최고 성능 가중치로 복원 완료")
                except Exception as e:
                    logger.warning(f"최고 성능 모델 복원 실패: {e}")
            
            return True
        
        return False
    
    def get_best_score(self) -> float:
        """최고 성능 점수 반환"""
        return self.best_score
    
    def get_best_step(self) -> int:
        """최고 성능이 나온 스텝 반환"""
        return self.best_step


class EarlyStoppingEvaluatorWrapper:
    """
    Evaluator를 래핑하여 Early Stopping 기능 추가
    
    sentence-transformers의 SequentialEvaluator와 호환되도록
    iterable 인터페이스를 제공합니다.
    """
    
    def __init__(self, evaluator, early_stopping: EarlyStoppingCallback):
        self.evaluator = evaluator
        self.early_stopping = early_stopping
        self.current_step = 0
        self.current_epoch = 0
    
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        """
        Evaluator 실행 및 Early Stopping 체크
        """
        # 원본 evaluator 실행
        result = self.evaluator(model, output_path, epoch, steps)
        
        # Step 업데이트
        self.current_step = steps if steps >= 0 else self.current_step
        self.current_epoch = epoch if epoch >= 0 else self.current_epoch
        
        # Early stopping 체크
        if result and isinstance(result, dict) and self.early_stopping:
            should_stop = self.early_stopping(result, self.current_step, self.current_epoch)
            if should_stop:
                # Early stopping 발생: 예외를 발생시켜 학습 중단
                raise EarlyStoppingException(
                    f"Early stopping triggered at step {self.current_step} "
                    f"(best {self.early_stopping.metric_key}={self.early_stopping.get_best_score():.4f} at step {self.early_stopping.get_best_step()})"
                )
        
        return result
    


class EarlyStoppingException(Exception):
    """Early Stopping이 발생했을 때 발생하는 예외"""
    pass

