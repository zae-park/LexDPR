# lex_dpr/trainer/base_trainer.py
from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from omegaconf import DictConfig
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

from ..data import load_passages
from ..eval import build_ir_evaluator, eval_recall_at_k
from ..models.factory import get_bi_encoder
from ..models.peft import attach_lora_to_st, enable_lora_only_train
from ..models.templates import TemplateMode, tq, tp
from ..utils.io import read_jsonl
from ..utils.seed import set_seed
from ..utils.web_logging import create_web_logger, WebLogger
from .early_stopping import (
    EarlyStoppingCallback,
    EarlyStoppingEvaluatorWrapper,
    EarlyStoppingException,
)
from .gradient_clipping import apply_gradient_clipping_to_model

logger = logging.getLogger("lex_dpr.trainer")


def _resolve_template_mode(cfg_model) -> TemplateMode:
    use_bge = bool(getattr(cfg_model, "use_bge_template", True))
    return TemplateMode.BGE if use_bge else TemplateMode.NONE


def _apply_multiply(examples: List[InputExample], multiply: int) -> List[InputExample]:
    if multiply <= 1:
        return examples
    # 단순 반복 대신 셔플을 적용하여 같은 예제가 연속으로 나타나지 않도록 함
    # 이렇게 하면 같은 데이터가 반복되어도 배치 내에서 다양성이 유지됨
    multiplied = examples * multiply
    random.shuffle(multiplied)
    return multiplied


@dataclass
class TrainerArtifacts:
    loader: DataLoader
    loss: losses.MultipleNegativesRankingLoss
    evaluator: Optional[object]
    steps_per_epoch: int
    warmup_steps: int


class WebLoggingEvaluatorWrapper:
    """
    sentence-transformers evaluator를 래핑하여 웹 로깅에 결과 전송
    
    sentence-transformers의 SequentialEvaluator와 호환되도록
    iterable 인터페이스를 제공합니다.
    """
    
    def __init__(self, evaluator, web_logger: WebLogger):
        self.evaluator = evaluator
        self.web_logger = web_logger
    
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        """evaluator 호출 및 결과를 웹 로깅으로 전송"""
        # 원본 evaluator 실행
        result = self.evaluator(model, output_path, epoch, steps)
        
        # 결과를 웹 로깅으로 전송
        if result and isinstance(result, dict):
            # sentence-transformers evaluator는 dict 형태로 메트릭을 반환
            # 예: "val_cosine_ndcg@10" -> "eval/ndcg@10"
            metrics = {}
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    # 메트릭 이름 정규화
                    # "val_cosine_ndcg@10" -> "eval/ndcg_at_10"
                    # "val_cosine_recall@5" -> "eval/recall_at_5"
                    metric_name = key
                    # "val_" 제거
                    if metric_name.startswith("val_"):
                        metric_name = metric_name[4:]
                    # "cosine_" 제거 (거리 메트릭은 보통 cosine이므로 생략)
                    if metric_name.startswith("cosine_"):
                        metric_name = metric_name[7:]
                    # "@" 기호를 "_at_"로 변경 (WandB는 @ 기호를 허용하지 않음)
                    metric_name = metric_name.replace("@", "_at_")
                    # "eval/" prefix 추가
                    metric_name = f"eval/{metric_name}"
                    metrics[metric_name] = float(value)
            
            if metrics:
                step = steps if steps >= 0 else epoch
                self.web_logger.log_metrics(metrics, step=step)
                logger.info(f"평가 메트릭을 웹 로깅 서비스에 전송했습니다: {len(metrics)}개 메트릭 (step={step})")
        
        return result
    
    def __iter__(self):
        """
        SequentialEvaluator와의 호환성을 위해 iterable 인터페이스 제공
        자신을 단일 항목으로 반환
        """
        return iter([self])


class WebLoggingCallback:
    """sentence-transformers fit() 메서드의 학습 중 loss를 WandB에 로깅하는 콜백"""
    
    def __init__(self, web_logger: WebLogger):
        self.web_logger = web_logger
        self.current_step = 0
        self.current_epoch = 0
    
    def __call__(self, score, epoch, steps):
        """학습 중 호출되는 콜백 (loss 값 로깅)"""
        if not self.web_logger or not self.web_logger.is_active:
            return
        
        # score는 일반적으로 loss 값
        # sentence-transformers의 fit() 메서드에서 제공하는 정보
        if isinstance(score, (int, float)):
            self.current_step = steps if steps >= 0 else self.current_step
            self.current_epoch = epoch if epoch >= 0 else self.current_epoch
            
            # loss를 WandB에 로깅
            metrics = {
                "train/loss": float(score),
            }
            self.web_logger.log_metrics(metrics, step=self.current_step)


class BiEncoderTrainer:
    """
    학습 스크립트와 분리된 BI-Encoder Trainer.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        logger.info(f"시드 설정: {cfg.seed}")
        
        # 웹 로깅 초기화
        self.web_logger = create_web_logger(cfg)
        
        self.template_mode = _resolve_template_mode(cfg.model)
        logger.info(f"템플릿 모드: {self.template_mode.value}")
        
        logger.info("데이터 로딩 중...")
        logger.info(f"  - Passages: {cfg.data.passages}")
        self.passages = load_passages(cfg.data.passages)
        logger.info(f"  - 로드된 Passages 수: {len(self.passages):,}")
        
        logger.info(f"  - Training Pairs: {cfg.data.pairs}")
        self.pairs = list(read_jsonl(cfg.data.pairs))
        logger.info(f"  - 로드된 Pairs 수: {len(self.pairs):,}")

        logger.info("인코더 빌드 중...")
        self.encoder = self._build_encoder()
        self.model = self.encoder.model

        logger.info("학습 예제 생성 중...")
        self.examples = self._build_examples()
        logger.info(f"  - 생성된 예제 수: {len(self.examples):,}")
        
        self.batch_size = self._resolve_batch_size(len(self.examples))
        logger.info(f"  - 배치 크기: {self.batch_size}")
        
        logger.info("학습 아티팩트 준비 중...")
        self.artifacts = self._build_artifacts()
        logger.info(f"  - 에포크당 스텝 수: {self.artifacts.steps_per_epoch:,}")
        logger.info(f"  - 총 스텝 수: {self.artifacts.steps_per_epoch * cfg.trainer.epochs:,}")
        logger.info(f"  - Warmup 스텝 수: {self.artifacts.warmup_steps:,}")
        if self.artifacts.evaluator:
            logger.info(f"  - 평가기: 활성화 (평가 스텝: {cfg.trainer.eval_steps})")
        else:
            logger.info(f"  - 평가기: 비활성화")
        
        # 웹 로깅에 하이퍼파라미터 로깅
        if self.web_logger and self.web_logger.is_active:
            self._log_hyperparameters()

    # ------------------------------
    # Build helpers
    # ------------------------------
    def _build_encoder(self):
        max_len = int(getattr(self.cfg.model, "max_len", 0) or 0)
        query_max_len = int(getattr(self.cfg.model, "query_max_len", 0) or 0)
        passage_max_len = int(getattr(self.cfg.model, "passage_max_len", 0) or 0)
        
        encoder = get_bi_encoder(
            self.cfg.model.bi_model,
            template=self.template_mode.value,
            max_len=max_len if max_len > 0 else None,
            query_max_len=query_max_len if query_max_len > 0 else None,
            passage_max_len=passage_max_len if passage_max_len > 0 else None,
        )
        
        # 로깅
        if query_max_len > 0 or passage_max_len > 0:
            logger.info(f"시퀀스 길이 설정: Query={encoder.query_max_seq_length}, Passage={encoder.passage_max_seq_length}")
        elif max_len > 0:
            logger.info(f"모델 최대 시퀀스 길이 설정: {encoder.model.max_seq_length}")
        
        # PEFT (LoRA) 지원
        peft_config = getattr(self.cfg.model, "peft", None)
        if peft_config and getattr(peft_config, "enabled", False):
            r = int(getattr(peft_config, "r", 16))
            alpha = int(getattr(peft_config, "alpha", 32))
            dropout = float(getattr(peft_config, "dropout", 0.05))
            target_modules = getattr(peft_config, "target_modules", None)
            if target_modules is None:
                # None으로 설정하면 attach_lora_to_st에서 자동 감지
                target_modules = None
            elif isinstance(target_modules, str):
                # 문자열로 된 경우 리스트로 변환
                target_modules = [m.strip() for m in target_modules.split(",")]
            
            if target_modules:
                logger.info(f"LoRA 어댑터 연결 중: r={r}, alpha={alpha}, dropout={dropout}, target_modules={target_modules}")
            else:
                logger.info(f"LoRA 어댑터 연결 중: r={r}, alpha={alpha}, dropout={dropout}, target_modules=auto-detect")
            encoder.model = attach_lora_to_st(
                encoder.model,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            )
            # PEFT 모델은 자동으로 base_model을 동결하고 LoRA만 학습 가능하게 설정함
            # enable_lora_only_train은 디버깅 및 확인용
            try:
                enable_lora_only_train(encoder.model)
                logger.info("LoRA 어댑터 연결 완료. LoRA 파라미터만 학습됩니다.")
            except Exception as e:
                logger.warning(f"enable_lora_only_train 실패: {e}")
                logger.info("PEFT 기본 설정으로 계속 진행합니다...")
                # PEFT가 자동으로 처리하도록 함
                encoder.model.train()
        
        # Gradient checkpointing 활성화 (메모리 절약)
        if getattr(self.cfg.trainer, "gradient_checkpointing", False):
            from sentence_transformers import models as st_models
            # SentenceTransformer의 첫 번째 Transformer 모듈 찾기
            for module in encoder.model.modules():
                if isinstance(module, st_models.Transformer):
                    if hasattr(module, "auto_model"):
                        base_model = module.auto_model
                        # PEFT 모델인 경우 base_model에서 찾기
                        if hasattr(base_model, "base_model"):
                            base_model = base_model.base_model.model
                        
                        if hasattr(base_model, "gradient_checkpointing_enable"):
                            base_model.gradient_checkpointing_enable()
                            logger.info("Gradient checkpointing 활성화됨.")
                        elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "gradient_checkpointing_enable"):
                            base_model.encoder.gradient_checkpointing_enable()
                            logger.info("Gradient checkpointing 활성화됨 (encoder).")
                        break
        
        return encoder

    def _build_examples(self) -> List[InputExample]:
        examples: List[InputExample] = []
        miss_pos = 0
        
        # Hard negative 사용 여부 및 비율 확인
        use_hard_negatives = bool(getattr(self.cfg.data, "use_hard_negatives", False))
        hard_negative_ratio = float(getattr(self.cfg.data, "hard_negative_ratio", 0.0))
        
        if use_hard_negatives and hard_negative_ratio > 0:
            logger.info(f"Hard negative 사용: 비율={hard_negative_ratio:.2f}")
        
        for row in self.pairs:
            q_text = tq(row["query_text"], self.template_mode)
            for pid in row["positive_passages"]:
                passage = self.passages.get(pid)
                if not passage:
                    miss_pos += 1
                    continue
                p_text = tp(passage["text"], self.template_mode)
                
                # Hard negative 포함 여부 결정
                if use_hard_negatives and hard_negative_ratio > 0:
                    # Hard negative가 있는 경우에만 포함
                    hard_neg_ids = [nid for nid in row.get("hard_negatives", []) if nid in self.passages]
                    if hard_neg_ids:
                        import random
                        # 비율에 따라 샘플링할 hard negative 개수 결정
                        # 예상 배치 크기를 고려하여 hard negative 개수 결정
                        # hard_negative_ratio에 따라 샘플링
                        # 예: 배치 크기 128, hard_negative_ratio=0.3이면
                        #     in-batch negative = 127개, hard negative = 약 54개 (127 * 0.3 / (1-0.3))
                        #     하지만 실제로는 데이터셋에서 샘플링하므로, 평균적으로 비율에 맞게 샘플링
                        
                        # Hard negative를 비율에 따라 샘플링
                        # 실제 비율은 배치 내에서 결정되므로, 여기서는 가능한 한 포함
                        # 나중에 loss 함수에서 비율 조절
                        neg_texts = []
                        for nid in hard_neg_ids:
                            neg_passage = self.passages.get(nid)
                            if neg_passage:
                                neg_texts.append(tp(neg_passage["text"], self.template_mode))
                        
                        if neg_texts:
                            # [query, positive, ...hard_negatives] 형태로 생성
                            # MultipleNegativesRankingLoss는 기본적으로 (query, positive)만 처리하므로,
                            # hard negative는 무시되지만, 나중에 custom loss에서 사용 가능
                            examples.append(InputExample(texts=[q_text, p_text] + neg_texts))
                        else:
                            examples.append(InputExample(texts=[q_text, p_text]))
                    else:
                        examples.append(InputExample(texts=[q_text, p_text]))
                else:
                    # Hard negative 사용 안 함
                    examples.append(InputExample(texts=[q_text, p_text]))

        if miss_pos:
            logger.warning(f"corpus에 없는 positive passage ID {miss_pos}개 건너뜀")

        multiply = int(getattr(self.cfg.data, "multiply", 0) or 0)
        if multiply > 1:
            original_count = len(examples)
            examples = _apply_multiply(examples, multiply)
            logger.info(f"예제 증폭: {original_count:,} -> {len(examples):,} (x{multiply})")

        if not examples:
            raise ValueError(
                "학습 예제가 없습니다. pairs/positive ids가 corpus와 일치하는지 확인하세요."
            )
        return examples

    def _resolve_batch_size(self, n_examples: int) -> int:
        batch_size = int(self.cfg.data.batches.bi)
        if n_examples < batch_size:
            logger.warning(f"배치 크기 조정: {batch_size} -> {n_examples} (예제 수 부족)")
            batch_size = n_examples
        self.cfg.data.batches.bi = batch_size
        return batch_size

    def _build_artifacts(self) -> TrainerArtifacts:
        # 테스트 실행 모드: 최대 100 iteration으로 제한
        test_run = getattr(self.cfg, "test_run", False)
        max_steps = 100 if test_run else None
        
        if test_run and max_steps:
            # 테스트 실행 모드: 제한된 예제만 사용
            max_examples = max_steps * self.batch_size
            limited_examples = self.examples[:max_examples]
            logger.info(f"🧪 테스트 실행 모드: {len(limited_examples):,}개 예제만 사용 (최대 {max_steps} iteration)")
            examples_to_use = limited_examples
        else:
            examples_to_use = self.examples
        
        loader = DataLoader(
            examples_to_use,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        
        # Hard negative 사용 여부 및 비율 확인
        use_hard_negatives = bool(getattr(self.cfg.data, "use_hard_negatives", False))
        hard_negative_ratio = float(getattr(self.cfg.data, "hard_negative_ratio", 0.0))
        
        # Loss 함수 선택
        if use_hard_negatives and hard_negative_ratio > 0:
            from .losses import build_mixed_negatives_loss
            loss = build_mixed_negatives_loss(
                self.model,
                temperature=self.cfg.trainer.temperature,
                hard_negative_ratio=hard_negative_ratio
            )
            logger.info(f"MixedNegativesRankingLoss 사용: hard_negative_ratio={hard_negative_ratio:.2f}")
        else:
            loss = losses.MultipleNegativesRankingLoss(self.model, scale=self.cfg.trainer.temperature)
        
        # Gradient clipping 적용
        gradient_clip_norm = float(getattr(self.cfg.trainer, "gradient_clip_norm", 0.0))
        self.gradient_clipping_hook = None
        if gradient_clip_norm > 0:
            self.gradient_clipping_hook = apply_gradient_clipping_to_model(
                self.model,
                max_norm=gradient_clip_norm,
            )

        evaluator = None
        early_stopping = None
        
        if self.cfg.trainer.eval_pairs and os.path.exists(self.cfg.trainer.eval_pairs):
            # IR evaluator 생성 (평가 배치 크기 설정: 메모리 절약)
            # InformationRetrievalEvaluator는 쿼리를 하나씩 처리하므로,
            # batch_size는 corpus encoding에만 사용됨 (더 크게 설정 가능)
            eval_batch_size = min(64, max(32, self.batch_size))  # 평가는 적당한 배치로 (corpus encoding용)
            base_evaluator, _ = build_ir_evaluator(
                passages=self.passages,
                eval_pairs_path=self.cfg.trainer.eval_pairs,
                read_jsonl_fn=read_jsonl,
                k_vals=self.cfg.trainer.k_values,
                template=self.template_mode,
                batch_size=eval_batch_size,
            )
            
            # Validation loss evaluator 추가
            from ..eval import ValidationLossEvaluator
            # Validation loss 계산 시 전체 corpus에서 negative 샘플링 (실전 모방)
            use_full_corpus_negatives = bool(getattr(self.cfg.trainer, "use_full_corpus_negatives", True))
            num_negatives_per_query = int(getattr(self.cfg.trainer, "num_negatives_per_query", 1000))
            
            val_loss_evaluator = ValidationLossEvaluator(
                model=self.model,
                passages=self.passages,
                eval_pairs_path=self.cfg.trainer.eval_pairs,
                read_jsonl_fn=read_jsonl,
                temperature=self.cfg.trainer.temperature,
                template=self.template_mode,
                batch_size=min(32, self.batch_size),
                use_full_corpus_negatives=use_full_corpus_negatives,
                num_negatives_per_query=num_negatives_per_query,
            )
            
            # 두 evaluator를 결합
            from sentence_transformers.evaluation import SequentialEvaluator
            base_evaluator = SequentialEvaluator([val_loss_evaluator, base_evaluator])
            
            # Progress bar 억제를 위한 래퍼 추가
            class SuppressProgressBarEvaluator:
                """Progress bar를 억제하는 evaluator 래퍼"""
                def __init__(self, evaluator):
                    self.evaluator = evaluator
                
                def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
                    import os
                    import sys
                    from io import StringIO
                    from tqdm import tqdm
                    
                    # tqdm 출력 억제
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = StringIO()
                    sys.stderr = StringIO()
                    
                    # tqdm 비활성화
                    old_disable = getattr(tqdm, '_instances', None)
                    tqdm._instances = set()  # tqdm 인스턴스 추적 비활성화
                    
                    try:
                        result = self.evaluator(model, output_path, epoch, steps)
                    finally:
                        # 복원
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        if old_disable is not None:
                            tqdm._instances = old_disable
                    
                    return result
                
                def __iter__(self):
                    """SequentialEvaluator 호환성"""
                    if hasattr(self.evaluator, '__iter__'):
                        return iter(self.evaluator)
                    return iter([self])
            
            # Progress bar 억제 래퍼 적용
            suppressed_evaluator = SuppressProgressBarEvaluator(base_evaluator)
            
            # 래퍼 체인: Web Logging -> Suppressed Evaluator
            # Early Stopping은 warmup_steps 계산 후에 추가됨
            current_evaluator = suppressed_evaluator
            
            # 웹 로깅 래퍼 추가
            if self.web_logger and self.web_logger.is_active:
                evaluator = WebLoggingEvaluatorWrapper(current_evaluator, self.web_logger)
            else:
                evaluator = current_evaluator
        elif self.cfg.trainer.eval_pairs:
            logger.warning(f"eval_pairs 파일을 찾을 수 없습니다: {self.cfg.trainer.eval_pairs}. 평가를 건너뜁니다.")

        steps_per_epoch = max(1, math.ceil(len(examples_to_use) / self.batch_size))
        
        # 테스트 실행 모드: epochs를 1로 강제
        effective_epochs = 1 if test_run else self.cfg.trainer.epochs
        if test_run and self.cfg.trainer.epochs > 1:
            logger.info(f"🧪 테스트 실행 모드: epochs를 1로 제한 (원래 설정: {self.cfg.trainer.epochs})")
        
        total_steps = steps_per_epoch * effective_epochs
        # Warmup ratio 설정 (기본값: 0.05 = 5%)
        # Warmup ratio를 낮춰서 learning rate가 너무 빨리 상승하는 것을 방지
        # Cosine annealing에 더 빨리 접어들도록 하여 학습 안정성 향상
        warmup_ratio = float(getattr(self.cfg.trainer, "warmup_ratio", 0.05))
        warmup_steps = max(10, int(total_steps * warmup_ratio))
        
        # Early Stopping 설정 (warmup_steps 계산 후)
        early_stopping_config = getattr(self.cfg.trainer, "early_stopping", None)
        if early_stopping_config and getattr(early_stopping_config, "enabled", False):
            metric_key = getattr(early_stopping_config, "metric", "cosine_ndcg@10")
            patience = int(getattr(early_stopping_config, "patience", 3))
            min_delta = float(getattr(early_stopping_config, "min_delta", 0.0))
            mode = getattr(early_stopping_config, "mode", "max")
            restore_best = getattr(early_stopping_config, "restore_best_weights", True)
            
            # Warmup 스텝 수를 early stopping에 전달하여 warmup 기간 동안 더 관대하게 처리
            early_stopping = EarlyStoppingCallback(
                model=self.model,
                out_dir=self.cfg.out_dir,
                metric_key=metric_key,
                patience=patience,
                min_delta=min_delta,
                mode=mode,
                restore_best_weights=restore_best,
                warmup_steps=warmup_steps,
            )
            logger.info(f"Early Stopping 활성화됨 (warmup_steps={warmup_steps})")
            
            # Early Stopping 래퍼를 evaluator에 추가
            # evaluator가 이미 WebLoggingEvaluatorWrapper로 래핑되어 있을 수 있으므로
            # 내부 evaluator를 찾아서 Early Stopping 래퍼를 추가
            if evaluator:
                # WebLoggingEvaluatorWrapper인 경우 내부 evaluator에 래핑
                if isinstance(evaluator, WebLoggingEvaluatorWrapper):
                    inner_evaluator = evaluator.evaluator
                    wrapped_evaluator = EarlyStoppingEvaluatorWrapper(inner_evaluator, early_stopping)
                    evaluator.evaluator = wrapped_evaluator
                else:
                    evaluator = EarlyStoppingEvaluatorWrapper(evaluator, early_stopping)

        return TrainerArtifacts(
            loader=loader,
            loss=loss,
            evaluator=evaluator,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
        )
    
    def _get_early_stopping(self) -> Optional[EarlyStoppingCallback]:
        """Early Stopping 콜백 반환 (내부용)"""
        if self.artifacts.evaluator:
            if isinstance(self.artifacts.evaluator, EarlyStoppingEvaluatorWrapper):
                return self.artifacts.evaluator.early_stopping
            elif isinstance(self.artifacts.evaluator, WebLoggingEvaluatorWrapper):
                if hasattr(self.artifacts.evaluator.evaluator, "early_stopping"):
                    return self.artifacts.evaluator.evaluator.early_stopping
        return None

    # ------------------------------
    # Web Logging Helpers
    # ------------------------------
    def _log_hyperparameters(self) -> None:
        """하이퍼파라미터를 웹 로깅 서비스에 전송"""
        if not self.web_logger or not self.web_logger.is_active:
            return
        
        params = {
            "mode": self.cfg.mode,
            "seed": self.cfg.seed,
            "trainer.epochs": self.cfg.trainer.epochs,
            "trainer.lr": self.cfg.trainer.lr,
            "trainer.batch_size": self.batch_size,
            "trainer.gradient_accumulation_steps": getattr(self.cfg.trainer, "gradient_accumulation_steps", 1),
            "trainer.use_amp": self.cfg.trainer.use_amp,
            "trainer.temperature": self.cfg.trainer.temperature,
            "model.bi_model": self.cfg.model.bi_model,
            "model.use_bge_template": self.cfg.model.use_bge_template,
            "model.max_len": getattr(self.cfg.model, "max_len", None),
            "data.passages_count": len(self.passages),
            "data.pairs_count": len(self.pairs),
            "data.examples_count": len(self.examples),
        }
        
        # PEFT 설정 추가
        if hasattr(self.cfg.model, "peft") and self.cfg.model.peft.enabled:
            params["model.peft.enabled"] = True
            params["model.peft.r"] = self.cfg.model.peft.r
            params["model.peft.alpha"] = self.cfg.model.peft.alpha
            params["model.peft.dropout"] = self.cfg.model.peft.dropout
        else:
            params["model.peft.enabled"] = False
        
        self.web_logger.log_params(params)
    
    def _log_evaluation_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """평가 메트릭을 웹 로깅 서비스에 전송"""
        if not self.web_logger or not self.web_logger.is_active:
            return
        
        self.web_logger.log_metrics(metrics, step=step)
    
    # ------------------------------
    # Public API
    # ------------------------------
    def train(self) -> None:
        # 테스트 실행 모드 확인
        test_run = getattr(self.cfg, "test_run", False)
        effective_epochs = 1 if test_run else self.cfg.trainer.epochs
        
        # sentence-transformers의 fit() 메서드는 gradient_accumulation_steps를 지원하지 않음
        # 대신 배치 사이즈를 조정하여 효과적인 배치 크기를 조절
        gradient_accumulation_steps = int(getattr(self.cfg.trainer, "gradient_accumulation_steps", 1))
        if gradient_accumulation_steps > 1:
            effective_batch_size = self.batch_size * gradient_accumulation_steps
            logger.info(f"참고: gradient_accumulation_steps={gradient_accumulation_steps}는 sentence-transformers에서 지원되지 않습니다.")
            logger.info(f"효과적인 배치 크기: {self.batch_size} × {gradient_accumulation_steps} = {effective_batch_size}")
        
        if test_run:
            logger.info(f"🧪 테스트 실행 모드: 학습 시작 (에포크: {effective_epochs}, 최대 {self.artifacts.steps_per_epoch} iteration, 학습률: {self.cfg.trainer.lr})")
            logger.info(f"  - Warmup 스텝: {self.artifacts.warmup_steps} (전체 step의 {self.artifacts.warmup_steps/max(1, self.artifacts.steps_per_epoch)*100:.1f}%)")
            logger.info(f"  - Scheduler: Warm-up + Cosine Annealing")
            # Gradient clipping 상태
            if hasattr(self, 'gradient_clipping_hook') and self.gradient_clipping_hook:
                logger.info(f"  - Gradient Clipping: 활성화 (max_norm={getattr(self.cfg.trainer, 'gradient_clip_norm', 0.0)})")
            else:
                logger.info(f"  - Gradient Clipping: 비활성화")
            # Early stopping 상태
            early_stopping = self._get_early_stopping()
            if early_stopping:
                logger.info(f"  - Early Stopping: 활성화 (metric={early_stopping.metric_key}, patience={early_stopping.patience})")
            else:
                logger.info(f"  - Early Stopping: 비활성화")
        else:
            logger.info(f"학습 시작 (에포크: {effective_epochs}, 학습률: {self.cfg.trainer.lr})")
        logger.info("")
        
        # 학습 중 loss 로깅을 위한 콜백 추가
        callback = None
        if self.web_logger and self.web_logger.is_active:
            callback = WebLoggingCallback(self.web_logger)
            logger.info("학습 중 loss를 WandB에 로깅합니다.")
        
        # Early Stopping 정보 출력
        early_stopping = self._get_early_stopping()
        if early_stopping:
            logger.info(f"Early Stopping 활성화: {early_stopping.metric_key} 모니터링 (patience={early_stopping.patience})")
        
        try:
            # Optimizer 파라미터 구성
            optimizer_params = {"lr": self.cfg.trainer.lr}
            
            # Weight decay 추가 (기본값: 0.01)
            weight_decay = float(getattr(self.cfg.trainer, "weight_decay", 0.01))
            if weight_decay > 0:
                optimizer_params["weight_decay"] = weight_decay
            
            # AdamW beta 파라미터 추가 (선택사항)
            if hasattr(self.cfg.trainer, "beta1"):
                optimizer_params["betas"] = (
                    float(self.cfg.trainer.beta1),
                    float(getattr(self.cfg.trainer, "beta2", 0.999))
                )
            
            # AdamW epsilon 추가 (선택사항)
            if hasattr(self.cfg.trainer, "eps"):
                optimizer_params["eps"] = float(self.cfg.trainer.eps)
            
            # 평가 전 메모리 정리 (OOM 방지)
            if self.artifacts.evaluator:
                import torch
                import gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
            
            self.model.fit(
                train_objectives=[(self.artifacts.loader, self.artifacts.loss)],
                epochs=effective_epochs,
                warmup_steps=self.artifacts.warmup_steps,
                scheduler="warmupcosine",
                optimizer_params=optimizer_params,
                use_amp=bool(self.cfg.trainer.use_amp),
                show_progress_bar=True,
                evaluator=self.artifacts.evaluator,
                evaluation_steps=self.cfg.trainer.eval_steps if self.artifacts.evaluator else None,
                callback=callback,  # 학습 중 loss 로깅 콜백
            )
        except EarlyStoppingException as e:
            logger.info(f"Early Stopping으로 인해 학습이 조기 종료되었습니다: {e}")
            # Early stopping이 발생했지만 정상적인 종료로 처리
        except Exception as e:
            # 예외 발생 시 메모리 정리
            import torch
            import gc
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.warning("예외 발생 후 GPU 메모리 정리 완료")
            except Exception:
                pass
            raise  # 예외 재발생
        finally:
            # 학습 종료 후 항상 메모리 정리 (다음 run을 위해)
            import torch
            import gc
            try:
                if torch.cuda.is_available():
                    # 모델을 CPU로 이동하여 GPU 메모리에서 제거
                    try:
                        if hasattr(self, 'model') and self.model is not None:
                            # 모델을 CPU로 이동
                            self.model.to('cpu')
                            logger.debug("모델을 CPU로 이동 완료")
                        if hasattr(self, 'encoder') and self.encoder is not None:
                            if hasattr(self.encoder, 'model') and self.encoder.model is not None:
                                self.encoder.model.to('cpu')
                                logger.debug("Encoder 모델을 CPU로 이동 완료")
                    except Exception as e:
                        logger.debug(f"모델 CPU 이동 중 오류 (무시됨): {e}")
                    
                    # 강력한 메모리 정리
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # 모든 GPU 디바이스에서 메모리 정리
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()  # IPC 메모리 정리
                    
                    # Python GC로 남은 객체 정리
                    gc.collect()
                    gc.collect()  # 추가 GC (순환 참조 정리)
                    logger.debug("학습 종료 후 GPU 메모리 정리 완료")
            except Exception:
                pass

        logger.info("")
        
        # 모델 로컬 저장 (항상 수행)
        logger.info("모델 저장 중...")
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        final_model_path = os.path.join(self.cfg.out_dir, "bi_encoder")
        self.model.save(final_model_path)
        logger.info(f"✅ 최종 모델 저장 완료: {final_model_path}")
        
        # Early Stopping이 활성화된 경우 최고 성능 모델 확인
        early_stopping = self._get_early_stopping()
        model_path_for_artifact = None  # artifact 업로드를 위한 경로 (최고 성능 모델 우선)
        
        if early_stopping and early_stopping.get_best_step() >= 0:
            best_path = os.path.join(self.cfg.out_dir, "bi_encoder_best")
            if os.path.exists(best_path):
                logger.info(f"✅ 최고 성능 모델 경로: {best_path}")
                logger.info(f"최고 성능: {early_stopping.metric_key}={early_stopping.get_best_score():.4f} (step {early_stopping.get_best_step()})")
                # 최고 성능 모델을 artifact로 업로드
                model_path_for_artifact = best_path
            else:
                logger.info("최고 성능 모델이 아직 저장되지 않았습니다. 최종 모델을 artifact로 업로드합니다.")
                model_path_for_artifact = final_model_path
        else:
            # Early Stopping이 비활성화된 경우 최종 모델을 artifact로 업로드
            logger.info("Early Stopping이 비활성화되어 있습니다. 최종 모델을 artifact로 업로드합니다.")
            model_path_for_artifact = final_model_path
        
        # 웹 로깅에 모델 아티팩트 저장 (최고 성능 모델 우선)
        if self.web_logger and self.web_logger.is_active and model_path_for_artifact:
            try:
                logger.info(f"모델 artifact 업로드 중: {model_path_for_artifact}")
                self.web_logger.log_artifact(model_path_for_artifact, artifact_path="model")
                logger.info(f"✅ 모델이 웹 로깅 서비스에 업로드되었습니다: {model_path_for_artifact}")
            except Exception as e:
                logger.warning(f"모델 아티팩트 업로드 실패: {e}")
                import traceback
                logger.debug(f"상세 에러: {traceback.format_exc()}")

        if self.cfg.trainer.eval_pairs and os.path.exists(self.cfg.trainer.eval_pairs):
            logger.info("")
            logger.info("최종 평가 실행 중...")
            recall = eval_recall_at_k(
                encoder=self.encoder,
                passages=self.passages,
                eval_pairs_path=self.cfg.trainer.eval_pairs,
                read_jsonl_fn=read_jsonl,
                k=self.cfg.trainer.k,
            )
            logger.info(f"✅ Recall@{self.cfg.trainer.k}: {recall:.4f} ({recall*100:.2f}%)")
            
            # 웹 로깅에 최종 평가 결과 전송
            if self.web_logger and self.web_logger.is_active:
                self._log_evaluation_metrics({
                    f"eval/recall@{self.cfg.trainer.k}": recall,
                })
        
        # Gradient clipping hook 제거
        if hasattr(self, 'gradient_clipping_hook') and self.gradient_clipping_hook:
            stats = self.gradient_clipping_hook.get_stats()
            logger.info(
                f"Gradient clipping 통계: "
                f"총 {stats['total_backwards']}회 backward, "
                f"{stats['clipped_backwards']}회 clipping "
                f"(비율: {stats['clipping_ratio']:.2%}, "
                f"마지막 norm: {stats['last_norm']:.4f})"
            )
            self.gradient_clipping_hook.remove_hook()
        
        # 웹 로깅 종료
        if self.web_logger and self.web_logger.is_active:
            self.web_logger.finish()


def train_bi(cfg: DictConfig) -> None:
    """
    편의 함수: 단일 호출로 학습 실행.
    """
    trainer = BiEncoderTrainer(cfg)
    trainer.train()

