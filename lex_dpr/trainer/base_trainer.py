# lex_dpr/trainer/base_trainer.py
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional

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

logger = logging.getLogger("lex_dpr.trainer")


def _resolve_template_mode(cfg_model) -> TemplateMode:
    use_bge = bool(getattr(cfg_model, "use_bge_template", True))
    return TemplateMode.BGE if use_bge else TemplateMode.NONE


def _apply_multiply(examples: List[InputExample], multiply: int) -> List[InputExample]:
    if multiply <= 1:
        return examples
    return examples * multiply


@dataclass
class TrainerArtifacts:
    loader: DataLoader
    loss: losses.MultipleNegativesRankingLoss
    evaluator: Optional[object]
    steps_per_epoch: int
    warmup_steps: int


class BiEncoderTrainer:
    """
    학습 스크립트와 분리된 BI-Encoder Trainer.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        logger.info(f"시드 설정: {cfg.seed}")
        
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

    # ------------------------------
    # Build helpers
    # ------------------------------
    def _build_encoder(self):
        max_len = int(getattr(self.cfg.model, "max_len", 0) or 0)
        encoder = get_bi_encoder(
            self.cfg.model.bi_model,
            template=self.template_mode.value,
            max_len=max_len if max_len > 0 else None,
        )
        if max_len > 0:
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
        for row in self.pairs:
            q_text = tq(row["query_text"], self.template_mode)
            for pid in row["positive_passages"]:
                passage = self.passages.get(pid)
                if not passage:
                    miss_pos += 1
                    continue
                p_text = tp(passage["text"], self.template_mode)
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
        loader = DataLoader(
            self.examples,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        loss = losses.MultipleNegativesRankingLoss(self.model, scale=self.cfg.trainer.temperature)

        evaluator = None
        if self.cfg.trainer.eval_pairs and os.path.exists(self.cfg.trainer.eval_pairs):
            evaluator, _ = build_ir_evaluator(
                passages=self.passages,
                eval_pairs_path=self.cfg.trainer.eval_pairs,
                read_jsonl_fn=read_jsonl,
                k_vals=self.cfg.trainer.k_values,
                template=self.template_mode,
            )
        elif self.cfg.trainer.eval_pairs:
            logger.warning(f"eval_pairs 파일을 찾을 수 없습니다: {self.cfg.trainer.eval_pairs}. 평가를 건너뜁니다.")

        steps_per_epoch = max(1, math.ceil(len(self.examples) / self.batch_size))
        total_steps = steps_per_epoch * self.cfg.trainer.epochs
        warmup_steps = max(10, int(total_steps * 0.1))

        return TrainerArtifacts(
            loader=loader,
            loss=loss,
            evaluator=evaluator,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
        )

    # ------------------------------
    # Public API
    # ------------------------------
    def train(self) -> None:
        # sentence-transformers의 fit() 메서드는 gradient_accumulation_steps를 지원하지 않음
        # 대신 배치 사이즈를 조정하여 효과적인 배치 크기를 조절
        gradient_accumulation_steps = int(getattr(self.cfg.trainer, "gradient_accumulation_steps", 1))
        if gradient_accumulation_steps > 1:
            effective_batch_size = self.batch_size * gradient_accumulation_steps
            logger.info(f"참고: gradient_accumulation_steps={gradient_accumulation_steps}는 sentence-transformers에서 지원되지 않습니다.")
            logger.info(f"효과적인 배치 크기: {self.batch_size} × {gradient_accumulation_steps} = {effective_batch_size}")
        
        logger.info(f"학습 시작 (에포크: {self.cfg.trainer.epochs}, 학습률: {self.cfg.trainer.lr})")
        logger.info("")
        
        self.model.fit(
            train_objectives=[(self.artifacts.loader, self.artifacts.loss)],
            epochs=self.cfg.trainer.epochs,
            warmup_steps=self.artifacts.warmup_steps,
            scheduler="warmupcosine",
            optimizer_params={"lr": self.cfg.trainer.lr},
            use_amp=bool(self.cfg.trainer.use_amp),
            show_progress_bar=True,
            evaluator=self.artifacts.evaluator,
            evaluation_steps=self.cfg.trainer.eval_steps if self.artifacts.evaluator else None,
        )

        logger.info("")
        logger.info("모델 저장 중...")
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        save_path = os.path.join(self.cfg.out_dir, "bi_encoder")
        self.model.save(save_path)
        logger.info(f"✅ 모델 저장 완료: {save_path}")

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


def train_bi(cfg: DictConfig) -> None:
    """
    편의 함수: 단일 호출로 학습 실행.
    """
    trainer = BiEncoderTrainer(cfg)
    trainer.train()

