# lex_dpr/trainer/base_trainer.py
from __future__ import annotations

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

        self.template_mode = _resolve_template_mode(cfg.model)
        self.passages = load_passages(cfg.data.passages)
        self.pairs = list(read_jsonl(cfg.data.pairs))

        self.encoder = self._build_encoder()
        self.model = self.encoder.model

        self.examples = self._build_examples()
        self.batch_size = self._resolve_batch_size(len(self.examples))
        self.artifacts = self._build_artifacts()

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
            print(f"[BiEncoderTrainer] set model.max_seq_length = {encoder.model.max_seq_length}")
        
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
                print(f"[BiEncoderTrainer] Attaching LoRA adapter: r={r}, alpha={alpha}, dropout={dropout}, target_modules={target_modules}")
            else:
                print(f"[BiEncoderTrainer] Attaching LoRA adapter: r={r}, alpha={alpha}, dropout={dropout}, target_modules=auto-detect")
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
                print(f"[BiEncoderTrainer] LoRA adapter attached. Only LoRA parameters will be trained.")
            except Exception as e:
                print(f"[BiEncoderTrainer] Warning: enable_lora_only_train failed: {e}")
                print(f"[BiEncoderTrainer] Continuing with PEFT default settings...")
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
                            print(f"[BiEncoderTrainer] Gradient checkpointing enabled.")
                        elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "gradient_checkpointing_enable"):
                            base_model.encoder.gradient_checkpointing_enable()
                            print(f"[BiEncoderTrainer] Gradient checkpointing enabled (encoder).")
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
            print(f"[BiEncoderTrainer] skipped positives not in corpus: {miss_pos}")

        multiply = int(getattr(self.cfg.data, "multiply", 0) or 0)
        if multiply > 1:
            examples = _apply_multiply(examples, multiply)

        if not examples:
            raise ValueError(
                "[BiEncoderTrainer] No training examples. 확인: pairs/positive ids가 corpus와 일치하는지."
            )
        return examples

    def _resolve_batch_size(self, n_examples: int) -> int:
        batch_size = int(self.cfg.data.batches.bi)
        if n_examples < batch_size:
            print(f"[BiEncoderTrainer] Reducing batch size: {batch_size} -> {n_examples}")
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
            print(f"[BiEncoderTrainer] Warning: eval_pairs file not found: {self.cfg.trainer.eval_pairs}. Skipping evaluation.")

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
            print(f"[BiEncoderTrainer] Note: gradient_accumulation_steps={gradient_accumulation_steps} is set but not supported by sentence-transformers.")
            print(f"[BiEncoderTrainer] Effective batch size = {self.batch_size} * {gradient_accumulation_steps} = {self.batch_size * gradient_accumulation_steps}")
        
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

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        save_path = os.path.join(self.cfg.out_dir, "bi_encoder")
        self.model.save(save_path)
        print(f"[BiEncoderTrainer] saved model to {save_path}")

        if self.cfg.trainer.eval_pairs and os.path.exists(self.cfg.trainer.eval_pairs):
            recall = eval_recall_at_k(
                encoder=self.encoder,
                passages=self.passages,
                eval_pairs_path=self.cfg.trainer.eval_pairs,
                read_jsonl_fn=read_jsonl,
                k=self.cfg.trainer.k,
            )
            print(f"[Eval] Recall@{self.cfg.trainer.k}: {recall:.4f}")


def train_bi(cfg: DictConfig) -> None:
    """
    편의 함수: 단일 호출로 학습 실행.
    """
    trainer = BiEncoderTrainer(cfg)
    trainer.train()

