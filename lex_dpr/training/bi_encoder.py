# lex_dpr/training/bi_encoder.py
from __future__ import annotations

import math
import os
from typing import List

from omegaconf import DictConfig
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

from ..data import load_passages
from ..eval import build_ir_evaluator, eval_recall_at_k
from ..models.factory import get_bi_encoder
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


def train_bi(cfg: DictConfig) -> None:
    """
    Bi-Encoder 학습 루틴.
    """
    set_seed(cfg.seed)

    template_mode = _resolve_template_mode(cfg.model)

    passages = load_passages(cfg.data.passages)
    pairs = list(read_jsonl(cfg.data.pairs))

    max_len = int(getattr(cfg.model, "max_len", 0) or 0)
    encoder = get_bi_encoder(
        cfg.model.bi_model,
        template=template_mode.value,
        max_len=max_len if max_len > 0 else None,
    )
    model = encoder.model

    if max_len > 0:
        print(f"[train_bi] set model.max_seq_length = {model.max_seq_length}")

    examples: List[InputExample] = []
    miss_pos = 0
    for row in pairs:
        q_text = tq(row["query_text"], template_mode)
        for pid in row["positive_passages"]:
            passage = passages.get(pid)
            if not passage:
                miss_pos += 1
                continue
            p_text = tp(passage["text"], template_mode)
            examples.append(InputExample(texts=[q_text, p_text]))

    if miss_pos:
        print(f"[train_bi] skipped positives not in corpus: {miss_pos}")

    multiply = int(getattr(cfg.data, "multiply", 0) or 0)
    if multiply > 1:
        examples = _apply_multiply(examples, multiply)

    n_examples = len(examples)
    if n_examples == 0:
        raise ValueError(
            "[train_bi] No training examples. pairs/positives가 corpus id와 매칭되는지 확인하세요."
        )

    batch_size = int(cfg.data.batches.bi)
    if n_examples < batch_size:
        print(f"[train_bi] Reducing batch size: {batch_size} -> {n_examples}")
        batch_size = n_examples
        cfg.data.batches.bi = n_examples  # 로그 및 추후 참조를 위해 업데이트

    loader = DataLoader(examples, batch_size=batch_size, shuffle=True, drop_last=False)
    loss = losses.MultipleNegativesRankingLoss(model, scale=cfg.trainer.temperature)

    evaluator = None
    if cfg.trainer.eval_pairs:
        evaluator, _ = build_ir_evaluator(
            passages=passages,
            eval_pairs_path=cfg.trainer.eval_pairs,
            read_jsonl_fn=read_jsonl,
            k_vals=cfg.trainer.k_values,
            template=template_mode,
        )

    steps_per_epoch = max(1, math.ceil(len(examples) / batch_size))
    total_steps = steps_per_epoch * cfg.trainer.epochs
    warmup_steps = max(10, int(total_steps * 0.1))

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=cfg.trainer.epochs,
        warmup_steps=warmup_steps,
        scheduler="warmupcosine",
        optimizer_params={"lr": cfg.trainer.lr},
        use_amp=bool(cfg.trainer.use_amp),
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=cfg.trainer.eval_steps if evaluator else None,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, "bi_encoder")
    model.save(save_path)
    print(f"[train_bi] saved model to {save_path}")

    if cfg.trainer.eval_pairs:
        recall = eval_recall_at_k(
            encoder=encoder,
            passages=passages,
            eval_pairs_path=cfg.trainer.eval_pairs,
            read_jsonl_fn=read_jsonl,
            k=cfg.trainer.k,
        )
        print(f"[Eval] Recall@{cfg.trainer.k}: {recall:.4f}")

