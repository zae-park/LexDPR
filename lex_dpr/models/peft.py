# lex_dpr/models/peft.py
from typing import Optional, Iterable
from sentence_transformers import SentenceTransformer, models as st_models
from peft import LoraConfig, get_peft_model, TaskType

"""
model = SentenceTransformer(cfg.model.bi_model)
from lex_dpr.models.peft import attach_lora_to_st, enable_lora_only_train
model = attach_lora_to_st(model, r=16, alpha=32, dropout=0.05, target_modules=["q_proj","v_proj"])
enable_lora_only_train(model)
"""

def _get_st_transformer(st_model: SentenceTransformer) -> st_models.Transformer:
    """
    SentenceTransformer 내부 첫 모듈(Transformer)을 가져온다.
    주의: 커스텀 구성이면 순서가 다를 수 있으니 필요하면 보강.
    """
    for m in st_model.modules():
        if isinstance(m, st_models.Transformer):
            return m
    raise RuntimeError("No sentence_transformers.models.Transformer found in SentenceTransformer.")

def attach_lora_to_st(
    st_model: SentenceTransformer,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[Iterable[str]] = None,
) -> SentenceTransformer:
    """
    SentenceTransformer의 내부 Transformer에 LoRA 어댑터를 부착.
    - target_modules 예시: ["q_proj","v_proj"] 또는 ["query","value"] 등 모델에 맞게.
    """
    t = _get_st_transformer(st_model)
    base = t.auto_model  # HuggingFace AutoModel
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # 분류가 아닌 임베딩 추출
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules) if target_modules else None,
        inference_mode=False,
    )
    peft_model = get_peft_model(base, cfg)
    # Sentence-Transformers의 Transformer 모듈이 참조하는 모델을 교체
    t.auto_model = peft_model
    return st_model

def enable_lora_only_train(st_model: SentenceTransformer):
    """LoRA 가중치만 학습되도록 동결."""
    for p in st_model.parameters():
        p.requires_grad = False
    t = _get_st_transformer(st_model)
    for n, p in t.auto_model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
