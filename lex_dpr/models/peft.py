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

def _find_target_modules(model, candidate_patterns: list) -> list:
    """
    모델에서 사용 가능한 target_modules를 자동으로 찾습니다.
    attention 관련 레이어만 찾습니다.
    """
    import torch.nn as nn
    available_modules = set()
    
    # 모든 모듈을 순회하면서 attention 관련 레이어 찾기
    for name, module in model.named_modules():
        # attention 관련 레이어인지 확인
        if "attention" in name.lower() or "attn" in name.lower():
            # Linear 레이어인지 확인
            if isinstance(module, nn.Linear):
                # 모듈 이름의 마지막 부분 확인 (예: "query", "value", "q_proj", "v_proj")
                name_parts = name.split(".")
                last_part = name_parts[-1].lower()
                
                # candidate_patterns와 매칭
                for pattern in candidate_patterns:
                    if pattern.lower() in last_part:
                        available_modules.add(pattern)
                        break
    
    # query, value 또는 q_proj, v_proj 우선순위로 정렬
    priority_order = ["query", "value", "q_proj", "v_proj", "key", "k_proj"]
    sorted_modules = sorted(available_modules, key=lambda x: priority_order.index(x) if x in priority_order else 999)
    
    # 최대 2개만 반환 (query, value 또는 q_proj, v_proj)
    return sorted_modules[:2] if sorted_modules else []


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
    - None이면 자동으로 모델 구조를 확인하여 적절한 모듈을 찾습니다.
    """
    t = _get_st_transformer(st_model)
    base = t.auto_model  # HuggingFace AutoModel
    
    # target_modules가 None이면 자동 감지
    if target_modules is None:
        # 일반적인 패턴들을 시도
        candidates = ["query", "value", "key", "q_proj", "v_proj", "k_proj"]
        found_modules = _find_target_modules(base, candidates)
        if found_modules:
            target_modules = found_modules[:2]  # query, value 또는 q_proj, v_proj만 사용
            print(f"[attach_lora_to_st] Auto-detected target_modules: {target_modules}")
        else:
            # 기본값으로 query, value 시도
            target_modules = ["query", "value"]
            print(f"[attach_lora_to_st] Using default target_modules: {target_modules}")
    
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # 분류가 아닌 임베딩 추출
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules) if target_modules else None,
        inference_mode=False,
    )
    peft_model = get_peft_model(base, cfg)
    # PEFT 모델을 train 모드로 설정
    peft_model.train()
    # Sentence-Transformers의 Transformer 모듈이 참조하는 모델을 교체
    t.auto_model = peft_model
    # SentenceTransformer도 train 모드로 설정
    st_model.train()
    return st_model

def enable_lora_only_train(st_model: SentenceTransformer):
    """LoRA 가중치만 학습되도록 동결."""
    # 모델을 train 모드로 설정
    st_model.train()
    
    t = _get_st_transformer(st_model)
    peft_model = t.auto_model
    
    # PEFT 모델의 trainable 파라미터 정보 출력
    if hasattr(peft_model, "print_trainable_parameters"):
        print("[enable_lora_only_train] PEFT model trainable parameters:")
        peft_model.print_trainable_parameters()
    
    # PEFT 모델의 모든 파라미터를 확인하여 LoRA 파라미터만 학습 가능하게 설정
    # PEFT는 자동으로 base_model을 동결하지만, 명시적으로 확인
    trainable_count = 0
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            # LoRA 파라미터인지 확인
            if "lora_" not in name and "lora_A" not in name and "lora_B" not in name:
                # LoRA가 아닌 파라미터가 학습 가능하면 경고
                print(f"[enable_lora_only_train] Warning: Non-LoRA parameter '{name}' is trainable")
    
    # SentenceTransformer의 다른 모듈들(pooling 등)은 학습 가능하게 유지
    # 이들은 보통 작은 파라미터 수를 가지므로 학습 가능하게 두는 것이 좋음
    
    # 전체 모델의 학습 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in st_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in st_model.parameters())
    if total_params > 0:
        print(f"[enable_lora_only_train] Total trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    else:
        print(f"[enable_lora_only_train] ERROR: No trainable parameters found! This will cause training to fail.")
        # 디버깅: 어떤 파라미터들이 있는지 확인
        print("[enable_lora_only_train] Debugging: Checking parameter names...")
        for name, param in st_model.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
