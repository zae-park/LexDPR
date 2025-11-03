# lex_dpr/models/export.py
import os
from sentence_transformers import SentenceTransformer

def save_sentence_transformer(model: SentenceTransformer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir)

def load_sentence_transformer(path: str) -> SentenceTransformer:
    return SentenceTransformer(path)

# --- ONNX 내보내기 (선택) ---
# 참고: 아래는 "Transformer encoder"만 내보냄. mean-pooling은 런타임에서 처리 권장.
def export_transformer_to_onnx(
    model: SentenceTransformer, onnx_path: str, opset: int = 17, max_seq_len: int = 512
):
    from transformers.onnx import export, FeaturesManager
    from pathlib import Path
    import torch
    # Sentence-Transformers 내부 Transformer 획득
    from sentence_transformers import models as st_models
    transformer = None
    for m in model.modules():
        if isinstance(m, st_models.Transformer):
            transformer = m
            break
    if transformer is None:
        raise RuntimeError("No Transformer module found for ONNX export.")

    hf_model = transformer.auto_model
    tokenizer = transformer.tokenizer

    task = "feature-extraction"
    onnx_dir = Path(onnx_path).parent
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic axes를 가진 ONNX 내보내기
    export(
        preprocessor=tokenizer,
        model=hf_model,
        config=FeaturesManager.get_exporter_config(task, hf_model.config.__class__),
        opset=opset,
        output=Path(onnx_path),
        device="cpu",
        overwrite=True,
    )
    return str(onnx_path)
