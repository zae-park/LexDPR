# lex_dpr/models/encoders.py
from typing import Iterable, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from .templates import TemplateMode, tq, tp

class BiEncoder:
    def __init__(self, name_or_path: str, template: TemplateMode = TemplateMode.BGE,
                 normalize: bool = True, max_seq_length: Optional[int] = None,
                 trust_remote_code: bool = True, peft_adapter_path: Optional[str] = None):
        """
        Args:
            name_or_path: 모델 경로 또는 HuggingFace 모델 이름
            template: 템플릿 모드
            normalize: 임베딩 정규화 여부
            max_seq_length: 최대 시퀀스 길이
            trust_remote_code: 원격 코드 신뢰 여부
            peft_adapter_path: PEFT 어댑터 경로 (None이면 자동 감지)
        """
        # SentenceTransformer는 PEFT 어댑터가 포함된 체크포인트를 자동으로 로드할 수 있음
        # adapter_config.json이 있으면 자동으로 감지하여 로드
        model_path = Path(name_or_path)
        if model_path.exists() and (model_path / "adapter_config.json").exists():
            # PEFT 어댑터가 포함된 체크포인트
            print(f"[BiEncoder] Loading model with PEFT adapter from {name_or_path}")
            # adapter_config.json에서 base 모델 정보 읽기
            import json
            with open(model_path / "adapter_config.json", "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            
            if base_model_name:
                # Base 모델을 먼저 로드하고 PEFT 어댑터 적용
                print(f"[BiEncoder] Loading base model: {base_model_name}")
                self.model = SentenceTransformer(base_model_name, trust_remote_code=trust_remote_code)
                # PEFT 어댑터 로드
                from .peft import _get_st_transformer
                from peft import PeftModel
                t = _get_st_transformer(self.model)
                base_model = t.auto_model
                peft_model = PeftModel.from_pretrained(base_model, str(model_path))
                t.auto_model = peft_model
                print(f"[BiEncoder] PEFT adapter loaded from {name_or_path}")
            else:
                # Base 모델 정보가 없으면 일반 로드 시도
                self.model = SentenceTransformer(str(model_path), trust_remote_code=trust_remote_code)
        elif peft_adapter_path:
            # 별도로 지정된 PEFT 어댑터 경로 (base 모델과 어댑터가 분리된 경우)
            print(f"[BiEncoder] Loading base model {name_or_path} with PEFT adapter from {peft_adapter_path}")
            self.model = SentenceTransformer(name_or_path, trust_remote_code=trust_remote_code)
            # SentenceTransformer 내부 모델에 PEFT 적용
            from .peft import _get_st_transformer
            from peft import PeftModel
            t = _get_st_transformer(self.model)
            base_model = t.auto_model
            if not hasattr(base_model, "base_model"):
                # PEFT 어댑터를 로드해야 하는 경우
                peft_model = PeftModel.from_pretrained(base_model, peft_adapter_path)
                t.auto_model = peft_model
                print(f"[BiEncoder] PEFT adapter loaded from {peft_adapter_path}")
        else:
            # 일반 모델 로드
            self.model = SentenceTransformer(name_or_path, trust_remote_code=trust_remote_code)
        
        if max_seq_length:
            # Sentence-Transformers가 내부 tokenizer 길이를 이 값으로 사용
            self.model.max_seq_length = int(max_seq_length)
        
        # 추론 모드로 설정 (임베딩 추출 시 학습이 아닌 추론)
        self.model.eval()
        
        self.template = template
        self.normalize = normalize

    def encode_queries(self, queries: Iterable[str], batch_size=64) -> np.ndarray:
        """쿼리 임베딩 생성 (추론 모드)"""
        # 명시적으로 eval 모드로 설정
        self.model.eval()
        texts = [tq(q, self.template) for q in queries]
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=self.normalize, show_progress_bar=False)

    def encode_passages(self, passages: Iterable[str], batch_size=64) -> np.ndarray:
        """패시지 임베딩 생성 (추론 모드)"""
        # 명시적으로 eval 모드로 설정
        self.model.eval()
        texts = [tp(p, self.template) for p in passages]
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=self.normalize, show_progress_bar=False)
