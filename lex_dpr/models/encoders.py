# lex_dpr/models/encoders.py
from typing import Iterable, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from .templates import TemplateMode, tq, tp

class BiEncoder:
    def __init__(self, name_or_path: str, template: TemplateMode = TemplateMode.BGE,
                 normalize: bool = True, max_seq_length: Optional[int] = None,
                 query_max_seq_length: Optional[int] = None,
                 passage_max_seq_length: Optional[int] = None,
                 trust_remote_code: bool = True, peft_adapter_path: Optional[str] = None):
        """
        Args:
            name_or_path: 모델 경로 또는 HuggingFace 모델 이름
            template: 템플릿 모드
            normalize: 임베딩 정규화 여부
            max_seq_length: 최대 시퀀스 길이 (쿼리/패시지 공통, query_max_seq_length/passage_max_seq_length가 없을 때 사용)
            query_max_seq_length: 쿼리 최대 시퀀스 길이 (우선순위: query_max_seq_length > max_seq_length)
            passage_max_seq_length: 패시지 최대 시퀀스 길이 (우선순위: passage_max_seq_length > max_seq_length)
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
        
        # 모델의 원래 max_seq_length 확인 (토크나이저의 model_max_length)
        original_model_max_length = getattr(self.model.tokenizer, 'model_max_length', None)
        if original_model_max_length is None:
            # SentenceTransformer 내부 토크나이저 접근
            try:
                if hasattr(self.model, '_modules'):
                    for module in self.model._modules.values():
                        if hasattr(module, 'tokenizer'):
                            original_model_max_length = getattr(module.tokenizer, 'model_max_length', 512)
                            break
            except:
                pass
        if original_model_max_length is None:
            original_model_max_length = 512  # 기본값
        
        # ko-simcse 모델 확인 (jhgan/ko-sroberta-multitask)
        # 모델 이름이 ko-simcse 또는 jhgan/ko-sroberta-multitask인 경우 확인
        # original_model_max_length가 128인 경우도 ko-simcse일 가능성이 높으므로 클리핑
        is_ko_simcse = False
        model_name_lower = name_or_path.lower()
        if "ko-simcse" in model_name_lower or "ko-sroberta-multitask" in model_name_lower:
            is_ko_simcse = True
        
        # ko-simcse 모델의 경우 max_seq_length를 128로 제한
        # original_model_max_length가 128인 경우도 자동으로 클리핑 (안전한 처리)
        if (is_ko_simcse or original_model_max_length == 128) and original_model_max_length == 128:
            if max_seq_length and max_seq_length > 128:
                print(f"ℹ️  정보: ko-simcse 모델은 최대 길이 128을 지원합니다. max_seq_length를 {max_seq_length}에서 128로 조정합니다.")
                max_seq_length = 128
            if query_max_seq_length and query_max_seq_length > 128:
                print(f"ℹ️  정보: ko-simcse 모델은 최대 길이 128을 지원합니다. query_max_seq_length를 {query_max_seq_length}에서 128로 조정합니다.")
                query_max_seq_length = 128
            if passage_max_seq_length and passage_max_seq_length > 128:
                print(f"ℹ️  정보: ko-simcse 모델은 최대 길이 128을 지원합니다. passage_max_seq_length를 {passage_max_seq_length}에서 128로 조정합니다.")
                passage_max_seq_length = 128
        
        # 시퀀스 길이 설정 (쿼리/패시지 분리 지원)
        # 기본값: max_seq_length를 사용하되, query/passage 별도 설정이 있으면 우선 사용
        self.query_max_seq_length = query_max_seq_length if query_max_seq_length is not None else max_seq_length
        self.passage_max_seq_length = passage_max_seq_length if passage_max_seq_length is not None else max_seq_length
        
        # 모델의 기본 max_seq_length 설정 (더 큰 값으로 설정하여 유연성 확보)
        if max_seq_length:
            self.model.max_seq_length = int(max_seq_length)
        elif query_max_seq_length or passage_max_seq_length:
            # query/passage 중 더 큰 값으로 설정
            max_len = max(
                self.query_max_seq_length or 512,
                self.passage_max_seq_length or 512
            )
            self.model.max_seq_length = int(max_len)
        else:
            # 설정이 없으면 모델의 원래 길이 사용
            self.model.max_seq_length = int(original_model_max_length)
        
        # 경고: 설정한 길이가 원래 모델 길이와 다를 때
        final_max_len = self.model.max_seq_length
        if final_max_len > original_model_max_length:
            print(f"⚠️  경고: 설정한 max_seq_length({final_max_len})가 모델의 원래 길이({original_model_max_length})보다 깁니다.")
            print(f"   더 긴 시퀀스는 position embeddings가 부족할 수 있어 성능 저하가 발생할 수 있습니다.")
            print(f"   권장: {original_model_max_length} 이하로 설정하거나, 모델이 더 긴 길이를 지원하는지 확인하세요.")
        elif final_max_len < original_model_max_length:
            print(f"ℹ️  정보: 설정한 max_seq_length({final_max_len})가 모델의 원래 길이({original_model_max_length})보다 짧습니다.")
            print(f"   더 긴 입력은 자동으로 잘립니다(truncation). 이는 일반적으로 문제없습니다.")
        
        # Query/Passage 분리 설정 시 경고
        if query_max_seq_length or passage_max_seq_length:
            if self.query_max_seq_length and self.query_max_seq_length > original_model_max_length:
                print(f"⚠️  경고: query_max_seq_length({self.query_max_seq_length})가 모델 원래 길이({original_model_max_length})보다 깁니다.")
            if self.passage_max_seq_length and self.passage_max_seq_length > original_model_max_length:
                print(f"⚠️  경고: passage_max_seq_length({self.passage_max_seq_length})가 모델 원래 길이({original_model_max_length})보다 깁니다.")
        
        # 추론 모드로 설정 (임베딩 추출 시 학습이 아닌 추론)
        self.model.eval()
        
        self.template = template
        self.normalize = normalize

    def encode_queries(self, queries: Iterable[str], batch_size=64) -> np.ndarray:
        """쿼리 임베딩 생성 (추론 모드)"""
        # 명시적으로 eval 모드로 설정
        self.model.eval()
        texts = [tq(q, self.template) for q in queries]
        
        # 쿼리 전용 시퀀스 길이 적용
        original_max_seq_length = self.model.max_seq_length
        if self.query_max_seq_length is not None:
            self.model.max_seq_length = int(self.query_max_seq_length)
        
        try:
            result = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                     normalize_embeddings=self.normalize, show_progress_bar=False)
        finally:
            # 원래 길이로 복원
            self.model.max_seq_length = original_max_seq_length
        
        return result

    def encode_passages(self, passages: Iterable[str], batch_size=64) -> np.ndarray:
        """패시지 임베딩 생성 (추론 모드)"""
        # 명시적으로 eval 모드로 설정
        self.model.eval()
        texts = [tp(p, self.template) for p in passages]
        
        # 패시지 전용 시퀀스 길이 적용
        original_max_seq_length = self.model.max_seq_length
        if self.passage_max_seq_length is not None:
            self.model.max_seq_length = int(self.passage_max_seq_length)
        
        try:
            result = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                     normalize_embeddings=self.normalize, show_progress_bar=False)
        finally:
            # 원래 길이로 복원
            self.model.max_seq_length = original_max_seq_length
        
        return result
