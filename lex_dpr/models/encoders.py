# lex_dpr/models/encoders.py
from typing import Iterable, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .templates import TemplateMode, tq, tp

class BiEncoder:
    def __init__(self, name_or_path: str = "default", template: TemplateMode = TemplateMode.BGE,
                 normalize: bool = True, max_seq_length: Optional[int] = None,
                 query_max_seq_length: Optional[int] = None,
                 passage_max_seq_length: Optional[int] = None,
                 trust_remote_code: bool = True, peft_adapter_path: Optional[str] = None,
                 auto_download: bool = True):
        """
        Args:
            name_or_path: 모델 경로, HuggingFace 모델 이름, 또는 "default" (기본 모델 자동 다운로드)
            template: 템플릿 모드
            normalize: 임베딩 정규화 여부
            max_seq_length: 최대 시퀀스 길이 (쿼리/패시지 공통, query_max_seq_length/passage_max_seq_length가 없을 때 사용)
            query_max_seq_length: 쿼리 최대 시퀀스 길이 (우선순위: query_max_seq_length > max_seq_length)
            passage_max_seq_length: 패시지 최대 시퀀스 길이 (우선순위: passage_max_seq_length > max_seq_length)
            trust_remote_code: 원격 코드 신뢰 여부
            peft_adapter_path: PEFT 어댑터 경로 (None이면 자동 감지)
            auto_download: name_or_path="default"일 때 자동 다운로드 여부 (기본값: True)
        """
        # "default"인 경우 기본 모델 사용 (패키지 포함 모델 우선, 없으면 WandB 다운로드)
        if name_or_path == "default" or name_or_path is None:
            if auto_download:
                name_or_path = self._get_default_model_path()
            else:
                raise ValueError(
                    "name_or_path='default'이지만 auto_download=False입니다. "
                    "모델 경로를 지정하거나 auto_download=True로 설정하세요."
                )
        
        # SentenceTransformer는 PEFT 어댑터가 포함된 체크포인트를 자동으로 로드할 수 있음
        # adapter_config.json이 있으면 자동으로 감지하여 로드
        model_path = Path(name_or_path)
        self._model_path = str(model_path) if model_path.exists() else name_or_path  # 모델 경로 저장
        
        # 학습 설정 정보 자동 로드
        # 1. training_config.json에서 확인
        # 2. 없으면 config.py의 DEFAULT_MAX_LEN 사용
        training_max_len = None
        if model_path.exists() and (model_path / "training_config.json").exists():
            try:
                import json
                with open(model_path / "training_config.json", "r", encoding="utf-8") as f:
                    training_config = json.load(f)
                training_max_len = training_config.get("max_len")
            except Exception as e:
                pass
        
        # training_config.json에 없으면 config.py의 DEFAULT_MAX_LEN 사용
        if training_max_len is None:
            from .config import DEFAULT_MAX_LEN
            if DEFAULT_MAX_LEN is not None:
                training_max_len = DEFAULT_MAX_LEN
        
        # max_seq_length 자동 적용
        if training_max_len and max_seq_length is None:
            max_seq_length = training_max_len
            print(f"[BiEncoder] 학습 시 사용된 max_len({training_max_len})을 자동으로 적용합니다.")
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

    def get_embedding_dimension(self) -> int:
        """
        모델의 임베딩 차원을 반환합니다.
        
        Returns:
            임베딩 차원 (int)
        """
        # 테스트 임베딩 생성하여 차원 확인
        test_text = "test"
        test_embedding = self.model.encode([test_text], convert_to_numpy=True)
        return int(test_embedding.shape[1])
    
    def get_max_seq_length(self) -> int:
        """
        모델의 현재 최대 시퀀스 길이를 반환합니다.
        
        Returns:
            최대 시퀀스 길이 (int)
        """
        return int(self.model.max_seq_length)
    
    def get_training_config(self, model_path: Optional[str] = None) -> Optional[dict]:
        """
        다운로드한 모델에서 학습 시 사용된 설정 정보를 반환합니다.
        PEFT 어댑터인 경우 adapter_config.json에서 정보를 읽습니다.
        
        Args:
            model_path: 모델 경로 (None이면 자동 감지)
        
        Returns:
            학습 설정 정보 (dict) 또는 None
        """
        import json
        
        # 모델 경로 확인
        if model_path is None:
            # BiEncoder 초기화 시 전달된 경로 확인
            if hasattr(self, '_model_path'):
                model_path = self._model_path
            else:
                # SentenceTransformer에서 모델 경로 추출 시도
                try:
                    # tokenizer의 name_or_path에서 경로 추출
                    if hasattr(self.model, 'tokenizer'):
                        tokenizer_path = getattr(self.model.tokenizer, 'name_or_path', None)
                        if tokenizer_path and Path(tokenizer_path).exists():
                            model_path = str(Path(tokenizer_path).parent)
                except Exception:
                    pass
        
        if model_path:
            model_path = Path(model_path)
            # PEFT 어댑터 설정 확인
            if (model_path / "adapter_config.json").exists():
                try:
                    with open(model_path / "adapter_config.json", "r", encoding="utf-8") as f:
                        adapter_config = json.load(f)
                    return adapter_config
                except Exception:
                    pass
        
        return None
    
    def _get_default_model_path(self) -> str:
        """
        기본 모델 경로를 가져옵니다.
        1. 패키지에 포함된 모델 우선 사용 (DEFAULT_MODEL_PATH)
        2. 없으면 WandB에서 자동 다운로드 (DEFAULT_RUN_ID)
        
        Returns:
            모델 경로
        """
        from .config import DEFAULT_MODEL_PATH, DEFAULT_MAX_LEN
        
        # 방법 1: 패키지에 포함된 모델 사용
        if DEFAULT_MODEL_PATH:
            import importlib.resources
            try:
                # 패키지 내부 리소스로 접근
                # DEFAULT_MODEL_PATH는 "models/default_model" 같은 상대 경로
                package_path = importlib.resources.files("lex_dpr")
                model_path = package_path / DEFAULT_MODEL_PATH
                
                # 파일 시스템 경로로 변환
                if hasattr(model_path, '_path'):
                    # importlib.resources.files()는 Traversable 객체를 반환
                    # 실제 파일 시스템 경로로 변환
                    try:
                        actual_path = Path(str(model_path))
                    except:
                        # Traversable 객체인 경우 다른 방법 시도
                        import os
                        actual_path = Path(os.path.join(package_path._path if hasattr(package_path, '_path') else str(package_path), DEFAULT_MODEL_PATH))
                else:
                    actual_path = Path(str(model_path))
                
                if actual_path.exists():
                    print(f"[BiEncoder] 패키지에 포함된 모델 사용: {actual_path}")
                    return str(actual_path)
                else:
                    print(f"[BiEncoder] 패키지 내부 모델 경로를 찾을 수 없습니다: {actual_path}")
            except Exception as e:
                print(f"[BiEncoder] 패키지 내부 모델 로드 실패, WandB 다운로드로 전환: {e}")
                import traceback
                traceback.print_exc()
        
        # 방법 2: WandB에서 자동 다운로드
        return self._download_default_model()
    
    def _download_default_model(self) -> str:
        """
        패키지에 설정된 기본 모델을 WandB에서 자동 다운로드합니다.
        
        Returns:
            다운로드된 모델 경로
        """
        from .config import (
            DEFAULT_RUN_ID,
            DEFAULT_WANDB_PROJECT,
            DEFAULT_WANDB_ENTITY,
            DEFAULT_MODEL_CACHE_DIR,
            DEFAULT_METRIC,
            DEFAULT_GOAL,
        )
        
        if DEFAULT_RUN_ID is None:
            raise ValueError(
                "기본 모델이 설정되지 않았습니다. "
                "패키지 배포자가 DEFAULT_RUN_ID를 설정해야 합니다. "
                "또는 명시적으로 모델 경로를 지정하세요: BiEncoder('path/to/model')"
            )
        
        # 캐시 디렉토리 확인
        cache_dir = Path(os.path.expanduser(DEFAULT_MODEL_CACHE_DIR))
        model_dir = cache_dir / DEFAULT_RUN_ID
        
        # 이미 다운로드된 모델이 있으면 재사용
        if model_dir.exists() and (model_dir / "adapter_config.json").exists():
            print(f"[BiEncoder] 기존 모델 발견: {model_dir}")
            return str(model_dir)
        
        # WandB에서 모델 다운로드
        try:
            import wandb
            from wandb import Api
        except ImportError:
            raise ImportError(
                "WandB가 설치되지 않았습니다. 기본 모델을 다운로드하려면 "
                "'pip install wandb' 또는 'poetry install --extras wandb'로 설치하세요."
            )
        
        # WandB API 키 확인
        if not os.getenv("WANDB_API_KEY"):
            raise ValueError(
                "WANDB_API_KEY 환경 변수가 설정되지 않았습니다. "
                "기본 모델을 다운로드하려면 WandB API 키가 필요합니다."
            )
        
        print(f"[BiEncoder] 기본 모델 다운로드 중... (Run ID: {DEFAULT_RUN_ID})")
        
        # WandB API로 run 정보 가져오기
        api = Api()
        try:
            run = api.run(f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}/{DEFAULT_RUN_ID}")
        except Exception as e:
            raise ValueError(
                f"WandB run을 찾을 수 없습니다: {DEFAULT_RUN_ID}\n"
                f"프로젝트: {DEFAULT_WANDB_PROJECT}, 엔티티: {DEFAULT_WANDB_ENTITY}\n"
                f"에러: {e}"
            )
        
        # 모델 artifact 다운로드
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir = cache_dir / DEFAULT_RUN_ID
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # download_best_model.py의 함수 재사용
        import sys
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
            try:
                from download_best_model import download_model_artifact
                artifact_path = download_model_artifact(run, model_dir, "model")
                if artifact_path:
                    # artifact_path가 bi_encoder 디렉토리인지 확인
                    if (artifact_path / "bi_encoder").exists():
                        model_dir = artifact_path / "bi_encoder"
                    elif (artifact_path / "adapter_config.json").exists():
                        model_dir = artifact_path
                    else:
                        # artifact_path 내부에 bi_encoder가 있는지 확인
                        for subdir in artifact_path.iterdir():
                            if subdir.is_dir() and (subdir / "adapter_config.json").exists():
                                model_dir = subdir
                                break
                        else:
                            model_dir = artifact_path
                else:
                    raise RuntimeError("모델 다운로드에 실패했습니다.")
            except ImportError:
                # download_best_model을 import할 수 없으면 직접 다운로드
                artifacts = run.logged_artifacts()
                model_artifact = None
                for artifact in artifacts:
                    if "model" in artifact.name.lower():
                        model_artifact = artifact
                        break
                
                if not model_artifact:
                    raise ValueError(f"Run {DEFAULT_RUN_ID}에 모델 artifact가 없습니다.")
                
                artifact_path = model_artifact.download(root=str(model_dir))
                artifact_path = Path(artifact_path)
                # artifact_path 내부에 bi_encoder가 있는지 확인
                if (artifact_path / "bi_encoder").exists():
                    model_dir = artifact_path / "bi_encoder"
                elif (artifact_path / "adapter_config.json").exists():
                    model_dir = artifact_path
                else:
                    model_dir = artifact_path
        else:
            # scripts 디렉토리가 없으면 직접 다운로드
            artifacts = run.logged_artifacts()
            model_artifact = None
            for artifact in artifacts:
                if "model" in artifact.name.lower():
                    model_artifact = artifact
                    break
            
            if not model_artifact:
                raise ValueError(f"Run {DEFAULT_RUN_ID}에 모델 artifact가 없습니다.")
            
            artifact_path = model_artifact.download(root=str(model_dir))
            artifact_path = Path(artifact_path)
            # artifact_path 내부에 bi_encoder가 있는지 확인
            if (artifact_path / "bi_encoder").exists():
                model_dir = artifact_path / "bi_encoder"
            elif (artifact_path / "adapter_config.json").exists():
                model_dir = artifact_path
            else:
                model_dir = artifact_path
        
        print(f"[BiEncoder] ✅ 기본 모델 다운로드 완료: {model_dir}")
        return str(model_dir)

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
