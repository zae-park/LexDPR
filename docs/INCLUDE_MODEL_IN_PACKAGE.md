# 패키지에 모델 포함하기

## WandB Artifact를 패키지에 포함하는 방법

### 1. Artifact 다운로드

```bash
# WandB API 키 설정 (필요한 경우)
export WANDB_API_KEY=your_api_key

# Artifact 다운로드
python scripts/download_artifact_for_package.py \
    --artifact artifacts/model/model_trim-sweep-12 \
    --output lex_dpr/models/default_model \
    --project lexdpr \
    --entity zae-park
```

### 2. config.py 업데이트

`lex_dpr/models/config.py` 파일을 열고 다음을 설정:

```python
# 방법 1: 패키지에 포함된 모델 사용
DEFAULT_MODEL_PATH: str | None = "models/default_model"  # 패키지 내부 경로
DEFAULT_MAX_LEN: int | None = 128  # 학습 시 사용된 max_seq_length (configs/model.yaml에서 확인)

# 방법 2는 사용하지 않으므로 None으로 설정
DEFAULT_RUN_ID: str | None = None
```

### 3. pyproject.toml에 모델 파일 포함

`pyproject.toml`에 다음을 추가:

```toml
[tool.poetry]
# ... 기존 설정 ...

# 모델 파일 포함 (MANIFEST.in 방식도 가능)
include = [
    { path = "lex_dpr/models/default_model", format = "sdist" },
]
```

또는 `MANIFEST.in` 파일 생성:

```
include lex_dpr/models/default_model/**
recursive-include lex_dpr/models/default_model *
```

### 4. 패키지 빌드 및 테스트

```bash
# 패키지 빌드
poetry build

# 로컬에서 테스트
pip install dist/lexdpr-*.whl

# Python에서 테스트
python -c "from lex_dpr import BiEncoder; encoder = BiEncoder(); print('✅ 모델 로드 성공')"
```

### 5. 모델 크기 확인

```bash
# 모델 디렉토리 크기 확인
du -sh lex_dpr/models/default_model

# PEFT 어댑터만 포함되므로 보통 수 MB ~ 수십 MB입니다.
```

### 주의사항

- PEFT 어댑터만 포함되므로 Base 모델은 HuggingFace에서 자동 다운로드됩니다.
- `training_config.json`이 모델에 포함되어 있으면 `max_len`이 자동으로 적용됩니다.
- 없으면 `config.py`의 `DEFAULT_MAX_LEN`을 설정해야 합니다.

