# 패키지 배포 가이드

## 기본 모델 설정

패키지 배포 시 특정 학습된 모델을 기본으로 사용하도록 설정할 수 있습니다.

**두 가지 방법이 있습니다:**

### 방법 1: 패키지에 모델 포함 (권장) ⭐

PEFT 어댑터만 포함하므로 패키지 크기가 작습니다 (수 MB).

#### 1-1. 모델 준비

```bash
# 학습된 모델 다운로드
poetry run lex-dpr download-model --sweep-id <sweep-id> --output-dir models/default_model

# 또는 직접 모델 경로 지정
# 모델은 checkpoint/lexdpr/bi_encoder 같은 경로에 있어야 함
```

#### 1-2. 모델을 패키지에 포함

```bash
# 모델을 패키지 디렉토리로 복사
cp -r checkpoint/lexdpr/bi_encoder lex_dpr/models/default_model
```

#### 1-3. config.py 설정

`lex_dpr/models/config.py` 파일을 열고 설정:

```python
# lex_dpr/models/config.py

# 방법 1: 패키지에 포함된 모델 사용 (권장)
DEFAULT_MODEL_PATH: str | None = "models/default_model"  # 패키지 내부 경로
DEFAULT_MAX_LEN: int | None = 128  # 학습 시 사용된 max_seq_length

# 방법 2는 사용하지 않으므로 None으로 설정
DEFAULT_RUN_ID: str | None = None
```

#### 1-4. pyproject.toml에 모델 파일 포함

`pyproject.toml`에 다음을 추가:

```toml
[tool.poetry]
# ... 기존 설정 ...

# 모델 파일 포함
include = [
    { path = "lex_dpr/models/default_model", format = "sdist" },
]
```

또는 `MANIFEST.in` 파일 생성:

```
include lex_dpr/models/default_model/**
```

#### 1-5. 패키지 빌드 및 배포

```bash
# 패키지 빌드
poetry build

# 또는 PyPI에 배포
poetry publish
```

### 방법 2: WandB에서 자동 다운로드

패키지에 모델을 포함하지 않고, 사용자가 처음 사용할 때 자동으로 다운로드합니다.

#### 2-1. config.py 설정

`lex_dpr/models/config.py` 파일을 열고 설정:

```python
# lex_dpr/models/config.py

# 방법 1은 사용하지 않으므로 None으로 설정
DEFAULT_MODEL_PATH: str | None = None

# 방법 2: WandB에서 자동 다운로드
DEFAULT_RUN_ID: str | None = "your_run_id_here"  # 예: "abc123xyz"
DEFAULT_MAX_LEN: int | None = 128  # 학습 시 사용된 max_seq_length

# WandB 설정
DEFAULT_WANDB_PROJECT: str = "lexdpr"
DEFAULT_WANDB_ENTITY: str = "zae-park"
```

#### 2-2. 패키지 빌드 및 배포

```bash
# 패키지 빌드
poetry build

# 또는 PyPI에 배포
poetry publish
```

### 3. 사용자 사용 방법

사용자는 run ID를 몰라도 자동으로 기본 모델을 사용할 수 있습니다:

```python
from lex_dpr import BiEncoder

# 기본 모델 자동 사용
encoder = BiEncoder()  # 또는 BiEncoder("default")

# 방법 1 (패키지 포함)인 경우:
# - 패키지에 포함된 모델을 즉시 사용
# - 학습 시 사용된 max_len 자동 적용

# 방법 2 (WandB 다운로드)인 경우:
# - 첫 실행 시: WandB에서 지정된 run ID의 모델을 자동 다운로드
# - ~/.lexdpr/models/{run_id}/ 에 저장
# - 학습 시 사용된 max_len 자동 적용
# - 이후 실행 시: 캐시된 모델을 재사용 (다운로드 불필요)
```

### 4. 기본 모델 설정 정보

`lex_dpr/models/config.py`에서 다음 설정을 변경할 수 있습니다:

**방법 1 (패키지 포함) 설정:**
- `DEFAULT_MODEL_PATH`: 패키지에 포함된 모델 경로 (상대 경로)
- `DEFAULT_MAX_LEN`: 학습 시 사용된 max_seq_length

**방법 2 (WandB 다운로드) 설정:**
- `DEFAULT_RUN_ID`: 기본 WandB Run ID
- `DEFAULT_MAX_LEN`: 학습 시 사용된 max_seq_length
- `DEFAULT_WANDB_PROJECT`: WandB 프로젝트 이름 (기본값: "lexdpr")
- `DEFAULT_WANDB_ENTITY`: WandB 엔티티 이름 (기본값: "zae-park")
- `DEFAULT_MODEL_CACHE_DIR`: 모델 캐시 디렉토리 (기본값: "~/.lexdpr/models")
- `DEFAULT_METRIC`: 최고 성능 모델 선택 메트릭 (기본값: "eval/recall_at_10")
- `DEFAULT_GOAL`: 메트릭 최적화 목표 (기본값: "maximize")

### 5. 배포 전 체크리스트

- [ ] `DEFAULT_RUN_ID`가 올바르게 설정되었는지 확인
- [ ] 해당 run이 WandB에 존재하고 모델 artifact가 업로드되어 있는지 확인
- [ ] `training_config.json`이 모델 artifact에 포함되어 있는지 확인 (max_len 자동 적용을 위해)
- [ ] 패키지 버전 업데이트 (필요한 경우)

### 6. 버전별 다른 모델 사용

패키지 버전마다 다른 모델을 사용하려면:

1. 새 버전 배포 전에 `DEFAULT_RUN_ID`를 업데이트
2. 패키지 버전 업데이트 (`pyproject.toml`의 `version` 필드)
3. 빌드 및 배포

사용자는 패키지를 업그레이드하면 자동으로 새 모델을 사용하게 됩니다.

### 7. 사용자 환경 변수

사용자가 기본 모델 대신 다른 모델을 사용하려면:

```python
# 방법 1: 특정 모델 경로 지정
encoder = BiEncoder("path/to/custom/model")

# 방법 2: 다른 run ID 사용 (WandB에서 직접 다운로드)
# poetry run lex-dpr download-model --sweep-id <sweep-id>
encoder = BiEncoder("checkpoint/best_model/bi_encoder")
```

### 8. 주의사항

- 기본 모델을 사용하려면 사용자가 `WANDB_API_KEY` 환경 변수를 설정해야 합니다.
- 첫 실행 시 인터넷 연결이 필요합니다 (모델 다운로드).
- 모델은 `~/.lexdpr/models/{run_id}/`에 캐시되므로, 디스크 공간이 필요합니다.

