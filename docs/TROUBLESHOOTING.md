# 문제 해결 가이드

## SafetensorError: header too large

### 원인
- 다운로드 중 파일 손상
- 불완전한 다운로드
- 디스크 공간 부족

### 해결 방법

#### 1. 모델 다시 다운로드

```bash
# 기존 디렉토리 삭제
rm -rf lex_dpr/models/default_model

# 다시 다운로드
poetry run python scripts/download_artifact_for_package.py \
    --artifact artifacts/model/model_trim-sweep-12 \
    --output lex_dpr/models/default_model \
    --project lexdpr \
    --entity zae-park
```

#### 2. Windows에서 삭제 및 재다운로드

```cmd
# 기존 디렉토리 삭제
rmdir /s /q lex_dpr\models\default_model

# 다시 다운로드
poetry run python scripts/download_artifact_for_package.py --artifact artifacts/model/model_trim-sweep-12 --output lex_dpr/models/default_model --project lexdpr --entity zae-park
```

#### 3. 파일 검증

다운로드 후 파일이 올바른지 확인:

```python
from pathlib import Path

model_file = Path("lex_dpr/models/default_model/adapter_model.safetensors")
if model_file.exists():
    size_mb = model_file.stat().st_size / 1024 / 1024
    print(f"파일 크기: {size_mb:.2f} MB")
    # 일반적으로 PEFT 어댑터는 수 MB ~ 수십 MB입니다
    if size_mb < 0.1:  # 100KB 미만이면 손상 가능성
        print("⚠️ 파일이 너무 작습니다. 다시 다운로드하세요.")
```

#### 4. 수동으로 파일 확인

```bash
# 파일 목록 확인
ls -lh lex_dpr/models/default_model/

# 파일 크기 확인
du -sh lex_dpr/models/default_model/
```

필수 파일:
- `adapter_config.json` (설정 파일)
- `adapter_model.safetensors` (가중치 파일, 보통 수 MB ~ 수십 MB)
- 기타 토크나이저 파일들

### 예방 방법

1. **안정적인 네트워크 환경에서 다운로드**
2. **충분한 디스크 공간 확인**
3. **다운로드 중 중단하지 않기**

## 기타 문제

### 모델 로드 실패

```python
# 명시적으로 경로 지정
from lex_dpr import BiEncoder

encoder = BiEncoder("lex_dpr/models/default_model")
```

### max_seq_length 불일치 경고

학습 시 사용된 `max_seq_length`와 다른 값으로 설정하면 경고가 발생합니다.
`training_config.json`이 있으면 자동으로 올바른 값이 적용됩니다.

### WandB API 키 오류

Windows에서 환경 변수 설정:
```cmd
set WANDB_API_KEY=your_api_key
```

또는 WandB 로그인:
```bash
poetry run wandb login
```

