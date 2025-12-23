# 모델 다운로드 후 다음 단계

## ✅ 완료된 작업

1. ✅ 모델 다운로드: `lex_dpr/models/default_model`
2. ✅ `config.py` 설정: `DEFAULT_MODEL_PATH = "models/default_model"`, `DEFAULT_MAX_LEN = 128`
3. ✅ `pyproject.toml` 설정: 모델 파일 포함 설정 추가

## 📋 다음 단계

### 1. 모델 파일 확인

다운로드된 모델이 올바르게 있는지 확인:

```bash
# Windows CMD
dir lex_dpr\models\default_model

# 또는 Python으로 확인
python -c "from pathlib import Path; print(list(Path('lex_dpr/models/default_model').iterdir()))"
```

필수 파일:
- ✅ `adapter_config.json` (PEFT 어댑터 설정)
- ✅ `adapter_model.safetensors` (PEFT 어댑터 가중치)
- ✅ 기타 토크나이저 파일들

### 2. 패키지 빌드

```bash
poetry build
```

빌드가 성공하면 `dist/` 디렉토리에 패키지 파일이 생성됩니다:
- `lexdpr-0.1.0.tar.gz` (소스 배포판)
- `lexdpr-0.1.0-py3-none-any.whl` (휠 파일)

### 3. 로컬 테스트 (선택사항)

빌드된 패키지를 로컬에서 테스트:

```bash
# 빌드된 패키지 설치
pip install dist/lexdpr-*.whl --force-reinstall

# 테스트
python -c "from lex_dpr import BiEncoder; encoder = BiEncoder(); print('✅ 모델 로드 성공')"
```

또는 더 자세한 테스트:

```python
from lex_dpr import BiEncoder

# 기본 모델 사용 (패키지에 포함된 모델)
encoder = BiEncoder()

# 임베딩 생성 테스트
query_emb = encoder.encode_queries(["테스트 질의"])
passage_emb = encoder.encode_passages(["테스트 패시지"])

print(f"Query embedding shape: {query_emb.shape}")
print(f"Passage embedding shape: {passage_emb.shape}")
print(f"Max seq length: {encoder.get_max_seq_length()}")
print(f"Embedding dimension: {encoder.get_embedding_dimension()}")
```

### 4. 모델 크기 확인

```bash
# Windows
dir /s lex_dpr\models\default_model

# 또는 PowerShell
Get-ChildItem -Path lex_dpr\models\default_model -Recurse | Measure-Object -Property Length -Sum
```

PEFT 어댑터만 포함되므로 보통 수 MB ~ 수십 MB입니다.

### 5. 패키지 배포 (선택사항)

```bash
# PyPI에 배포 (테스트 서버)
poetry publish --repository testpypi

# PyPI에 배포 (실제 서버)
poetry publish
```

## ⚠️ 주의사항

1. **Base 모델**: PEFT 어댑터만 포함되므로, 사용자는 Base 모델(`ko-simcse`)을 HuggingFace에서 자동으로 다운로드합니다.

2. **모델 크기**: 패키지에 포함된 모델은 PEFT 어댑터만이므로 작습니다. Base 모델은 별도로 다운로드됩니다.

3. **Git에 포함**: 모델 파일을 Git에 포함시킬지 결정해야 합니다:
   - 포함: `.gitignore`에서 제외
   - 제외: `.gitignore`에 `lex_dpr/models/default_model/` 추가

## 🔍 문제 해결

### 모델 로드 실패 시

```python
from lex_dpr import BiEncoder

# 명시적으로 경로 지정하여 테스트
encoder = BiEncoder("lex_dpr/models/default_model")
```

### 패키지 빌드 실패 시

1. `pyproject.toml`의 `include` 설정 확인
2. 모델 파일이 올바른 경로에 있는지 확인
3. `MANIFEST.in` 파일 생성 (필요한 경우)

