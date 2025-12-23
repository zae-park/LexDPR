# LexDPR 패키지 설치 및 사용 가이드

## 📦 패키지 설치

### Poetry를 사용한 설치 (권장)

```bash
# 저장소 클론
git clone <repository-url>
cd LexDPR

# 의존성 설치
poetry install

# 또는 개발 모드로 설치
poetry install --with dev
```

### pip를 사용한 설치

```bash
# 저장소 클론
git clone <repository-url>
cd LexDPR

# 패키지 설치
pip install -e .
```

## ✅ 설치 확인

패키지가 제대로 설치되었는지 확인:

```python
# Python 인터프리터에서 테스트
python -c "from lex_dpr import BiEncoder, TemplateMode; print('✅ 설치 성공')"
```

또는 테스트 스크립트 실행:

```bash
python test_embedding_import.py
```

## 🔍 필수 의존성 확인

LexDPR 임베딩 기능을 사용하기 위해 다음 패키지가 필요합니다:

- `sentence-transformers` (>=3.0.1,<4.0.0)
- `transformers` (>=4.38.0,<4.44.0)
- `torch` (>=2.4,<2.6)
- `numpy` (>=1.26.0,<3.0.0)
- `peft` (>=0.10.0,<0.11.0) - PEFT 어댑터 사용 시

의존성 확인:

```python
import sentence_transformers
import transformers
import torch
import numpy as np
from peft import PeftModel  # 선택적

print(f"sentence-transformers: {sentence_transformers.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"torch: {torch.__version__}")
print(f"numpy: {np.__version__}")
```

## 🚀 기본 사용법

### 1. Python API 사용

```python
from lex_dpr import BiEncoder, TemplateMode

# 모델 로드
encoder = BiEncoder(
    "checkpoint/lexdpr/bi_encoder",  # 모델 경로 또는 HuggingFace 모델 이름
    template=TemplateMode.BGE,        # 또는 TemplateMode.NONE
    normalize=True,                   # 임베딩 정규화 (기본값: True)
    max_seq_length=512,              # 최대 시퀀스 길이
)

# 질의 임베딩 생성
queries = [
    "법률 질의 텍스트 1",
    "법률 질의 텍스트 2",
]
query_embeddings = encoder.encode_queries(queries, batch_size=64)
print(f"Query embeddings shape: {query_embeddings.shape}")  # (2, embedding_dim)

# 패시지 임베딩 생성
passages = [
    "법률 문서 패시지 1",
    "법률 문서 패시지 2",
]
passage_embeddings = encoder.encode_passages(passages, batch_size=64)
print(f"Passage embeddings shape: {passage_embeddings.shape}")  # (2, embedding_dim)
```

### 2. HuggingFace 모델 사용

```python
from lex_dpr import BiEncoder

# HuggingFace Hub에서 모델 로드
encoder = BiEncoder("jhgan/ko-sroberta-multitask")

# 또는 학습된 모델 사용
encoder = BiEncoder("checkpoint/lexdpr/bi_encoder")
```

### 3. PEFT 어댑터 사용

```python
from lex_dpr import BiEncoder

# PEFT 어댑터가 포함된 체크포인트 (자동 감지)
encoder = BiEncoder("checkpoint/lexdpr/bi_encoder")

# 또는 별도로 지정
encoder = BiEncoder(
    "base_model_name",
    peft_adapter_path="checkpoint/lexdpr/bi_encoder",
)
```

### 4. Query/Passage별 다른 최대 길이 설정

```python
from lex_dpr import BiEncoder, TemplateMode

encoder = BiEncoder(
    "checkpoint/lexdpr/bi_encoder",
    template=TemplateMode.BGE,
    normalize=True,
    query_max_seq_length=128,    # 질의는 짧게
    passage_max_seq_length=512,  # 패시지는 길게
)
```

## ⚠️ 문제 해결

### ImportError 발생 시

1. **패키지가 설치되지 않은 경우**:
   ```bash
   poetry install
   # 또는
   pip install -e .
   ```

2. **Python 경로 문제**:
   ```bash
   # 현재 디렉토리 확인
   pwd
   # Python 경로 확인
   python -c "import sys; print(sys.path)"
   ```

3. **의존성 누락**:
   ```bash
   # 의존성 재설치
   poetry install --no-cache
   ```

### 모델 로드 실패 시

1. **모델 경로 확인**:
   ```python
   from pathlib import Path
   model_path = Path("checkpoint/lexdpr/bi_encoder")
   print(f"모델 경로 존재: {model_path.exists()}")
   print(f"필수 파일 확인: {(model_path / 'config.json').exists()}")
   ```

2. **HuggingFace 모델 다운로드 확인**:
   - 인터넷 연결 확인
   - HuggingFace Hub 접근 권한 확인
   - 모델 이름 정확성 확인

3. **PEFT 어댑터 문제**:
   ```python
   # adapter_config.json 확인
   from pathlib import Path
   adapter_path = Path("checkpoint/lexdpr/bi_encoder")
   if (adapter_path / "adapter_config.json").exists():
       import json
       with open(adapter_path / "adapter_config.json") as f:
           config = json.load(f)
       print(f"Base model: {config.get('base_model_name_or_path')}")
   ```

### 메모리 부족 시

```python
# 배치 크기 줄이기
encoder = BiEncoder("model_path")
embeddings = encoder.encode_queries(queries, batch_size=8)  # 기본값 64에서 줄임

# CPU 사용 (GPU 메모리 부족 시)
import torch
encoder = BiEncoder("model_path")
encoder.model.to("cpu")
```

## 📥 학습된 모델 다운로드

WandB에 업로드된 학습된 모델을 다운로드할 수 있습니다:

### CLI 사용

```bash
# 기본 사용 (최고 성능 모델 자동 다운로드)
poetry run lex-dpr download-model

# 특정 Sweep ID 지정
poetry run lex-dpr download-model --sweep-id <sweep-id>

# 메트릭 및 출력 경로 지정
poetry run lex-dpr download-model \
  --metric eval/ndcg@10 \
  --output-dir checkpoint/my_model \
  --project lexdpr \
  --entity zae-park
```

### Python 스크립트 사용

```bash
python scripts/download_best_model.py \
  --sweep-id <sweep-id> \
  --metric eval/recall_at_10 \
  --output-dir checkpoint/best_model
```

## 📏 임베딩 차원 및 시퀀스 길이

### ⚠️ 중요: max_seq_length vs embedding dimension 구분

두 가지 다른 개념을 혼동하지 마세요:

#### 1. max_seq_length (max_len): 입력 텍스트의 최대 토큰 수

- **의미**: 모델이 한 번에 처리할 수 있는 입력 텍스트의 최대 길이
- **예**: `max_seq_length=128` → 최대 128개 토큰까지 처리
- **학습 시 설정**: `configs/sweep.yaml`에서 `model.max_len: 128`
- **확인**: `encoder.get_max_seq_length()` → `128`
- **영향**: 입력 텍스트가 이 길이를 초과하면 자동으로 잘림(truncation)

#### 2. embedding dimension: 출력 벡터의 차원 수

- **의미**: 각 텍스트가 변환되는 벡터의 크기
- **예**: `embedding_dim=384` → 384차원 벡터로 변환
- **모델에 따라 결정**: `multilingual-e5-small`은 384차원, `ko-simcse`는 768차원
- **확인**: `encoder.get_embedding_dimension()` → `384`
- **영향**: 벡터 검색, 유사도 계산 등에 사용

### 실제 사용 예시

```python
from lex_dpr import BiEncoder

encoder = BiEncoder("checkpoint/lexdpr/bi_encoder")

# 1. max_seq_length 확인 (입력 길이 제한)
max_seq_len = encoder.get_max_seq_length()
print(f"Max seq length: {max_seq_len}")  # 128 (입력 텍스트 최대 토큰 수)

# 2. embedding dimension 확인 (출력 벡터 크기)
embedding_dim = encoder.get_embedding_dimension()
print(f"Embedding dimension: {embedding_dim}")  # 384 (출력 벡터 차원)

# 3. 실제 임베딩 생성
query_emb = encoder.encode_queries(["질의 텍스트"])
passage_emb = encoder.encode_passages(["패시지 텍스트"])

print(f"Query embedding shape: {query_emb.shape}")    # (1, 384)
print(f"Passage embedding shape: {passage_emb.shape}")  # (1, 384)

# 설명:
# - 첫 번째 차원(1): 텍스트 개수
# - 두 번째 차원(384): 임베딩 차원 (벡터 크기)
# - max_seq_length(128)는 입력 텍스트가 128 토큰을 초과하면 잘림
```

### 학습 시 사용된 시퀀스 길이

**학습 시 사용된 시퀀스 길이**: `max_len: 128` (configs/model.yaml, configs/sweep.yaml)

- 질의와 패시지는 **같은 모델**을 사용하지만, **다른 템플릿**을 적용합니다:
  - 질의: `"Represent this sentence for searching relevant passages: {q}"`
  - 패시지: `"Represent this sentence for retrieving relevant passages: {p}"`
- 시퀀스 길이는 학습 시 설정한 값과 동일하게 사용해야 합니다:
  ```python
  encoder = BiEncoder(
      "checkpoint/lexdpr/bi_encoder",
      max_seq_length=128,  # 학습 시 사용한 길이와 동일하게 설정
  )
  ```

### 학습 설정 확인

학습 시 사용된 설정은 다음 파일에서 확인할 수 있습니다:

- `configs/model.yaml`: `max_len: 128`
- `configs/sweep.yaml`: `model.max_len: 128` (sweep 사용 시)

임베딩 생성 시 동일한 설정을 사용해야 합니다:

```python
# 학습 시 max_len=128로 학습했다면
encoder = BiEncoder(
    "checkpoint/lexdpr/bi_encoder",
    max_seq_length=128,  # 학습 시와 동일하게
    template=TemplateMode.BGE,  # 학습 시와 동일하게
)
```

## 📦 모델 저장 형식 및 크기

### WandB에 저장되는 모델

**WandB에 저장되는 모델은 PEFT (LoRA) 어댑터만 저장됩니다.**

- **저장되는 것**: PEFT 어댑터만 (adapter_config.json, adapter_model.safetensors 등)
- **저장되지 않는 것**: Base 모델 (HuggingFace에서 자동 다운로드)
- **크기**: 매우 작음 (수 MB ~ 수십 MB)
  - 예: LoRA r=8, alpha=16인 경우 약 5-10MB
  - Base 모델 (ko-simcse 등)은 수백 MB ~ 수 GB

### 패키지에 포함 가능성

PEFT 어댑터만 저장되므로 **패키지에 포함 가능한 크기**입니다. 하지만:

- ✅ **장점**: 어댑터만 포함하면 패키지 크기가 작음
- ⚠️ **주의사항**: 
  - Base 모델은 여전히 HuggingFace에서 다운로드 필요
  - 사용자가 인터넷 연결이 필요함
  - Base 모델 크기가 크므로 패키지에 포함하기 어려움

**권장 사항**: 
- 어댑터만 패키지에 포함하고, base 모델은 HuggingFace에서 자동 다운로드
- 또는 모델을 별도로 배포하고 패키지는 다운로드 스크립트만 제공

### 다운로드한 모델에서 학습 설정 확인

다운로드한 모델에서 학습 시 사용된 설정을 확인할 수 있습니다:

```python
from lex_dpr import BiEncoder

# 모델 로드
encoder = BiEncoder("checkpoint/lexdpr/bi_encoder")

# 현재 max_seq_length 확인
current_max_len = encoder.get_max_seq_length()
print(f"현재 모델 max_seq_length: {current_max_len}")

# PEFT 어댑터 설정 확인 (PEFT 모델인 경우)
training_config = encoder.get_training_config()
if training_config:
    print(f"Base 모델: {training_config.get('base_model_name_or_path')}")
    print(f"LoRA r: {training_config.get('r')}")
    print(f"LoRA alpha: {training_config.get('lora_alpha')}")
    print(f"Target modules: {training_config.get('target_modules')}")
```

**Sweep으로 학습한 경우**:
- Sweep은 다양한 `max_len` 값을 시도할 수 있습니다
- 다운로드한 모델의 `get_max_seq_length()`로 실제 사용된 길이 확인 가능
- 또는 WandB run의 config에서 `model.max_len` 값 확인

## 📝 추가 리소스

- [README.md](../README.md): 전체 프로젝트 문서
- [임베딩 사용 가이드](../README.md#4-2-임베딩-생성-및-사용): 상세한 사용 예시
- [CLI 사용법](../README.md): 명령줄 인터페이스 사용법

