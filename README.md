# 🏛️ LexDPR  
**구조화되고 계층적인 법령 및 규범 문서를 위한 Dense Passage Retrieval 모델**

LexDPR은 **법령, 규정, 비조치의견서 등과 같은 구조화된 문서**를 대상으로 하는 **Dense Passage Retrieval (DPR)** 모델입니다.  
조·항·호 단위의 계층적 구조를 가진 문서를 효율적으로 인덱싱하고, 의미적 일관성을 유지하며 검색 성능을 향상시키는 것을 목표로 합니다.

**목표:**
- 조·항·호 단위의 계층적 구조 문서 효율적 인덱싱
- 의미적 일관성 유지
- 검색 성능 향상

---

## 📘 프로젝트 개요

**기존 상용 임베딩 모델(OpenAI, Cohere, Sentence-Transformers 등)의 문제:**

- **계층 구조**가 깊은 문서(조/항/호 등)에 대한 표현 부족  
- **법령 문맥 의존성**이 높은 구문 처리의 불안정성  
- **의미적으로 연결된 문장 간 거리 문제**로 인한 검색 정확도 저하  

**LexDPR의 특징:**

- 법령 문서 구조에 최적화된 Dense Passage Retrieval 파이프라인
- RAG 시스템의 중간 검색기(retriever) 역할 수행

---

## ⚙️ 아키텍처 개요

LexDPR은 구조 인식형(Dense Passage Retrieval with structure-awareness) **듀얼 인코더(Dual Encoder)** 모델입니다.

```
Query Encoder (Sentence-BERT / Legal-BERT)
     │
     ▼
Query Vector
     │
     ├──> Passage Encoder (조문/항 단위)
     │         └─ 구조적 단위별 컨텍스트 윈도우 처리
     │
     ▼
Similarity Scoring (dot product / cosine)
     │
  Top-k 구조 단위 검색 결과 출력
```

LexDPR은 RAG 파이프라인의 생성기(generator)와 독립적으로 동작하며, **retriever 계층에만 집중**합니다.

**특징:**
- 생성기(generator)와 독립적 동작
- Retriever 계층 전용

---

## 🧩 프로젝트 구조

```
📁 LexDPR/
 ├── lex_dpr/                   # 패키지 메인 코드
 │    ├── models/               # 모델 관련 코드
 │    │   ├── encoders.py       # BiEncoder 클래스
 │    │   ├── templates.py      # 템플릿 모드
 │    │   └── config.py         # 기본 모델 설정
 │    ├── cli/                  # CLI 명령어 구현
 │    │   ├── embed.py          # 임베딩 생성 명령어
 │    │   └── ...
 │    └── ...
 │
 ├── docs/                      # 문서
 │    ├── TRAINING.md           # 모델 학습 가이드
 │    └── ...
 │
 ├── README.md
 └── pyproject.toml             # 패키지 설정
```

---

## 빠른 시작

### 1. 설치

```bash
# 기본 설치
pip install .

# 개발 모드 설치 (코드 수정 시 즉시 반영)
pip install -e .

# 설치 확인
python -c "from lex_dpr import BiEncoder; print('✅ 설치 성공')"
```

### 2. 기본 사용법

패키지 설치 후 바로 사용할 수 있습니다. 내장된 PEFT 모델이 자동으로 로드됩니다.

```python
from lex_dpr import BiEncoder
import numpy as np

# 기본 모델 사용 (내장된 PEFT 어댑터 자동 로드)
# base 모델은 HuggingFace에서 자동 다운로드됩니다
encoder = BiEncoder()

# 질의 임베딩 생성
queries = [
    "법인세 신고 기한은 언제인가요?",
    "근로기준법상 최저임금은 어떻게 결정되나요?"
]
query_embeddings = encoder.encode_queries(queries)

# 패시지 임베딩 생성
passages = [
    "법인세는 사업연도 종료일로부터 3개월 이내에 신고하여야 한다.",
    "최저임금은 근로자의 생계비, 유사직종의 임금 및 노동생산성을 고려하여 결정한다."
]
passage_embeddings = encoder.encode_passages(passages)

# 유사도 계산 (질의-패시지 매칭)
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embeddings, passage_embeddings)

# 가장 유사한 패시지 찾기
for i, query in enumerate(queries):
    best_match_idx = np.argmax(similarities[i])
    print(f"질의: {query}")
    print(f"매칭된 패시지: {passages[best_match_idx]}")
    print(f"유사도: {similarities[i][best_match_idx]:.4f}\n")
```

**출력 예시:**
```
[BiEncoder] 패키지에 포함된 PEFT 모델 사용: .../lex_dpr/models/default_model
[BiEncoder] Loading base model: intfloat/multilingual-e5-small
[BiEncoder] PEFT adapter loaded from .../lex_dpr/models/default_model
[BiEncoder] 학습 시 사용된 max_len(384)을 자동으로 적용합니다.

질의: 법인세 신고 기한은 언제인가요?
매칭된 패시지: 법인세는 사업연도 종료일로부터 3개월 이내에 신고하여야 한다.
유사도: 0.8523

질의: 근로기준법상 최저임금은 어떻게 결정되나요?
매칭된 패시지: 최저임금은 근로자의 생계비, 유사직종의 임금 및 노동생산성을 고려하여 결정한다.
유사도: 0.9145
```

---

## 사용 예시

### 1. 설치

```bash
# 기본 설치
pip install .

# 개발 모드 설치 (코드 수정 시 즉시 반영)
pip install -e .

# 설치 확인
python -c "from lex_dpr import BiEncoder, TemplateMode; print('✅ 설치 성공')"
```

### 2. 임베딩 생성

#### Python API

**기본 사용 (권장):**
```python
from lex_dpr import BiEncoder

# 기본 모델 사용 (내장된 PEFT 어댑터 자동 로드)
encoder = BiEncoder()

# 질의 임베딩 생성
queries = ["법률 질의 텍스트 1", "법률 질의 텍스트 2"]
query_embeddings = encoder.encode_queries(queries, batch_size=64)

# 패시지 임베딩 생성
passages = ["법률 문서 패시지 1", "법률 문서 패시지 2"]
passage_embeddings = encoder.encode_passages(passages, batch_size=64)

# 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embeddings, passage_embeddings)
```

**특정 모델 경로 지정:**
```python
from lex_dpr import BiEncoder, TemplateMode

encoder = BiEncoder(
    "checkpoint/lexdpr/bi_encoder",
    template=TemplateMode.BGE,
    normalize=True,
    max_seq_length=512,
)
```

#### CLI 방식

```bash
# 질의 임베딩 생성
lex-dpr embed \
  --model checkpoint/lexdpr/bi_encoder \
  --input data/queries.jsonl \
  --outdir embeddings \
  --prefix queries \
  --type query \
  --batch-size 64 \
  --template bge

# 패시지 임베딩 생성
lex-dpr embed \
  --model checkpoint/lexdpr/bi_encoder \
  --input data/processed/law_passages.jsonl \
  --outdir embeddings \
  --prefix passages \
  --type passage \
  --batch-size 64 \
  --template bge
```

**출력 파일 형식:**
- NPZ 형식 (기본): `{prefix}.npz` (ids와 embeddings 포함)
- NPY 형식: `{prefix}_ids.npy`, `{prefix}_embeds.npy`

#### 스크립트 방식

```bash
# Python 스크립트 직접 실행
python scripts/embed_corpus.py \
  --model checkpoint/lexdpr/bi_encoder \
  --input data/queries.jsonl \
  --outdir embeddings \
  --prefix queries \
  --type query
```

### 3. max_seq_length vs embedding dimension

두 가지 다른 개념을 구분해야 합니다:

**1. max_seq_length (max_len)**: 입력 텍스트의 최대 토큰 수
- 모델이 한 번에 처리할 수 있는 입력 텍스트의 최대 길이
- 예: `max_seq_length=128` → 최대 128개 토큰까지 처리
- 확인: `encoder.get_max_seq_length()` → 128

**2. embedding dimension**: 출력 벡터의 차원 수
- 각 텍스트가 변환되는 벡터의 크기
- 예: `embedding_dim=384` → 384차원 벡터로 변환
- 모델에 따라 결정: multilingual-e5-small은 384차원
- 확인: `encoder.get_embedding_dimension()` → 384

**예시:**

```python
from lex_dpr import BiEncoder

encoder = BiEncoder("checkpoint/lexdpr/bi_encoder")
print(f"Max seq length: {encoder.get_max_seq_length()}")      # 128 (입력 길이 제한)
print(f"Embedding dimension: {encoder.get_embedding_dimension()}")  # 384 (출력 벡터 크기)

query_emb = encoder.encode_queries(["질의"])
print(f"Query shape: {query_emb.shape}")  # (1, 384)
# - 첫 번째 차원(1): 질의 개수
# - 두 번째 차원(384): 임베딩 차원 (벡터 크기)
# - max_seq_length(128)는 입력 텍스트가 128 토큰을 초과하면 잘림
```

### 4. 고급 사용법

#### Query와 Passage에 서로 다른 최대 길이 설정

```python
from lex_dpr import BiEncoder, TemplateMode

encoder = BiEncoder(
    "checkpoint/lexdpr/bi_encoder",
    template=TemplateMode.BGE,
    normalize=True,
    query_max_seq_length=128,  # 질의는 짧게
    passage_max_seq_length=512,  # 패시지는 길게
)
```

#### PEFT 어댑터 사용

```python
encoder = BiEncoder(
    "base_model_name",
    peft_adapter_path="checkpoint/lexdpr/bi_encoder",  # PEFT 어댑터 경로
)
```

#### 모델 정보 확인

```python
# 모델의 현재 max_seq_length 확인
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

---

## 📚 추가 문서

- **[모델 학습 가이드](docs/TRAINING.md)**: 데이터 준비, 모델 학습, 하이퍼파라미터 튜닝 등 학습 관련 가이드
- **[Git LFS 사용 가이드](docs/GIT_LFS_GUIDE.md)**: 모델 체크포인트와 대용량 파일을 Git LFS로 관리하는 방법

---

## 📄 Model Ablation Study

### 🧩 한국어 전용 리트리버 후보 (Bi-Encoder)

| 항목               | **KoSimCSE-roberta-multitask**                               | **KLUE-RoBERTa-base-bi**                               | **KoE5-small**                                             |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------- |
| **모델 타입**      | Sentence-BERT형 Bi-Encoder                                   | RoBERTa 기반 Dual Encoder (LexDPR에 적합)              | E5 계열 Encoder (한국어 전용)                              |
| **파라미터 수**    | ≈110 M                                                       | ≈125 M                                                 | ≈80 M                                                      |
| **학습 방식**      | SimCSE + Multitask (STS, NLI)                                | KLUE 태스크 기반 pretrain + contrastive fine-tune 가능 | Instruction-style (E5 objective: “query: …”, “passage: …”) |
| **언어 범위**      | 한국어 only                                                  | 한국어 only                                            | 한국어 only (OpenKoE5)                                     |
| **임베딩 품질**    | 일상 문장, QA, 짧은 질의에 강함                              | 문장 길이 중간~긴 법령 문체에 안정적                   | E5 objective로 문맥매칭 성능 우수                          |
| **장점**           | • 경량·빠름<br>• GPU 메모리↓                                 | • KLUE 표준 문체 적합<br>• 파인튜닝 용이               | • 최신 E5 프레임워크 구조<br>• cosine 정규화 안정          |
| **약점**           | • 전문 법령 도메인 약함                                      | • Pretrained 모델 공개 적음                            | • 상대적으로 적은 공개 체크포인트                          |
| **적합 시나리오**  | 빠른 프로토타입, 저자원 환경                                 | 중대형 리트리버 구축 (LexDPR 구조에 자연 적합)         | 도메인 확장 실험, E5형식 파이프라인 일치                   |
| **LexDPR 적용 시** | 그대로 `--model BM-K/KoSimCSE-roberta-multitask`로 학습 가능 | KLUE-RoBERTa-bi로 학습 시 조문/항 단위 안정            | `KoE5-small`은 E5 prefix(`query:`, `passage:`) 유지 필요   |


### 🧩 한국어 전용 리랭커 후보 (Cross-Encoder)

| 항목               | **KLUE-RoBERTa-large (Cross-Encoder)**                             | **KR-ELECTRA-discriminator**                          |
| ------------------ | ------------------------------------------------------------------ | ----------------------------------------------------- |
| **모델 타입**      | Transformer Cross-Encoder                                          | ELECTRA 기반 Cross-Encoder                            |
| **파라미터 수**    | ≈355 M                                                             | ≈110 M                                                |
| **학습 방식**      | (q,p) 쌍 입력 → relevance classification                           | (q,p) 쌍 입력 → relevance classification              |
| **언어 범위**      | 한국어 only                                                        | 한국어 only                                           |
| **특징**           | RoBERTa 기반 문맥 이해 강력, 긴 문장에도 안정                      | 연산 가볍고 학습 속도 빠름                            |
| **장점**           | • 높은 정밀도(Top-10 rerank 성능) <br>• 문장 길이 긴 법령에도 적합 | • 경량, 빠른 재랭크 • GPU 자원 절약                   |
| **약점**           | • 지연시간·메모리↑                                                 | • 세밀한 논리 관계 파악 한계                          |
| **적합 시나리오**  | 오프라인 인덱스 재랭크, 중요 질의 정밀 평가                        | 실시간 QA, 저자원 환경                                |
| **LexDPR 적용 시** | `bge-reranker-large` 대체로 사용 가능 (입력: [CLS] q [SEP] p)      | 소규모 실시간 재랭크기 or lightweight reranker로 적합 |


### 🔍 조합

| 목적                               | 리트리버            | 리랭커                 | 코멘트                                             |
| ---------------------------------- | ------------------- | ---------------------- | -------------------------------------------------- |
| **정밀도 중심 (법령 검색 품질 ↑)** | **KLUE-RoBERTa-bi** | **KLUE-RoBERTa-large** | LexDPR의 기본 DPR 구조와 완벽 호환, 법령 문체 안정 |
| **경량·빠른 검색**                 | **KoSimCSE**        | **KR-ELECTRA**         | 실시간 질의 응답용, 빠른 inference                 |
| **확장형(Instruction 기반)**       | **KoE5-small**      | **KLUE-RoBERTa-large** | E5 포맷 유지로 multilingual E5-mistral 전환 용이   |


**! 2번 (KoSimCSE+KR-ELECTRA)** 이후 **1번 (KLUE-RoBERTa-bi+KLUE-RoBERTa-large)**


---

## 📚 추가 문서

- **[모델 학습 가이드](docs/TRAINING.md)**: 데이터 준비, 모델 학습, 하이퍼파라미터 튜닝 등 학습 관련 가이드
- **[Git LFS 사용 가이드](docs/GIT_LFS_GUIDE.md)**: 모델 체크포인트와 대용량 파일을 Git LFS로 관리하는 방법
