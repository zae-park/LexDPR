# 🏛️ LexDPR  

**법령, 규정, 비조치의견서 등과 같은 구조화된 문서**를 대상으로 **Dense Passage Retrieval (DPR)** 모델을 학습합니다.  
조문·항·호 단위의 계층적 구조를 가진 문서의 분할 및 가공과 의미적 일관성을 유지한 임베딩으로 검색 성능을 향상시킵니다.

---

## 📘 개요

**기존 상용 임베딩 모델(OpenAI, Cohere, Sentence-Transformers 등)의 문제:**

- **계층 구조**가 깊은 문서(조/항/호 등)에 대한 표현 부족  
- **법령 문맥 의존성**이 높은 구문 처리의 불안정성  
- **의미적으로 연결된 문장 간 거리 문제**로 인한 검색 정확도 저하  

**LexDPR:**

- 법령 문서 구조에 최적화된 Dense Passage Retrieval 파이프라인
- RAG 시스템의 중간 검색기(retriever) 역할 수행

---

LexDPR은 개념적으로 **듀얼 인코더(Dual Encoder)** 모델입니다.
즉, RAG 파이프라인의 생성기(generator)와 독립적으로 동작하며, **retriever 계층에만 집중**합니다.

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

---

## 🧩 프로젝트 구조 (요약약)

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

```python
from lex_dpr import BiEncoder
import numpy as np

# 기본 모델 사용
encoder = BiEncoder()

# 질의 임베딩 생성
queries = ["통신과금서비스 등록은 어떻게 하나요?"]
query_embeddings = encoder.encode_queries(queries)

# 패시지 임베딩 생성
passages = ["1 통신과금서비스를 제공하려는 자는 대통령령으로 정하는 바에 따라 다음 각 호의 사항을 갖추어 과학기술정보통신부장관에게 등록하여야 한다."]
passage_embeddings = encoder.encode_passages(passages)

# 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(query_embeddings, passage_embeddings)[0][0]
print(f"유사도: {similarity:.4f}")
```

---

## 사용 예시

### 1. Git LFS 설정

패키지에 포함된 모델 파일은 Git LFS로 관리됩니다. 패키지를 사용하기 전에 Git LFS로 실제 파일을 다운로드해야 합니다.

**Git LFS 설치:**

```bash
# Red Hat/CentOS 계열
sudo yum install -y git-lfs

# Ubuntu/Debian 계열
sudo apt-get install -y git-lfs

# Git LFS 초기화 (처음 한 번만)
git lfs install
```

**모델 파일 다운로드:**

```bash
# 모든 LFS 파일 다운로드
git lfs pull

# 또는 특정 디렉토리만 다운로드
git lfs pull --include="lex_dpr/models/default_model/**"
```

**트러블슈팅:**

**`safetensors_rust.SafetensorError: Error while deserializing header: header too large` 에러:**

이 에러는 패키지에 포함된 모델 파일이 Git LFS 포인터 파일로만 존재할 때 발생합니다.

**해결 방법:**

1. **Git LFS로 실제 파일 다운로드:**
   ```bash
   # Git LFS 설치 확인
   git lfs version
   
   # Git LFS 초기화 (처음 한 번만)
   git lfs install
   
   # 실제 파일 다운로드
   git lfs pull
   ```

2. **특정 디렉토리만 다운로드:**
   ```bash
   git lfs pull --include="lex_dpr/models/default_model/**"
   ```

3. **LFS 파일 확인:**
   ```bash
   # LFS로 추적되는 파일 목록 확인
   git lfs ls-files
   
   # 특정 파일이 LFS 포인터인지 확인
   head -n 1 lex_dpr/models/default_model/adapter_model.safetensors
   # "version https://git-lfs.github.com/spec/v1"로 시작하면 포인터 파일
   ```

### 2. 임베딩 생성

#### Python API

**기본 사용 (권장):**
```python
from lex_dpr import BiEncoder

# 기본 모델 사용
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

**입력 JSONL 파일 형식:**

각 줄은 JSON 객체이며, `id`와 `text` 필드를 포함해야 합니다. 추가 필드는 선택사항입니다.

**질의 파일 예시 (`queries.jsonl`):**
```jsonl
{"id": "q1", "text": "통신과금서비스 등록은 어떻게 하나요?"}
{"id": "q2", "text": "정보통신망법상 통신과금서비스 제공자의 의무는?"}
{"id": "q3", "text": "통신과금서비스 등록 요건은 무엇인가요?"}
```

**패시지 파일 예시 (`passages.jsonl`):**
```jsonl
{"id": "LAW_000030_제53조_①", "parent_id": "LAW_000030_제53조", "type": "법령", "law_id": "000030", "law_name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률", "article": "제53조", "effective_date": "20251001", "text": "1 통신과금서비스를 제공하려는 자는 대통령령으로 정하는 바에 따라 다음 각 호의 사항을 갖추어 과학기술정보통신부장관에게 등록하여야 한다. <개정 2008.2.29, 2013.3.23, 2017.7.26>"}
{"id": "LAW_000030_제53조_②", "parent_id": "LAW_000030_제53조", "type": "법령", "law_id": "000030", "law_name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률", "article": "제53조", "effective_date": "20251001", "text": "2 제1항에 따라 등록한 사항을 변경하려는 자는 변경등록을 하여야 한다."}
{"id": "LAW_000030_제54조_①", "parent_id": "LAW_000030_제54조", "type": "법령", "law_id": "000030", "law_name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률", "article": "제54조", "effective_date": "20251001", "text": "1 통신과금서비스제공자는 이용자의 지급의사 확인, 거래내용의 증빙 및 분쟁조정을 위한 기록을 5년간 보관하여야 한다."}
```

**기본 모델 사용 (권장):**
```bash
# 질의 임베딩 생성
lex-dpr embed \
  --model default \
  --input queries.jsonl \
  --outdir embeddings \
  --prefix queries \
  --type query \
  --batch-size 64

# 패시지 임베딩 생성
lex-dpr embed \
  --model default \
  --input passages.jsonl \
  --outdir embeddings \
  --prefix passages \
  --type passage \
  --batch-size 64
```

**특정 모델 경로 지정:**
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

**커스텀 필드명 사용:**
```bash
# id 필드가 "doc_id", text 필드가 "content"인 경우
lex-dpr embed \
  --model default \
  --input data.jsonl \
  --outdir embeddings \
  --prefix docs \
  --type passage \
  --id-field doc_id \
  --text-field content
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


