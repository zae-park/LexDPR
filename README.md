# 🏛️ LexDPR  
**구조화되고 계층적인 법령 및 규범 문서를 위한 Dense Passage Retrieval 모델**

LexDPR은 **법령, 규정, 비조치의견서 등과 같은 구조화된 문서**를 대상으로 하는 **Dense Passage Retrieval (DPR)** 모델입니다.  
조·항·호 단위의 계층적 구조를 가진 문서를 효율적으로 인덱싱하고, 의미적 일관성을 유지하며 검색 성능을 향상시키는 것을 목표로 합니다.

---

## 📘 프로젝트 개요

기존의 상용 임베딩 모델(OpenAI, Cohere, Sentence-Transformers 등)은 다음과 같은 문제를 가집니다:

- **계층 구조**가 깊은 문서(조/항/호 등)에 대한 표현 부족  
- **법령 문맥 의존성**이 높은 구문 처리의 불안정성  
- **의미적으로 연결된 문장 간 거리 문제**로 인한 검색 정확도 저하  

LexDPR은 **법령 문서 구조에 최적화된 Dense Passage Retrieval 파이프라인**으로, **RAG 시스템의 중간 검색기(retriever)** 역할을 수행합니다.

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

---

## 🧩 프로젝트 구조

```
📁 LexDPR/
 ├── data/
 │    ├── statutes/             # 법령 문서 (조/항/호 단위)
 │    ├── no_action_letters/    # 비조치의견서 데이터
 │    ├── queries/              # 질의 데이터 (JSONL)
 │    └── processed/            # 전처리 후 corpus.jsonl
 │
 ├── scripts/
 │    ├── preprocess_acts.py    # 법령 문서 전처리 및 청크 생성
 │    ├── encode_passages.py    # 패시지 임베딩 생성
 │    ├── encode_queries.py     # 질의 임베딩 생성
 │    ├── build_index.py        # FAISS 인덱스 구축
 │    └── evaluate.py           # 평가 (Recall@k 등)
 │
 ├── configs/
 │    ├── base.yaml             # 기본 하이퍼파라미터 설정
 │    ├── model.yaml            # 인코더 아키텍처 설정
 │    └── data.yaml             # 데이터 경로 및 전처리 옵션
 │
 ├── run_demo_real.sh           # 전체 파이프라인 실행 스크립트
 ├── README.md
 └── requirements.txt
```

---

## 🔍 주요 기능

- **실제 Sentence-Transformers 기반 DPR 인코더 사용**  
  `sentence-transformers/all-MiniLM-L6-v2` 기반 인코딩 (필요 시 Legal-BERT 교체 가능)
- **구조 인식형 청크 분할**  
  조문·항 단위의 계층 구조를 분석해 문맥 단위별로 세분화
- **FAISS 기반 벡터 인덱싱**  
  대용량 법령 데이터의 효율적 검색
- **Recall@k 평가 자동화**  
  `data/queries/queries.jsonl`의 positive passage 기준으로 정확도 평가
- **간단한 실행**  
  `run_demo_real.sh` 하나로 전처리→임베딩→인덱스→평가 일괄 수행

---

## 🧠 활용 분야

- 법령 및 규제 문서 검색 시스템  
- 비조치의견서 / 행정해석 질의응답 검색  
- 규제 준수(Compliance) 자동화 도구  
- 계약서 / 정책 / 규정 기반 QA RAG 시스템  

---

## 사용 예시

```bash
# 1. 의존성 설치 (Poetry 패키지 매니저)
poetry install

# 웹 로깅 서비스 사용 시 (선택사항)
# 모든 웹 로깅 서비스 설치:
poetry install --extras "web-logging"
# 또는 개별 서비스만 설치:
poetry install --extras "wandb"      # WandB만
poetry install --extras "neptune"    # Neptune만
poetry install --extras "mlflow"     # MLflow만
# 여러 서비스 동시 설치:
poetry install --extras "wandb neptune"

# 개발 시 (웹 로깅 서비스 포함하여 개발)
# 방법 1: extras 사용
poetry install --extras "web-logging"
# 방법 2: 개발 그룹과 함께 설치 (향후 추가 예정)
# poetry install --with dev

# 2. 설정 파일 초기화
poetry run lex-dpr config init

# 3. 학습 실행
poetry run lex-dpr train
# 또는 설정 오버라이드:
poetry run lex-dpr train trainer.epochs=5 trainer.lr=3e-5
```



## 🚀 사용 예시 (DEPRECATED)

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 전체 파이프라인 실행 (모델 가중치 자동 다운로드)
bash run_demo_real.sh
```

또는 수동 실행:

```bash
# 전처리
python scripts/preprocess_acts.py --input data/statutes --output data/processed/corpus.jsonl
python scripts/preprocess_acts.py --input data/no_action_letters --output data/processed/tmp.jsonl
cat data/processed/tmp.jsonl >> data/processed/corpus.jsonl && rm data/processed/tmp.jsonl

# 임베딩 생성
python scripts/encode_passages.py --input data/processed/corpus.jsonl --outdir checkpoint
python scripts/encode_queries.py --queries data/queries/queries.jsonl --outdir checkpoint

# 인덱스 빌드 및 평가
python scripts/build_index.py --input checkpoint --output index --factory Flat --metric dot
python scripts/evaluate.py --index_dir index --queries data/queries/queries.jsonl --top_k 10
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


---

## 📄 인용 정보

```
@misc{lexdpr2025,
  author = {박성재},
  title  = {LexDPR: 구조화된 법령 문서를 위한 Dense Passage Retrieval 모델},
  year   = {2025},
  url    = {https://github.com/zae-park/LexDPR}
}
```

---

## 🧾 라이선스

MIT License  
공공데이터(예: 비조치의견서, 법령 DB)는 각 출처의 오픈라이선스 정책을 반드시 준수해야 합니다.

---

## 📚 추가 문서

- **[Git LFS 사용 가이드](docs/GIT_LFS_GUIDE.md)**: 모델 체크포인트와 대용량 파일을 Git LFS로 관리하는 방법, org-mirror와 origin 동기화 시 주의사항


###
- https://www.law.go.kr/DRF/lawSearch.do?OC=hanwhasbank01&target=prec&type=HTML&&query=
- https://www.law.go.kr/DRF/lawSearch.do?query=*&target=fsc&OC=hanwhasbank01&search=2&display=20&nw=1&page=2&refAdr=law.go.kr&type=HTML&popYn=N

### 
