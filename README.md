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
poetry install --extras "mlflow"     # MLflow만
# 여러 서비스 동시 설치:
poetry install --extras "wandb mlflow"

# 개발 시 (웹 로깅 서비스 포함하여 개발)
# 방법 1: extras 사용
poetry install --extras "web-logging"
# 방법 2: 개발 그룹과 함께 설치 (향후 추가 예정)
# poetry install --with dev

# 2. 설정 파일 초기화
poetry run lex-dpr config init

# 2-1. (선택) 판례 크롤링 - law.go.kr에서 판례 JSON 수집
#    PAGE 번호를 기준으로 시작 페이지와 최대 페이지 수를 지정할 수 있습니다.
#    (첫 실행 시 데이터 준비용으로 권장)
poetry run lex-dpr crawl-precedents --max-pages 50
# 또는
poetry run lex-dpr crawl-precedents --start-page 51 --max-pages 50

# 2-2. 질의-passage 쌍 생성 (train/valid/test split 포함)
#     - law/admin/precedent passage를 이용해 pairs_train/valid/test를 생성합니다.
poetry run lex-dpr gen-data
# 결과 파일:
#   - data/pairs_train.jsonl
#   - data/pairs_train_valid.jsonl
#   - data/pairs_train_test.jsonl
#   - data/pairs_eval.jsonl (valid 세트 복사본, 학습/평가에 사용)

# 3. 학습 명령어 정리
# ============================================
# 📌 명령어별 용도 요약:
# 
# 1. train: 지정된 파라미터로 학습
#    - configs/base.yaml 기반으로 학습
#    - 모든 기능을 수동으로 설정 가능
#    - 실제 학습 시 사용
#
# 2. smoke-train: 모든 기능 활성화 + 반복 파라미터만 제한
#    - 모든 기능 자동 활성화 (LR scheduler, gradient clipping, early stopping)
#    - test_run=true, epochs=1로 제한하여 빠른 테스트
#    - 파이프라인 동작 확인용
#
# 3. sweep: 지정된 파라미터 범위를 하이퍼파라미터 탐색
#    - configs/sweep.yaml 기반으로 실제 하이퍼파라미터 탐색
#    - WandB Sweep을 통한 Bayesian optimization
#    - 실제 최적화 시 사용
#
# 4. smoke-sweep: 최소한의 기능 + 최소한의 반복으로 sweep 테스트
#    - sweep 명령어에 --smoke-test 플래그 사용
#    - 또는 configs/smoke_sweep.yaml 사용
#    - Sweep 파이프라인 동작 확인용
# ============================================

# 3-1. 정상 학습 (train)
#     - configs/base.yaml 기반으로 학습
#     - 모든 하이퍼파라미터를 수동으로 설정
poetry run lex-dpr train
# 또는 설정 오버라이드:
poetry run lex-dpr train trainer.epochs=5 trainer.lr=3e-5

# 3-2. 빠른 SMOKE TEST 학습 (smoke-train)
#     - 모든 기능 자동 활성화:
#       * Learning Rate Scheduler: Warm-up + Cosine Annealing
#       * Gradient Clipping: 활성화 (max_norm=1.0)
#       * Early Stopping: 활성화 (patience=2)
#     - 반복 파라미터만 제한: test_run=true, epochs=1
#     - 파이프라인 동작 확인용
poetry run lex-dpr smoke-train
# 추가 하이퍼파라미터는 덮어쓸 수 있습니다 (test_run/epochs/기능 활성화는 고정):
poetry run lex-dpr smoke-train trainer.lr=3e-5

# 3-2. Early Stopping 활성화
#     - Validation 메트릭을 모니터링하여 학습을 조기 종료
#     - 최고 성능 모델을 자동으로 저장
#     configs/base.yaml에서 설정:
#       trainer:
#         early_stopping:
#           enabled: true
#           metric: "cosine_ndcg@10"  # 모니터링할 메트릭
#           patience: 3  # 개선이 없을 때 기다릴 평가 횟수
#           mode: "max"  # "max" 또는 "min"
#     또는 명령줄에서:
poetry run lex-dpr train trainer.early_stopping.enabled=true trainer.early_stopping.patience=5

# 학습 스케줄러: Warm-up + Cosine Annealing
# - 전체 학습 step의 10%에서 warmup 수행
# - 이후 cosine annealing으로 학습률 감소
# - 자동으로 설정되므로 별도 설정 불필요

# 3-3. Gradient Clipping 활성화
#     - Gradient explosion 방지를 위한 gradient clipping
#     configs/base.yaml에서 설정:
#       trainer:
#         gradient_clip_norm: 1.0  # 최대 노름 값 (0.0이면 비활성화)
#     또는 명령줄에서:
poetry run lex-dpr train trainer.gradient_clip_norm=1.0

# 4. 학습된 모델 평가
#    MRR@k, NDCG@k, MAP@k, Precision/Recall@k 등 Retrieval 메트릭을 계산합니다.

# 기본 평가 (JSON 출력)
poetry run lex-dpr eval
poetry run lex-dpr eval \
  --model checkpoint/lexdpr/bi_encoder \
  --passages data/processed/merged_corpus.jsonl \
  --eval-pairs data/pairs_eval.jsonl \
  --k-values 1 3 5 10 \
  --output eval_results.json

# 상세 분석 리포트 (쿼리별, 소스별, 실패 케이스 분석 포함)
poetry run lex-dpr eval \
  --model checkpoint/lexdpr/bi_encoder \
  --detailed \
  --report eval_detailed_report.txt \
  --output eval_detailed_results.json

# 여러 모델 비교 평가 (Sweep으로 학습된 모델들 비교)
poetry run lex-dpr eval \
  --compare-models \
    checkpoint/model1 \
    checkpoint/model2 \
    checkpoint/model3 \
  --compare-output model_comparison_report.txt \
  --output model_comparison.json

# 5. 하이퍼파라미터 튜닝 (WandB Sweep)
# ============================================
# 📌 Sweep 명령어 정리:
#
# 1. sweep (실제): configs/sweep.yaml 기반으로 실제 하이퍼파라미터 탐색
#    - Bayesian optimization으로 최적 파라미터 탐색
#    - 여러 날짜에 나눠서 실행 가능
#
# 2. sweep (smoke-test): 최소한의 기능 + 최소한의 반복으로 sweep 테스트
#    - --smoke-test 플래그 사용 또는 configs/smoke_sweep.yaml 사용
#    - Sweep 파이프라인 동작 확인용
# ============================================

# 5-1. 실제 하이퍼파라미터 탐색 (sweep)
#     - configs/sweep.yaml 기반으로 실제 하이퍼파라미터 탐색
#     - Bayesian optimization으로 최적 파라미터 탐색
#     - 여러 날짜에 나눠서 실행 가능
poetry run lex-dpr sweep --config configs/sweep.yaml --no-smoke-test

# 5-2. Sweep 파이프라인 테스트 (smoke-sweep)
#     - 최소한의 기능, 최소한의 반복 파라미터로 sweep 테스트
#     - Sweep 파이프라인 동작 확인용
poetry run lex-dpr sweep --smoke-test
# 또는 설정 파일 직접 지정:
poetry run lex-dpr sweep --config configs/smoke_sweep.yaml --smoke-test

# 5-3. 스윕 설정 파일 생성 (템플릿)
poetry run lex-dpr sweep init --output configs/my_sweep.yaml
# SMOKE TEST 모드용 템플릿 생성:
poetry run lex-dpr sweep init --output configs/smoke_sweep.yaml --smoke-test

# 5-4. 설정 파일 편집 (탐색할 파라미터 범위 설정)
# vim configs/my_sweep.yaml
#
# 예시 설정 (configs/my_sweep.yaml):
# ---
# method: bayes  # grid, random, bayes 중 선택
# metric:
#   name: eval/ndcg@10
#   goal: maximize
# parameters:
#   trainer.lr:
#     distribution: log_uniform_values
#     values: [1e-6, 1e-5, 1e-4, 1e-3]
#   trainer.temperature:
#     distribution: uniform
#     min: 0.01
#     max: 0.2
# fixed:
#   trainer.epochs: 3
#   data.pairs: data/pairs_train.jsonl
#   data.passages: data/merged_corpus.jsonl
# # 시간 기반 제어 (선택사항)
# time_window: "1-8"  # 1시~8시에만 실행 (KST 기준)
# timezone: "Asia/Seoul"
# # Early termination 설정 (선택사항, 베이지안 탐색 수렴 시 자동 종료)
# early_terminate:
#   type: hyperband
#   min_iter: 3
#   max_iter: 27
#   s: 2

# 5-3. 스윕 시작 (WandB에 스윕 생성)
# 방법 1: 스윕 생성 + 에이전트 자동 실행 (기본 동작)
poetry run lex-dpr sweep
# 또는
poetry run lex-dpr sweep start --config configs/my_sweep.yaml

# 방법 2: 스윕만 생성하고 에이전트는 나중에 실행
poetry run lex-dpr sweep --no-run-agent
# 또는
poetry run lex-dpr sweep start --config configs/my_sweep.yaml --no-run-agent

# SMOKE TEST 모드로 실행 (test_run=true, epochs=1 자동 적용):
poetry run lex-dpr sweep start --config configs/my_sweep.yaml --smoke-test

# 5-6. 에이전트 실행 (여러 날짜/머신에서 나눠서 실행 가능)
# 설정 파일에서 스윕 ID 자동 읽기:
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml

# 스윕 ID 직접 지정:
poetry run lex-dpr sweep agent <sweep-id>

# 특정 횟수만 실행 (예: 오늘은 5개만):
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 5

# 시간 기반 제어 (특정 시간대에만 실행):
# CLI에서 직접 지정:
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --time-window "1-8" --count 10
# 또는 설정 파일의 time_window 사용 (자동 적용)

# 여러 날짜에 나눠서 실행하는 방법:
# 첫 날: poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10 --time-window "1-8"
# 둘째 날: poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10 --time-window "1-8"
# 셋째 날: poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10 --time-window "1-8"
# (같은 스윕에 계속 참여하여 탐색 진행)

# WandB 대시보드에서 진행 상황 확인:
# https://wandb.ai/<entity>/<project>/sweeps/<sweep-id>

# 스윕 종료 조건:
# - 기본적으로 무한정 실행됨 (모든 파라미터 조합 탐색)
# - --count 옵션으로 실행 횟수 제한 가능
# - WandB 대시보드에서 수동으로 중단 가능
# - 스윕 설정에서 early_terminate 설정 가능 (베이지안 탐색 시 최적 파라미터 찾으면 자동 종료)
# - 시간 기반 제어: time_window 설정 시 지정된 시간 범위 밖에서는 자동 대기
```

---

## 🔧 WandB Sweep 하이퍼파라미터 튜닝 상세 가이드

LexDPR은 WandB Sweep을 통한 하이퍼파라미터 자동 튜닝을 지원합니다. 여러 날짜에 나눠서 실행하거나, 특정 시간대에만 실행하는 등 유연한 스윕 관리가 가능합니다.

### 주요 기능

- **다양한 탐색 방법**: Grid Search, Random Search, Bayesian Optimization
- **여러 날짜/머신에서 실행**: 같은 스윕에 여러 에이전트가 참여하여 병렬 탐색
- **시간 기반 제어**: 특정 시간대에만 실행하도록 설정 가능 (예: 야간 시간대)
- **Early Termination**: 베이지안 탐색 시 성능 개선이 없으면 자동 종료
- **SMOKE TEST 모드**: 빠른 검증을 위한 축소 모드 지원

### 스윕 설정 파일 예시

```yaml
# configs/my_sweep.yaml
method: bayes  # grid, random, bayes 중 선택

metric:
  name: eval/ndcg@10
  goal: maximize

parameters:
  trainer.lr:
    distribution: log_uniform_values
    values: [1e-6, 1e-5, 1e-4, 1e-3]
  
  trainer.temperature:
    distribution: uniform
    min: 0.01
    max: 0.2
  
  trainer.gradient_accumulation_steps:
    values: [4, 8, 16]

fixed:
  trainer.epochs: 3
  data.pairs: data/pairs_train.jsonl
  data.passages: data/merged_corpus.jsonl

# 시간 기반 제어 (선택사항)
time_window: "1-8"  # 1시~8시에만 실행 (KST 기준)
timezone: "Asia/Seoul"

# Early Termination 설정 (선택사항)
early_terminate:
  type: hyperband
  min_iter: 3
  max_iter: 27
  s: 2
```

### 시간 기반 제어 사용법

스윕 에이전트를 특정 시간대에만 실행하도록 설정할 수 있습니다. 이는 GPU 리소스를 효율적으로 사용하거나, 특정 시간대에만 학습을 진행하고 싶을 때 유용합니다.

```bash
# CLI에서 직접 지정
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --time-window "1-8" --count 10

# 설정 파일에 time_window가 있으면 자동 적용
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10
```

시간 범위 밖에서 실행하면, 에이전트는 다음 시작 시간까지 자동으로 대기합니다.

### 여러 날짜에 나눠서 실행

큰 스윕을 여러 날에 걸쳐 실행할 수 있습니다:

```bash
# 첫 날: 10개 실행
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10 --time-window "1-8"

# 둘째 날: 또 10개 실행 (같은 스윕에 계속 참여)
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10 --time-window "1-8"

# 셋째 날: 마지막 10개 실행
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10 --time-window "1-8"
```

각 날짜마다 실행한 결과는 모두 같은 스윕에 누적되어 WandB 대시보드에서 확인할 수 있습니다.

### Early Termination 설정

베이지안 탐색(Bayesian Optimization)을 사용할 때, 성능 개선이 없으면 자동으로 스윕을 종료할 수 있습니다:

```yaml
early_terminate:
  type: hyperband
  min_iter: 3      # 최소 반복 횟수
  max_iter: 27     # 최대 반복 횟수
  s: 2             # Successive Halving 파라미터
```

이 설정은 WandB의 Hyperband 알고리즘을 사용하여 성능이 낮은 실행을 조기에 종료하고, 유망한 실행에 더 많은 리소스를 할당합니다.

### SMOKE TEST 모드

스윕 설정이 올바른지 빠르게 확인하고 싶을 때 SMOKE TEST 모드를 사용할 수 있습니다:

```bash
# SMOKE TEST 모드용 설정 파일 생성
poetry run lex-dpr sweep init --output configs/smoke_sweep.yaml --smoke-test

# SMOKE TEST 모드로 스윕 실행
poetry run lex-dpr sweep start --config configs/smoke_sweep.yaml --smoke-test
```

SMOKE TEST 모드에서는 `test_run=true`, `epochs=1`이 자동으로 적용되어 빠르게 검증할 수 있습니다.

---

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
