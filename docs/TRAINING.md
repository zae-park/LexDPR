# 모델 학습 가이드

이 문서는 LexDPR 패키지를 사용하여 모델을 학습하는 방법을 설명합니다.

## 목차

1. [초기 설정](#초기-설정)
2. [데이터 준비](#데이터-준비)
3. [모델 학습](#모델-학습)
4. [모델 평가](#모델-평가)
5. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)

---

## 초기 설정

### 설정 파일 초기화

프로젝트를 처음 clone한 경우, `configs/` 디렉토리에 설정 파일이 이미 포함되어 있습니다.  
설정 파일이 없는 경우에만 다음 명령어로 초기화할 수 있습니다:

```bash
# 설정 파일 초기화 (configs/ 디렉토리가 없거나 비어있는 경우)
lex-dpr config init

# 또는 Python 모듈 직접 실행
python -m lex_dpr.cli.config init
```

---

## 데이터 준비

### 1. 데이터 전처리 (Passage 생성)

법령, 행정규칙, 판례 JSON을 passage JSONL로 변환합니다.

**Passage 분할 단위:**
- **법령 (항 단위)**: 조문 → 항 → 호 계층 구조를 분석하여 항 단위로 passage 생성
- **행정규칙 (조문 단위)**: 조문 단위로 passage 생성
- **판례 (청크 단위)**: 길이 기반 슬라이딩 윈도우로 청크 분할 (기본값: 최대 1200자, 오버랩 200자)

#### CLI 방식

```bash
# 법령 Passage 생성
python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/laws \
  --out-law data/processed/law_passages.jsonl \
  --glob "**/*.json"

# 행정규칙 Passage 생성
python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/admin_rules \
  --out-admin data/processed/admin_passages.jsonl \
  --glob "**/*.json"

# 판례 Passage 생성
python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/precedents \
  --out-prec data/processed/prec_passages.jsonl \
  --glob "**/*.json"

# 법령을 호 단위까지 세분화 (선택사항)
python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/laws \
  --out-law data/processed/law_passages.jsonl \
  --glob "**/*.json" \
  --include-items

# Passage 코퍼스 병합 (선택사항, 평가용)
python -m lex_dpr.data_processing.merge_corpus \
  --law data/processed/law_passages.jsonl \
  --admin data/processed/admin_passages.jsonl \
  --out data/processed/merged_corpus.jsonl
```

### 2. 데이터셋 생성 (질의-passage 쌍)

law/admin/precedent passage를 이용해 train/valid/test 데이터셋을 생성합니다.

**판례 데이터 처리 방식:**
1. **우선순위**: 참조조문에서 법령/행정규칙 매칭 → positive passage로 사용
2. **Fallback**: 참조조문이 없거나 매칭 실패 시 판례 본문 청크 사용

#### CLI 방식

```bash
# 기본 사용 (기본 경로 사용)
lex-dpr gen-data

# 판례 원본 JSON 디렉토리 직접 지정
lex-dpr gen-data \
  --law data/processed/law_passages.jsonl \
  --admin data/processed/admin_passages.jsonl \
  --prec-json-dir data/precedents \
  --out data/pairs_train.jsonl
```

**결과 파일:**
- `data/pairs_train.jsonl`
- `data/pairs_train_valid.jsonl`
- `data/pairs_train_test.jsonl`
- `data/pairs_eval.jsonl` (valid 세트 복사본, 학습/평가에 사용)

### 3. 판례 크롤링 (선택사항)

law.go.kr에서 판례 JSON을 수집합니다.

```bash
# 기본 사용 (최대 50페이지)
lex-dpr crawl-precedents --max-pages 50

# 시작 페이지 지정
lex-dpr crawl-precedents --start-page 51 --max-pages 50
```

---

## 모델 학습

**명령어별 용도:**
- `train`: 지정된 파라미터로 학습 (configs/base.yaml 기반)
- `smoke-train`: 빠른 테스트용 (모든 기능 활성화, epochs=1)
- `sweep`: 하이퍼파라미터 탐색 (WandB Sweep 사용)
- `smoke-sweep`: Sweep 파이프라인 테스트용

### CLI 방식

```bash
# 정상 학습
lex-dpr train

# 설정 오버라이드
lex-dpr train trainer.epochs=5 trainer.lr=3e-5

# SMOKE TEST 학습 (빠른 테스트)
lex-dpr smoke-train
lex-dpr smoke-train trainer.lr=3e-5

# Early Stopping 활성화
lex-dpr train trainer.early_stopping.enabled=true trainer.early_stopping.patience=5

# Gradient Clipping 활성화
lex-dpr train trainer.gradient_clip_norm=1.0
```

**학습 스케줄러**: Warm-up + Cosine Annealing (자동 설정)

---

## 모델 평가

MRR@k, NDCG@k, MAP@k, Precision/Recall@k 등 Retrieval 메트릭을 계산합니다.

### CLI 방식

```bash
# 기본 평가 (JSON 출력)
lex-dpr eval

# 상세 옵션 지정
lex-dpr eval \
  --model checkpoint/lexdpr/bi_encoder \
  --passages data/processed/merged_corpus.jsonl \
  --eval-pairs data/pairs_eval.jsonl \
  --k-values 1 3 5 10 \
  --output eval_results.json

# 상세 분석 리포트 (쿼리별, 소스별, 실패 케이스 분석 포함)
lex-dpr eval \
  --model checkpoint/lexdpr/bi_encoder \
  --detailed \
  --report eval_detailed_report.txt \
  --output eval_detailed_results.json

# 여러 모델 비교 평가
lex-dpr eval \
  --compare-models \
    checkpoint/model1 \
    checkpoint/model2 \
    checkpoint/model3 \
  --compare-output model_comparison_report.txt \
  --output model_comparison.json

# WandB에 결과 로깅
lex-dpr eval \
  --model checkpoint/lexdpr/bi_encoder \
  --wandb \
  --wandb-project lexdpr-eval
```

**평가 메트릭:**
- MRR@k (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Recall@k (재현율)
- Precision@k (정밀도)
- MAP@k (Mean Average Precision)

---

## 하이퍼파라미터 튜닝

WandB Sweep을 통한 하이퍼파라미터 자동 튜닝을 지원합니다.

**주요 기능:**
- Grid Search, Random Search, Bayesian Optimization
- 여러 날짜/머신에서 나눠서 실행
- 시간 기반 제어 (특정 시간대에만 실행)
- Early Termination (성능 개선 없으면 자동 종료)
- SMOKE TEST 모드 (빠른 검증)

### CLI 방식

```bash
# 스윕 설정 파일 생성
lex-dpr sweep init --output configs/my_sweep.yaml
lex-dpr sweep init --output configs/smoke_sweep.yaml --smoke-test

# 실제 하이퍼파라미터 탐색
lex-dpr sweep --config configs/sweep.yaml --no-smoke-test

# SMOKE TEST 모드
lex-dpr sweep --smoke-test
lex-dpr sweep --config configs/smoke_sweep.yaml --smoke-test

# 스윕 시작 (에이전트 자동 실행)
lex-dpr sweep start --config configs/my_sweep.yaml

# 스윕만 생성 (에이전트는 나중에 실행)
lex-dpr sweep start --config configs/my_sweep.yaml --no-run-agent

# 에이전트 실행
lex-dpr sweep agent --config configs/my_sweep.yaml
lex-dpr sweep agent <sweep-id>

# 특정 횟수만 실행
lex-dpr sweep agent --config configs/my_sweep.yaml --count 5

# 시간 기반 제어
lex-dpr sweep agent --config configs/my_sweep.yaml --time-window "1-8" --count 10

# 백그라운드 실행
nohup lex-dpr sweep agent --config configs/sweep.yaml \
  > logs/sweep_agent.log 2>&1 &
```

**스윕 설정 파일 예시 (configs/my_sweep.yaml):**

```yaml
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

자세한 내용은 [WandB Sweep 하이퍼파라미터 튜닝 상세 가이드](../README.md#-wandb-sweep-하이퍼파라미터-튜닝-상세-가이드)를 참고하세요.

---

## 데이터 품질 분석

Passage corpus와 학습 데이터셋의 품질을 분석합니다.

### CLI 방식

```bash
# Passage corpus 분석
lex-dpr analyze-passages \
  --corpus data/processed/merged_corpus.jsonl \
  --tokenizer BAAI/bge-m3 \
  --output passage_analysis.txt \
  --json-output passage_analysis.json

# Train/Valid/Test 데이터셋 분석
lex-dpr analyze-pairs \
  --pairs-dir data \
  --passages data/processed/merged_corpus.jsonl \
  --tokenizer BAAI/bge-m3 \
  --output pairs_analysis.txt
```

**분석 항목:**
- Passage corpus: 총 passage 개수, 소스별 분포, 중복 탐지, 길이 분포
- 데이터셋: 크기, Positive/Negative 비율, 쿼리 타입별 분포, 토큰 길이 분포

---

## 임베딩 시각화

학습된 모델의 임베딩 품질을 시각화합니다.

**시각화 타입:**
- `embedding-space`: 임베딩 공간 시각화 (t-SNE/UMAP)
- `similarity`: Positive vs Negative 유사도 분포
- `heatmap`: 쿼리-패시지 유사도 히트맵
- `comparison`: 학습 전후 비교

```bash
# 모든 시각화 생성
lex-dpr visualize \
  --model checkpoint/lexdpr/bi_encoder \
  --passages data/merged_corpus.jsonl \
  --eval-pairs data/pairs_eval.jsonl \
  --output visualizations

# 특정 시각화만 생성
lex-dpr visualize \
  --model checkpoint/lexdpr/bi_encoder \
  --type similarity \
  --output visualizations
```

