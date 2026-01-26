# 임베딩 모델 파인튜닝 가이드

## 목차

- [임베딩 모델 파인튜닝 가이드](#임베딩-모델-파인튜닝-가이드)
  - [목차](#목차)
  - [1. 데이터](#1-데이터)
    - [1-1. 데이터 유형](#1-1-데이터-유형)
    - [1-2. 데이터 가공 (Passage 생성)](#1-2-데이터-가공-passage-생성)
    - [1-3. 데이터 전처리 (질의-Passage 쌍 생성)](#1-3-데이터-전처리-질의-passage-쌍-생성)
    - [1-4. 코퍼스 생성](#1-4-코퍼스-생성)
    - [1-5. 데이터 구조](#1-5-데이터-구조)
      - [Passage 형식](#passage-형식)
      - [Pair 형식](#pair-형식)
  - [2. 학습 명령어](#2-학습-명령어)
    - [2-1. Preset 명령어 요약](#2-1-preset-명령어-요약)
    - [2-2. 커스텀 설정 방법 \& Agent 사용 (Sweep 재사용)](#2-2-커스텀-설정-방법--agent-사용-sweep-재사용)
      - [설정 파일 생성](#설정-파일-생성)
      - [설정 파일 편집](#설정-파일-편집)
      - [스윕 생성 및 실행](#스윕-생성-및-실행)
    - [2-3. 로그 저장, 확인 방법 \& 체크포인트 경로](#2-3-로그-저장-확인-방법--체크포인트-경로)
      - [WandB 로그 저장 위치](#wandb-로그-저장-위치)
      - [로그 파일 구조](#로그-파일-구조)
      - [로그 확인 방법](#로그-확인-방법)
      - [학습 성패 판단 방법](#학습-성패-판단-방법)
      - [체크포인트 경로](#체크포인트-경로)
  - [부록. 전체 파이프라인 요약 및 로그 요약](#부록-전체-파이프라인-요약-및-로그-요약)
    - [전체 파이프라인 예시](#전체-파이프라인-예시)
    - [데이터 저장 위치 요약](#데이터-저장-위치-요약)
    - [로그 저장 위치 요약](#로그-저장-위치-요약)

---

## 1. 데이터

### 1-1. 데이터 유형

학습에는 총 3가지 유형(법령, 행정규칙, 판례) 데이터가 사용되었습니다. 각 데이터는 passage 단위로 chunk되며, 판례의 경우는 (사용자)질의-passage 쌍의 데이터를 구성하여 임베딩 품질을 높이는데 사용됩니다.

| 데이터 유형  | 저장 위치           | 파일명 형식           | 예시 파일명                                 |
| ------------ | ------------------- | --------------------- | ------------------------------------------- |
| **법령**     | `data/laws/`        | `{법령ID}.json`       | `000030.json`, `000130.json`, `001537.json` |
| **행정규칙** | `data/admin_rules/` | `{행정규칙ID}.json`   | `2100000243102.json`, `2200000106255.json`  |
| **판례**     | `data/precedents/`  | `{판례일련번호}.json` | `180200.json`, `170801.json`                |

### 1-2. 데이터 가공 (Passage 생성)

원시 JSON 데이터를 passage 형식으로 변환합니다:

```bash
# 법령 passage 생성
poetry run python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/laws \
  --out-law data/processed/law_passages.jsonl \
  --glob "**/*.json"

# 행정규칙 passage 생성
poetry run python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/admin_rules \
  --out-admin data/processed/admin_passages.jsonl \
  --glob "**/*.json"
```

**생성되는 파일:**
- `data/processed/law_passages.jsonl`: 법령 passage
- `data/processed/admin_passages.jsonl`: 행정규칙 passage

### 1-3. 데이터 전처리 (질의-Passage 쌍 생성)

학습에 사용할 질의-passage 쌍을 생성합니다:

```bash
poetry run python -m lex_dpr.data_processing.make_pairs \
  --law data/processed/law_passages.jsonl \
  --admin data/processed/admin_passages.jsonl \
  --prec-json-dir data/precedents \
  --out data/processed/pairs_train.jsonl \
  --use-admin-for-prec \
  --hn_per_q 32 \
  --max-positives-per-prec 5 \
  --max-workers 8
```

**주요 옵션:**
- `--hn_per_q`: 질의당 hard negative 개수 (기본값: 32)
- `--max-positives-per-prec`: 판례당 최대 positive passage 개수 (기본값: 5)
- `--max-workers`: 병렬 처리 워커 수

**생성되는 파일:**
- `data/processed/pairs_train.jsonl`: 학습용 질의-passage 쌍
- `data/processed/pairs_train_valid.jsonl`: 검증용 질의-passage 쌍 (자동 생성)
- `data/processed/prec_fallback_passages.jsonl`: 판례 fallback passage (자동 생성)

### 1-4. 코퍼스 생성

모든 passage를 하나의 파일로 병합합니다:

```bash
# Fallback passage가 있는 경우
poetry run python -m lex_dpr.data_processing.merge_corpus \
  --law data/processed/law_passages.jsonl \
  --admin data/processed/admin_passages.jsonl \
  --prec data/processed/prec_fallback_passages.jsonl \
  --out data/processed/merged_corpus.jsonl

# Fallback passage가 없는 경우
poetry run python -m lex_dpr.data_processing.merge_corpus \
  --law data/processed/law_passages.jsonl \
  --admin data/processed/admin_passages.jsonl \
  --out data/processed/merged_corpus.jsonl
```

**생성되는 파일:**
- `data/processed/merged_corpus.jsonl`: 모든 passage가 병합된 최종 코퍼스

### 1-5. 데이터 구조

#### Passage 형식

```json
{
  "id": "LAW_법령ID_제xx조_항_호",
  "parent_id": "법령ID",
  "type": "law",
  "text": "조문 내용...",
  "law_name": "법령명",
  "article": "제xx조"
}
```

#### Pair 형식

```json
{
  "query_text": "질의 텍스트",
  "positive_passages": ["LAW_...", "ADM_..."],
  "hard_negatives": ["LAW_...", "ADM_..."],
  "meta": {
    "type": "law",
    "law_name": "법령명",
    "article": "제xx조"
  }
}
```

---

## 2. 학습 명령어

### 2-1. Preset 명령어 요약

최초 학습 시, preset 명령어를 사용하여 fine-tuniung을 즉시 시작할 수 있습니다.

```bash
# 기본: 설정 생성 + 스윕 생성 + 에이전트 실행
poetry run lex-dpr sweep preset
```

**Preset에 포함된 하이퍼파라미터:**
- **학습률**: 5e-7 ~ 1e-4 (log_uniform)
- **Temperature**: 0.01 ~ 0.3 (uniform)
- **Weight Decay**: 0.01 ~ 0.5 (uniform)
- **Warmup Ratio**: 0.0 ~ 0.2 (uniform)
- **Gradient Accumulation Steps**: [16, 32, 64]
- **Gradient Clipping**: 1.0 ~ 20.0 (uniform)
- **LoRA rank**: [4, 8, 16]
- **LoRA alpha**: [8, 16, 32]
- **LoRA dropout**: 0.05 ~ 0.1 (uniform)
- **배치 크기**: [32, 64, 128, 256]
- **데이터 증폭**: [0, 1, 2]
- **Hard Negative 비율**: 0.2 ~ 0.6 (uniform)
- **Validation Negative 샘플링**: [16, 32, 64]
- **기본 모델**: [ko-simcse, bge-m3-ko, multilingual-minilm, multilingual-e5-small]
- **시퀀스 길이**: [256, 384, 512]

**Preset 옵션:**

| 옵션                 | 설정 파일 생성 | 스윕 생성 | 에이전트 실행 | 설명                                           |
| -------------------- | -------------- | --------- | ------------- | ---------------------------------------------- |
| **기본 (옵션 없음)** | ✅              | ✅         | ✅             | 모든 단계 실행                                 |
| **`--no-run-agent`** | ✅              | ✅         | ❌             | 설정 파일 생성 + 스윕 생성 (에이전트는 나중에) |
| **`--no-run`**       | ✅              | ❌         | ❌             | 설정 파일만 생성                               |

**사용 예시:**

```bash
# 1. 모든 단계 실행 (기본)
poetry run lex-dpr sweep preset
# → 설정 파일 생성 + 스윕 생성 + 에이전트 실행

# 2. 설정 파일만 생성
poetry run lex-dpr sweep preset --no-run
# → 설정 파일만 생성 (스윕 생성 안 함, 에이전트 실행 안 함)
# → 나중에: poetry run lex-dpr sweep --config configs/sweep.yaml

# 3. 스윕까지 생성 (에이전트는 나중에)
poetry run lex-dpr sweep preset --no-run-agent
# → 설정 파일 생성 + 스윕 생성 (에이전트 실행 안 함)
# → 나중에: poetry run lex-dpr sweep agent --config configs/sweep.yaml
```

### 2-2. 커스텀 설정 방법 & Agent 사용 (Sweep 재사용)

#### 설정 파일 생성

Preset에서 설정 파일만 생성한 뒤, 직접 값의 범위를 수정하여 fine-tuning의 범위를 제한할 수 있습니다.

```bash
# 넉넉한 범위의 템플릿 생성 (실행하지 않음)
poetry run lex-dpr sweep preset --output configs/my_sweep.yaml --no-run
```

#### 설정 파일 편집

생성된 설정 파일 중 각 하이퍼파라미터 범위를 수정합니다.
- 학습 시간 절감 및 탐색 범위 최적화를 위해 method는 bayes로 고정합니다.
- metric은 model 최적화의 목표입니다. (기본값: NACG@10 최대화)
- 각 parameter는 이산형이거나 연속형입니다.
  - 이산형 : 선택 가능한 값을 array로 설정
  - 연속형 : 선택 가능한 값의 범위와 분포를 설정 (ex. distribution: uniform은 균등분포를 의미)

```yaml
# configs/my_sweep.yaml
method: bayes  # grid, random, bayes 중 선택

metric:
  name: eval/ndcg_at_10
  goal: maximize

parameters:
  trainer.lr:
    distribution: log_uniform_values
    min: 0.000001
    max: 0.0001
  
  trainer.temperature:
    distribution: uniform
    min: 0.01
    max: 0.2

    ...

fixed:
  trainer.epochs: 3
  data.pairs: data/processed/pairs_train.jsonl
  data.passages: data/processed/merged_corpus.jsonl
```

#### 스윕 생성 및 실행

설정 파일을 사용하여 sweep이라는 단위로 최적화를 진행합니다.

```bash
# 설정 파일로 스윕 생성 및 실행
poetry run lex-dpr sweep --config configs/my_sweep.yaml

# 스윕만 생성 (에이전트는 나중에 실행)
poetry run lex-dpr sweep --config configs/my_sweep.yaml --no-run-agent
```

하나의 sweep을 여러 agent가 실행하거나, 중단 후 재실행할 수 있습니다.

```bash
# 설정 파일에서 스윕 ID 읽어서 실행
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml

# 스윕 ID를 직접 지정
poetry run lex-dpr sweep agent <sweep-id>

# 실행 횟수 제한
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10

# 시간 제한 설정 (예: 1시~8시에만 실행)
poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --time-window "1-8"
```

**Agent 사용 시나리오:**
- 여러 머신에서 동시에 실행하여 병렬 탐색
- 여러 날짜에 나눠서 실행 (시간 제한 옵션 사용)
- 특정 횟수만 실행하고 중단

### 2-3. 로그 저장, 확인 방법 & 체크포인트 경로

#### WandB 로그 저장 위치

모든 로그는 자동으로 오프라인 모드로 실행되어 로컬에만 저장됩니다:

- **로컬 저장 위치**: `wandb/run-{timestamp}-{run_id}/`
- **각 run의 로그**: 해당 디렉토리에 JSONL 형식으로 저장
- **메트릭**: `wandb/run-{timestamp}-{run_id}/wandb-metadata.json`
- **설정**: `wandb/run-{timestamp}-{run_id}/config.yaml`

#### 로그 파일 구조

```
wandb/
├── run-20240101_120000-abc123/  # 각 run의 디렉토리
│   ├── wandb-metadata.json      # 메타데이터
│   ├── config.yaml               # 하이퍼파라미터 설정
│   ├── wandb-events.jsonl        # 메트릭 로그 (JSONL 형식)
│   └── files/                    # 기타 파일들
├── run-20240101_130000-def456/
│   └── ...
└── latest-run -> run-20240101_120000-abc123/  # 최신 run 심볼릭 링크
```

#### 로그 확인 방법

**라인차트로 확인하기 (권장):**

상황에 따라 수많은 지표의 추이를 확인해야하는 경우가 있습니다.
따라서 로그를 라인차트로 시각화하여 확인하는 것이 가장 효과적입니다.
WandB 로그를 Python 스크립트나 도구를 사용하여 라인차트로 시각화할 수 있습니다:

```bash
# 최신 run 확인
ls -la wandb/latest-run/

# 특정 run의 메트릭 확인 (JSONL 형식)
cat wandb/run-20240101_120000-abc123/wandb-events.jsonl | jq

# 설정 확인
cat wandb/run-20240101_120000-abc123/config.yaml
```

**라인차트 확인의 장점:**
- 학습 진행 과정을 한눈에 파악 가능
- Train loss와 Valid loss의 수렴 패턴 확인 용이
- 메트릭 추이를 시간에 따라 추적 가능
- 여러 run을 비교하여 최적 하이퍼파라미터 선택 용이

#### 학습 성패 판단 방법

학습 종료 후 로그를 확인하여 모델 학습 성패를 판단할 수 있습니다:

**1. WandB 메트릭 확인**

```bash
# 특정 run의 메트릭 확인 (JSONL 형식)
cat wandb/run-20240101_120000-abc123/wandb-events.jsonl | jq

# 평가 메트릭만 필터링
cat wandb/run-20240101_120000-abc123/wandb-events.jsonl | jq 'select(.eval)'

# 최종 평가 메트릭 확인
cat wandb/run-20240101_120000-abc123/wandb-events.jsonl | jq 'select(.eval) | .eval' | tail -20
```

**주요 평가 메트릭:**
- `eval/ndcg_at_10`: NDCG@10 (높을수록 좋음)
- `eval/recall_at_10`: Recall@10 (높을수록 좋음)
- `eval/ndcg_at_5`: NDCG@5 (높을수록 좋음)
- `eval/recall_at_5`: Recall@5 (높을수록 좋음)

**Top-K 값 선택 참고:**
- Top-K 값은 LLM에 전달되는 passage의 수에 따라 결정됩니다.
- LLM의 컨텍스트 윈도우 크기와 실제 사용 시나리오를 고려하여 적절한 K 값을 선택해야 합니다.
- 예를 들어, LLM에 최대 5개의 passage만 전달할 수 있다면 `k_values: [1, 3, 5]`로 설정하는 것이 적절합니다.
- 기본 설정은 `k_values: [1, 3, 5, 10]`이며, 필요에 따라 조정할 수 있습니다.

**2. 학습 상태 확인**

```bash
# wandb-metadata.json에서 상태 확인
cat wandb/run-20240101_120000-abc123/wandb-metadata.json | jq '.state'
```

**상태 값:**
- `finished`: 정상 완료
- `failed`: 실패
- `running`: 실행 중
- `crashed`: 크래시

**3. 체크포인트 확인**

```bash
# 체크포인트 디렉토리 확인
ls -la checkpoint/lexdpr/bi_encoder/

# 최신 체크포인트 확인
ls -lt checkpoint/lexdpr/bi_encoder/ | head -10
```

**성공적인 학습의 기준:**
- ✅ `wandb-metadata.json`의 `state`가 `finished`
- ✅ 평가 메트릭이 정상적으로 기록됨 (NDCG@10, Recall@10 등)
- ✅ 체크포인트 파일이 생성됨 (`checkpoint-{step}/` 디렉토리 존재)
- ✅ **Loss의 수렴과 지표의 개선 정도를 함께 확인**: Train loss와 Valid loss가 발산하지 않고 지속적으로 수렴하면서, 동시에 평가 지표(NDCG@10, Recall@10 등)도 함께 개선되어야 합니다. Loss만 감소하고 지표가 개선되지 않거나, 반대로 지표만 개선되고 loss가 발산하는 경우는 문제가 있을 수 있습니다.
- ✅ **Train loss와 Valid loss가 발산하지 않고 지속적으로 수렴**: 학습이 원활하게 진행되고 있음을 의미합니다. 라인차트에서 두 loss가 함께 감소하며 수렴하는 패턴을 보여야 합니다.

**실패한 학습의 징후:**
- ❌ `state`가 `failed` 또는 `crashed`
- ❌ 평가 메트릭이 기록되지 않음
- ❌ 체크포인트가 생성되지 않음
- ❌ 메트릭 값이 NaN이거나 비정상적으로 낮음
- ❌ **Train loss와 Valid loss가 발산하거나 과도한 차이를 보임**: Overfitting 또는 학습 불안정을 의미합니다. 라인차트에서 Valid loss가 Train loss보다 크게 증가하거나 발산하는 패턴을 보이면 문제가 있습니다.
- ❌ **Loss는 수렴하지만 평가 지표가 개선되지 않음**: 학습이 진행되고 있지만 실제 성능 향상이 없는 경우입니다.

**모델 개선 판단 기준:**
- 이전 버전의 모델보다 개선된 모델의 결정은 **동일한 test dataset에 대한 지표 개선**을 기준으로 합니다.
- 서로 다른 test dataset으로 평가한 결과를 비교하면 안 됩니다.
- 동일한 test dataset에서 NDCG@10, Recall@10 등의 지표가 이전 모델보다 향상되었을 때만 개선된 것으로 판단합니다.
- 여러 run을 비교할 때도 동일한 평가 데이터셋(`eval_pairs`)을 사용했는지 확인해야 합니다.

#### 체크포인트 경로

모델 체크포인트는 다음 위치에 저장됩니다:

```
{out_dir}/bi_encoder/checkpoint-{step}/
```

기본값은 `checkpoint/lexdpr/bi_encoder/`입니다.

**학습 설정 파일:**
- `{out_dir}/bi_encoder/training_config.json`: 학습에 사용된 설정 저장

---

## 부록. 전체 파이프라인 요약 및 로그 요약

### 전체 파이프라인 예시

모든 단계를 한 번에 실행하는 스크립트 예시:

```bash
#!/bin/bash

# [1] Passage 생성
poetry run python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/admin_rules --out-admin data/processed/admin_passages.jsonl --glob "**/*.json"
poetry run python -m lex_dpr.data_processing.preprocess_auto \
  --src-dir data/laws --out-law data/processed/law_passages.jsonl --glob "**/*.json"

# [2] 질의-passage 쌍 생성
poetry run python -m lex_dpr.data_processing.make_pairs \
  --law data/processed/law_passages.jsonl \
  --admin data/processed/admin_passages.jsonl \
  --prec-json-dir data/precedents \
  --out data/processed/pairs_train.jsonl \
  --use-admin-for-prec --hn_per_q 32 --max-positives-per-prec 5 --max-workers 8

# [3] Passage 코퍼스 병합
poetry run python -m lex_dpr.data_processing.merge_corpus \
  --law data/processed/law_passages.jsonl \
  --admin data/processed/admin_passages.jsonl \
  --prec data/processed/prec_fallback_passages.jsonl \
  --out data/processed/merged_corpus.jsonl

# [4] 학습 시작
poetry run lex-dpr sweep preset
```

### 데이터 저장 위치 요약

| 데이터 타입               | 파일 경로                                     | 설명                              |
| ------------------------- | --------------------------------------------- | --------------------------------- |
| **원시 판례**             | `data/precedents/*.json`                      | 판례 원본 JSON                    |
| **법령 Passage**          | `data/processed/law_passages.jsonl`           | 법령에서 추출한 passage           |
| **행정규칙 Passage**      | `data/processed/admin_passages.jsonl`         | 행정규칙에서 추출한 passage       |
| **판례 Fallback Passage** | `data/processed/prec_fallback_passages.jsonl` | 판례에서 생성한 fallback passage  |
| **학습용 쌍**             | `data/processed/pairs_train.jsonl`            | 학습에 사용할 질의-passage 쌍     |
| **검증용 쌍**             | `data/processed/pairs_train_valid.jsonl`      | 검증에 사용할 질의-passage 쌍     |
| **병합 코퍼스**           | `data/processed/merged_corpus.jsonl`          | 모든 passage가 병합된 최종 코퍼스 |

### 로그 저장 위치 요약

| 항목                | 위치                                        | 설명                                    |
| ------------------- | ------------------------------------------- | --------------------------------------- |
| **WandB 로그**      | `wandb/run-{timestamp}-{run_id}/`           | 로컬에 저장되는 모든 학습 로그          |
| **모델 체크포인트** | `{out_dir}/bi_encoder/checkpoint-{step}/`   | 기본값: `checkpoint/lexdpr/bi_encoder/` |
| **학습 설정**       | `{out_dir}/bi_encoder/training_config.json` | 학습에 사용된 설정 저장                 |

