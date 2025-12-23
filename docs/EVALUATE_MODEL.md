# 모델 평가 가이드

## 현재 다운로드한 모델 평가하기

패키지에 포함된 모델 또는 다운로드한 모델의 성능을 평가할 수 있습니다.

## 기본 평가

### CLI 사용

```bash
# 기본 평가 (패키지에 포함된 모델 사용)
poetry run lex-dpr eval

# 특정 모델 경로 지정
poetry run lex-dpr eval --model lex_dpr/models/default_model

# 평가 데이터 및 k 값 지정
poetry run lex-dpr eval \
    --model lex_dpr/models/default_model \
    --eval-pairs data/pairs_eval.jsonl \
    --passages data/processed/merged_corpus.jsonl \
    --k-values 1 3 5 10 20
```

### Python 스크립트 사용

```bash
python scripts/evaluate.py \
    --model lex_dpr/models/default_model \
    --eval-pairs data/pairs_eval.jsonl \
    --passages data/processed/merged_corpus.jsonl \
    --k-values 1 3 5 10 20
```

## 평가 메트릭

평가 결과에는 다음 메트릭이 포함됩니다:

- **MRR@k** (Mean Reciprocal Rank): 평균 역순위
- **NDCG@k** (Normalized Discounted Cumulative Gain): 정규화된 누적 이득
- **MAP@k** (Mean Average Precision): 평균 정밀도
- **Recall@k**: 재현율
- **Precision@k**: 정밀도
- **Accuracy@k**: 정확도

## 상세 분석

### 상세 리포트 생성

```bash
poetry run lex-dpr eval \
    --model lex_dpr/models/default_model \
    --detailed \
    --report eval_report.txt
```

상세 리포트에는 다음이 포함됩니다:
- 쿼리별 성능 분석
- 소스별 성능 분석 (법령/행정규칙/판례)
- 실패 케이스 분석
- 쿼리/Passage 길이별 성능 분석

## WandB에 결과 로깅

```bash
poetry run lex-dpr eval \
    --model lex_dpr/models/default_model \
    --wandb \
    --wandb-project lexdpr-eval \
    --wandb-name model_trim-sweep-12-eval
```

## 여러 모델 비교

```bash
poetry run lex-dpr eval \
    --compare-models \
        lex_dpr/models/default_model \
        checkpoint/lexdpr/bi_encoder \
        jhgan/ko-sroberta-multitask \
    --compare-output model_comparison.txt
```

## 평가 데이터 준비

평가를 위해서는 다음 파일이 필요합니다:

1. **Passage 코퍼스**: `data/processed/merged_corpus.jsonl`
   - 모든 passage가 포함된 JSONL 파일
   - 형식: `{"id": "passage_id", "text": "passage_text"}`

2. **평가 쌍**: `data/pairs_eval.jsonl`
   - 질의와 positive passage ID 쌍
   - 형식: `{"query": "질의 텍스트", "positive": ["passage_id1", "passage_id2", ...]}`

## 예시 출력

```
============================================================
LexDPR 평가 결과: lex_dpr/models/default_model
============================================================

MRR 메트릭:
  mrr_at_1:  0.1234
  mrr_at_3:  0.2345
  mrr_at_5:  0.3456
  mrr_at_10: 0.4567

NDCG 메트릭:
  ndcg_at_1:  0.1234
  ndcg_at_3:  0.2345
  ndcg_at_5:  0.3456
  ndcg_at_10: 0.4567

Recall 메트릭:
  recall_at_1:  0.1234
  recall_at_3:  0.2345
  recall_at_5:  0.3456
  recall_at_10: 0.4567
```

## 주의사항

1. **모델 설정**: 평가 시 학습 시 사용된 `max_seq_length`와 동일하게 설정되어야 합니다.
   - 패키지에 포함된 모델은 자동으로 `training_config.json` 또는 `config.py`의 `DEFAULT_MAX_LEN`을 사용합니다.

2. **템플릿 모드**: 학습 시 사용된 템플릿 모드(BGE/NONE)와 동일하게 설정해야 합니다.
   - 기본값: `--template bge`

3. **메모리**: 대용량 코퍼스 평가 시 배치 크기를 조절하세요.
   - 기본값: `--batch-size 16`

