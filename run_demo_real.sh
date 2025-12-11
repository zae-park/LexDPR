#!/usr/bin/env bash
set -e
cd "/mnt/data/LexDPR_real2"

echo "[1/9] 판례 수집: precedents → data/precedents/*.json"
# 워커 수(default 4) 지연 시간 조절
poetry run python -m lex_dpr.crawler.crawl_precedents --output data/precedents --max-pages 3000 --max-workers 8 --delay 0.5

echo "[2/9] 전처리: statutes → corpus.jsonl"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/admin_rules --out-admin data/processed/admin_passages.jsonl --glob "**/*.json"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/laws   --out-law data/processed/law_passages.jsonl   --glob "**/*.json"
# 주의: 판례는 make_pairs에서 자동으로 처리되므로 별도 전처리 불필요
# poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/precedents --out-prec data/processed/prec_passages.jsonl --glob "**/*.json"


echo "[3/9] 질의-passage 쌍 생성: corpus → pairs_train.jsonl"
# Fallback passage도 자동으로 생성됨 (data/processed/prec_fallback_passages.jsonl)
poetry run python -m lex_dpr.data_processing.make_pairs --law data/processed/law_passages.jsonl --admin data/processed/admin_passages.jsonl --prec-json-dir data/precedents --out data/processed/pairs_train.jsonl --use-admin-for-prec --hn_per_q 32 --max-positives-per-prec 5 --max-workers 8


echo "[4/9] passage 코퍼스 병합: corpus → merged_corpus.jsonl"
# Fallback passage가 있으면 포함 (make_pairs에서 자동 생성됨)
if [ -f "data/processed/prec_fallback_passages.jsonl" ]; then
    poetry run python -m lex_dpr.data_processing.merge_corpus --law data/processed/law_passages.jsonl --admin data/processed/admin_passages.jsonl --prec data/processed/prec_fallback_passages.jsonl --out data/processed/merged_corpus.jsonl
else
    poetry run python -m lex_dpr.data_processing.merge_corpus --law data/processed/law_passages.jsonl --admin data/processed/admin_passages.jsonl --out data/processed/merged_corpus.jsonl
fi

echo "[5/9] Base Model 평가 (학습 전)"
# 평가용 쌍 파일 확인 (make_pairs에서 생성된 validation set 사용)
EVAL_PAIRS="data/processed/pairs_train_valid.jsonl"
if [ ! -f "$EVAL_PAIRS" ]; then
    echo "  경고: 평가용 쌍 파일을 찾을 수 없습니다: $EVAL_PAIRS"
    echo "  Base Model 평가를 건너뜁니다."
else
    # Base model 평가 (ko-simcse 또는 bge-m3-ko 등)
    # 필요에 따라 모델 경로를 변경하세요: jhgan/ko-sroberta-multitask, dragonkue/BGE-m3-ko
    BASE_MODEL="jhgan/ko-sroberta-multitask"  # ko-simcse base model
    echo "  Base Model: $BASE_MODEL"
    poetry run lex-dpr eval \
      --model "$BASE_MODEL" \
      --passages data/processed/merged_corpus.jsonl \
      --eval-pairs "$EVAL_PAIRS" \
      --k-values 1 3 5 10 20 \
      --template bge \
      --batch-size 16 \
      --output data/processed/base_model_eval.json \
      --report data/processed/base_model_eval_report.txt \
      --wandb \
      --wandb-project lexdpr-eval \
      --wandb-name "base-model-eval"
fi

echo "[6/9] 학습 시작"
poetry run python entrypoint_train.py 


echo "[7/9] 임베딩 생성: corpus → embeds"
# Passages 임베딩 추출 (전체 corpus 사용: 법령 + 행정규칙 + Fallback 판례)
poetry run python entrypoint_embed.py \
  --model checkpoint/lexdpr/bi_encoder \
  --input data/processed/merged_corpus.jsonl \
  --outdir embeds \
  --prefix passages \
  --type passage

# # Queries 임베딩 추출
# poetry run python entrypoint_embed.py \
#   --model checkpoint/lexdpr/bi_encoder \
#   --input data/queries/queries.jsonl \
#   --outdir embeds \
#   --prefix queries \
#   --type query


echo "[8/9] 인덱스 빌드 & 평가"
python scripts/build_index.py --input checkpoint --output index --factory Flat --metric dot
python scripts/evaluate.py --index_dir index --queries data/queries/queries.jsonl --top_k 10


echo "[9/9] 서버 시작"

# # API 서버 실행
# poetry run python entrypoint_api.py \
#   --model checkpoint/lexdpr/bi_encoder \
#   --host 0.0.0.0 \
#   --port 8000

# PEFT 모델 사용 시
poetry run python entrypoint_api.py \
  --model checkpoint/lexdpr/bi_encoder \
  --peft-adapter checkpoint/lexdpr/bi_encoder \
  --host 0.0.0.0 \
  --port 8000