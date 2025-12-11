#!/usr/bin/env bash
set -e
cd "/mnt/data/LexDPR_real2"

echo "[0/5] 판례 수집: precedents → data/precedents/*.json"
# 워커 수(default 4) 지연 시간 조절
poetry run python -m lex_dpr.crawler.crawl_precedents --output data/precedents --max-pages 3000 --max-workers 8 --delay 0.5

echo "[1/5] 전처리: statutes → corpus.jsonl"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/admin_rules --out-admin data/processed/admin_passages.jsonl --glob "**/*.json"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/laws   --out-law data/processed/law_passages.jsonl   --glob "**/*.json"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/precedents --out-prec data/processed/prec_passages.jsonl --glob "**/*.json"


echo "[2/5] 질의-passage 쌍 생성: corpus → pairs_train.jsonl"
poetry run python -m lex_dpr.data_processing.make_pairs --law data/processed/law_passages.jsonl --prec-json-dir data/precedents --out data/processed/pairs_train.jsonl --hn_per_q 10 --max-positives-per-prec 5
poetry run python -m lex_dpr.data_processing.make_pairs --law data/processed/law_passages.jsonl --admin data/processed/admin_passages.jsonl --prec-json-dir data/precedents --out data/processed/pairs_train.jsonl --use-admin-for-prec --hn_per_q 32 --max-positives-per-prec 5


echo "[3/5] passage 코퍼스 병합: corpus → merged_corpus.jsonl"
poetry run python -m lex_dpr.data_processing.merge_corpus --law   data/processed/law_passages.jsonl --admin data/processed/admin_passages.jsonl --out   data/processed/merged_corpus.jsonl

echo "[3/5] 학습 시작"
poetry run python entrypoint_train.py 


echo "[4/5] 임베딩 생성: corpus → embeds"
# Passages 임베딩 추출
poetry run python entrypoint_embed.py \
  --model checkpoint/lexdpr/bi_encoder \
  --input data/processed/law_passages.jsonl \
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


echo "[5/5] 인덱스 빌드 & 평가"
python scripts/build_index.py --input checkpoint --output index --factory Flat --metric dot
python scripts/evaluate.py --index_dir index --queries data/queries/queries.jsonl --top_k 10


echo '[6/5] 서버 시작'

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