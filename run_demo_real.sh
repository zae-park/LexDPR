#!/usr/bin/env bash
set -e
cd "/mnt/data/LexDPR_real2"

echo "[0/5] 판례 수집: precedents → data/precedents/*.json"
poetry run python lex_dpr/crawler/crawl_precedents.py --output data/precedents --max-pages 3000 --max-workers 8 --delay 0.5    # 워커 수(default 4) 지연 시간 조절


echo "[1/5] 전처리: statutes → corpus.jsonl"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/admin_rules --out-admin data/processed/admin_passages.jsonl --glob "**/*.json"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/laws   --out-law data/processed/law_passages.jsonl   --glob "**/*.json"
poetry run python -m lex_dpr.data_processing.preprocess_auto --src-dir data/precedents --out-prec data/processed/prec_passages.jsonl --glob "**/*.json"




poetry run python -m lex_dpr.data_processing.merge_corpus   --law   data/processed/law_passages.jsonl   --admin data/processed/admin_passages.jsonl   --out   data/processed/merged_corpus.jsonl
poetry run python -m lex_dpr.data_processing.make_pairs --law   data/processed/law_passages.jsonl --admin data/processed/admin_passages.jsonl --out   data/processed/pairs_train.jsonl

# poetry run python -m lex_dpr.data_processing.make_pairs --law   data/processed/law_passages.jsonl --admin data/processed/admin_passages.jsonl --prec  data/processed/prec_passages.jsonl --out   data/processed/pairs_train.jsonl


echo "[2/5] 전처리: no_action_letters 병합"
python scripts/preprocess_acts.py --input data/no_action_letters --output data/processed/tmp.jsonl
cat data/processed/tmp.jsonl >> data/processed/corpus.jsonl
rm data/processed/tmp.jsonl

echo "[3/5] Passage 인코딩"
python scripts/encode_passages.py --model sentence-transformers/all-MiniLM-L6-v2 --input data/processed/corpus.jsonl --outdir checkpoint --batch_size 64 --pooling mean

echo "[4/5] Query 인코딩"
python scripts/encode_queries.py --model sentence-transformers/all-MiniLM-L6-v2 --queries data/queries/queries.jsonl --outdir checkpoint --batch_size 64 --pooling mean

echo "[5/5] 인덱스 빌드 & 평가"
python scripts/build_index.py --input checkpoint --output index --factory Flat --metric dot
python scripts/evaluate.py --index_dir index --queries data/queries/queries.jsonl --top_k 10
