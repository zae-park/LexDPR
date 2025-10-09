#!/usr/bin/env bash
set -e
cd "/mnt/data/LexDPR_real2"

echo "[1/5] 전처리: statutes → corpus.jsonl"
python scripts/preprocess_acts.py --input data/statutes --output data/processed/corpus.jsonl

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
