#!/usr/bin/env bash
set -e

# 조문, 항, 호 단위로 passage 생성 및 pre-trained 모델 평가 스크립트
# 
# 사용법:
#   bash scripts/eval_chunk_units.sh
#   또는
#   bash scripts/eval_chunk_units.sh --models jhgan/ko-sroberta-multitask dragonkue/BGE-m3-ko

# 기본 설정
LAW_SRC_DIR="${LAW_SRC_DIR:-data/laws}"
ADMIN_SRC_DIR="${ADMIN_SRC_DIR:-data/admin_rules}"
PREC_JSON_DIR="${PREC_JSON_DIR:-data/precedents}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-data/eval_chunk_units}"
EVAL_PAIRS="${EVAL_PAIRS:-data/processed/pairs_train_valid.jsonl}"

# 평가할 모델들 (기본값)
MODELS=("jhgan/ko-sroberta-multitask" "dragonkue/BGE-m3-ko")

# 커맨드라인 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --law-src-dir)
            LAW_SRC_DIR="$2"
            shift 2
            ;;
        --admin-src-dir)
            ADMIN_SRC_DIR="$2"
            shift 2
            ;;
        --prec-json-dir)
            PREC_JSON_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --eval-pairs)
            EVAL_PAIRS="$2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: bash scripts/eval_chunk_units.sh [--models MODEL1 MODEL2 ...] [--law-src-dir DIR] [--output-dir DIR] [--eval-pairs FILE]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Passage Chunk 단위별 Pre-trained 모델 평가"
echo "=========================================="
echo ""
echo "설정:"
echo "  법령 소스 디렉토리: $LAW_SRC_DIR"
echo "  행정규칙 소스 디렉토리: $ADMIN_SRC_DIR"
echo "  판례 JSON 디렉토리: $PREC_JSON_DIR"
echo "  출력 디렉토리: $OUTPUT_BASE_DIR"
echo "  평가 쌍 파일: $EVAL_PAIRS"
echo "  평가 모델: ${MODELS[@]}"
echo ""

# 평가 쌍 파일 확인
if [ ! -f "$EVAL_PAIRS" ]; then
    echo "⚠️  경고: 평가 쌍 파일을 찾을 수 없습니다: $EVAL_PAIRS"
    echo "   먼저 make_pairs를 실행하여 평가 쌍을 생성하세요."
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_BASE_DIR"

# 결과 저장용
RESULTS_DIR="$OUTPUT_BASE_DIR/results"
mkdir -p "$RESULTS_DIR"

# ==========================================
# 1. 항 단위로 passage 생성 (기본값)
# ==========================================
echo "[1/6] 항 단위 passage 생성 중..."
PARAGRAPH_DIR="$OUTPUT_BASE_DIR/paragraph"
mkdir -p "$PARAGRAPH_DIR"

poetry run python -m lex_dpr.data_processing.preprocess_auto \
    --src-dir "$ADMIN_SRC_DIR" \
    --out-admin "$PARAGRAPH_DIR/admin_passages.jsonl" \
    --glob "**/*.json" || true

poetry run python -m lex_dpr.data_processing.preprocess_auto \
    --src-dir "$LAW_SRC_DIR" \
    --out-law "$PARAGRAPH_DIR/law_passages.jsonl" \
    --glob "**/*.json" || true

# 코퍼스 병합
if [ -f "$PARAGRAPH_DIR/admin_passages.jsonl" ] && [ -f "$PARAGRAPH_DIR/law_passages.jsonl" ]; then
    poetry run python -m lex_dpr.data_processing.merge_corpus \
        --law "$PARAGRAPH_DIR/law_passages.jsonl" \
        --admin "$PARAGRAPH_DIR/admin_passages.jsonl" \
        --out "$PARAGRAPH_DIR/merged_corpus.jsonl"
elif [ -f "$PARAGRAPH_DIR/law_passages.jsonl" ]; then
    poetry run python -m lex_dpr.data_processing.merge_corpus \
        --law "$PARAGRAPH_DIR/law_passages.jsonl" \
        --out "$PARAGRAPH_DIR/merged_corpus.jsonl"
fi

echo "✅ 항 단위 passage 생성 완료: $PARAGRAPH_DIR/merged_corpus.jsonl"
echo ""

# ==========================================
# 2. 호 단위로 passage 생성
# ==========================================
echo "[2/6] 호 단위 passage 생성 중..."
ITEM_DIR="$OUTPUT_BASE_DIR/item"
mkdir -p "$ITEM_DIR"

poetry run python -m lex_dpr.data_processing.preprocess_auto \
    --src-dir "$ADMIN_SRC_DIR" \
    --out-admin "$ITEM_DIR/admin_passages.jsonl" \
    --glob "**/*.json" || true

poetry run python -m lex_dpr.data_processing.preprocess_auto \
    --src-dir "$LAW_SRC_DIR" \
    --out-law "$ITEM_DIR/law_passages.jsonl" \
    --include-items \
    --glob "**/*.json" || true

# 코퍼스 병합
if [ -f "$ITEM_DIR/admin_passages.jsonl" ] && [ -f "$ITEM_DIR/law_passages.jsonl" ]; then
    poetry run python -m lex_dpr.data_processing.merge_corpus \
        --law "$ITEM_DIR/law_passages.jsonl" \
        --admin "$ITEM_DIR/admin_passages.jsonl" \
        --out "$ITEM_DIR/merged_corpus.jsonl"
elif [ -f "$ITEM_DIR/law_passages.jsonl" ]; then
    poetry run python -m lex_dpr.data_processing.merge_corpus \
        --law "$ITEM_DIR/law_passages.jsonl" \
        --out "$ITEM_DIR/merged_corpus.jsonl"
fi

echo "✅ 호 단위 passage 생성 완료: $ITEM_DIR/merged_corpus.jsonl"
echo ""

# ==========================================
# 3. 조문 단위로 passage 생성
# ==========================================
echo "[3/6] 조문 단위 passage 생성 중..."
ARTICLE_DIR="$OUTPUT_BASE_DIR/article"
mkdir -p "$ARTICLE_DIR"

# 조문 단위는 항이 없을 때만 생성되므로, 별도 처리
# 행정규칙은 이미 조문 단위이므로 그대로 사용
poetry run python -m lex_dpr.data_processing.preprocess_auto \
    --src-dir "$ADMIN_SRC_DIR" \
    --out-admin "$ARTICLE_DIR/admin_passages.jsonl" \
    --glob "**/*.json" || true

# 법령은 조문 단위로 강제 생성하는 스크립트 필요
# 일단 항이 없는 경우만 포함 (실제로는 조문 전체를 합치는 로직 필요)
poetry run python -m lex_dpr.data_processing.preprocess_auto \
    --src-dir "$LAW_SRC_DIR" \
    --out-law "$ARTICLE_DIR/law_passages.jsonl" \
    --glob "**/*.json" || true

# 조문 단위로 변환하는 Python 스크립트 실행
poetry run python -c "
from pathlib import Path
import json
from lex_dpr.utils.io import read_jsonl, write_jsonl

# 법령 passage를 조문 단위로 병합
law_passages = list(read_jsonl('$ARTICLE_DIR/law_passages.jsonl'))
article_dict = {}

for p in law_passages:
    article_key = p.get('article', '')
    if not article_key:
        continue
    
    if article_key not in article_dict:
        article_dict[article_key] = {
            'id': p.get('parent_id') or p.get('id'),
            'parent_id': p.get('parent_id') or p.get('id'),
            'type': p.get('type', '법령'),
            'law_id': p.get('law_id'),
            'law_name': p.get('law_name'),
            'article': article_key,
            'effective_date': p.get('effective_date'),
            'text': p.get('text', ''),
        }
    else:
        # 같은 조문의 다른 항/호를 합침
        article_dict[article_key]['text'] += '\n' + p.get('text', '')

# 조문 단위 passage 저장
article_passages = list(article_dict.values())
write_jsonl('$ARTICLE_DIR/law_passages_article.jsonl', article_passages)
print(f'조문 단위 passage 생성: {len(article_passages)}개')
"

# 코퍼스 병합
if [ -f "$ARTICLE_DIR/admin_passages.jsonl" ] && [ -f "$ARTICLE_DIR/law_passages_article.jsonl" ]; then
    poetry run python -m lex_dpr.data_processing.merge_corpus \
        --law "$ARTICLE_DIR/law_passages_article.jsonl" \
        --admin "$ARTICLE_DIR/admin_passages.jsonl" \
        --out "$ARTICLE_DIR/merged_corpus.jsonl"
elif [ -f "$ARTICLE_DIR/law_passages_article.jsonl" ]; then
    poetry run python -m lex_dpr.data_processing.merge_corpus \
        --law "$ARTICLE_DIR/law_passages_article.jsonl" \
        --out "$ARTICLE_DIR/merged_corpus.jsonl"
fi

echo "✅ 조문 단위 passage 생성 완료: $ARTICLE_DIR/merged_corpus.jsonl"
echo ""

# ==========================================
# 4. 각 chunk 단위별로 모델 평가
# ==========================================
echo "[4/6] 각 chunk 단위별 모델 평가 시작..."
echo ""

# 평가 결과 요약
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Passage Chunk 단위별 Pre-trained 모델 평가 결과" > "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "평가 일시: $(date)" >> "$SUMMARY_FILE"
echo "평가 쌍 파일: $EVAL_PAIRS" >> "$SUMMARY_FILE"
echo "평가 모델: ${MODELS[@]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for CHUNK_TYPE in "paragraph" "item" "article"; do
    CHUNK_DIR="$OUTPUT_BASE_DIR/$CHUNK_TYPE"
    CORPUS_FILE="$CHUNK_DIR/merged_corpus.jsonl"
    
    if [ ! -f "$CORPUS_FILE" ]; then
        echo "⚠️  경고: $CORPUS_TYPE 코퍼스 파일을 찾을 수 없습니다: $CORPUS_FILE"
        continue
    fi
    
    echo "----------------------------------------" >> "$SUMMARY_FILE"
    echo "Chunk 단위: $CHUNK_TYPE" >> "$SUMMARY_FILE"
    echo "----------------------------------------" >> "$SUMMARY_FILE"
    
    # Passage 개수 확인
    PASSAGE_COUNT=$(poetry run python -c "from lex_dpr.utils.io import read_jsonl; print(len(list(read_jsonl('$CORPUS_FILE'))))")
    echo "Passage 개수: $PASSAGE_COUNT" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(echo "$MODEL" | sed 's/[\/\-]/_/g')
        RESULT_FILE="$RESULTS_DIR/${CHUNK_TYPE}_${MODEL_NAME}.json"
        REPORT_FILE="$RESULTS_DIR/${CHUNK_TYPE}_${MODEL_NAME}.txt"
        
        echo "  평가 중: $CHUNK_TYPE / $MODEL"
        
        # 평가 실행
        poetry run lex-dpr eval \
            --model "$MODEL" \
            --corpus "$CORPUS_FILE" \
            --eval-pairs "$EVAL_PAIRS" \
            --output "$RESULT_FILE" \
            --report "$REPORT_FILE" \
            --k-values 1 3 5 10 20 \
            --batch-size 8 \
            --no-wandb || {
            echo "    ⚠️  평가 실패: $MODEL"
            continue
        }
        
        # 결과 요약 추출
        if [ -f "$RESULT_FILE" ]; then
            echo "    모델: $MODEL" >> "$SUMMARY_FILE"
            poetry run python -c "
import json
with open('$RESULT_FILE', 'r', encoding='utf-8') as f:
    results = json.load(f)
    for key in ['val_cosine_ndcg@10', 'val_cosine_recall@10', 'val_cosine_mrr@10']:
        if key in results:
            print(f'      {key}: {results[key]:.4f}')
" >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
        fi
    done
    echo "" >> "$SUMMARY_FILE"
done

# ==========================================
# 5. 결과 비교 및 출력
# ==========================================
echo "[5/6] 결과 비교 중..."
echo ""

# 비교 테이블 생성
COMPARISON_FILE="$RESULTS_DIR/comparison.txt"
poetry run python << EOF
import json
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
chunk_types = ["paragraph", "item", "article"]
models = ${MODELS[@]@Q}

comparison = {}
for chunk_type in chunk_types:
    comparison[chunk_type] = {}
    for model in models:
        model_name = model.replace("/", "_").replace("-", "_")
        result_file = results_dir / f"{chunk_type}_{model_name}.json"
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                comparison[chunk_type][model] = {
                    'ndcg@10': results.get('val_cosine_ndcg@10', 0),
                    'recall@10': results.get('val_cosine_recall@10', 0),
                    'mrr@10': results.get('val_cosine_mrr@10', 0),
                }

# 비교 테이블 출력
with open("$COMPARISON_FILE", 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("Passage Chunk 단위별 모델 성능 비교\n")
    f.write("=" * 100 + "\n\n")
    
    for model in models:
        f.write(f"\n모델: {model}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Chunk 단위':<20} {'NDCG@10':<15} {'Recall@10':<15} {'MRR@10':<15}\n")
        f.write("-" * 100 + "\n")
        
        for chunk_type in chunk_types:
            if chunk_type in comparison and model in comparison[chunk_type]:
                metrics = comparison[chunk_type][model]
                f.write(f"{chunk_type:<20} {metrics['ndcg@10']:<15.4f} {metrics['recall@10']:<15.4f} {metrics['mrr@10']:<15.4f}\n")
        
        f.write("\n")

print("✅ 비교 결과 저장: $COMPARISON_FILE")
EOF

# ==========================================
# 6. 최종 요약 출력
# ==========================================
echo "[6/6] 최종 요약"
echo ""
echo "=========================================="
echo "평가 완료!"
echo "=========================================="
echo ""
echo "📊 결과 파일:"
echo "  - 요약: $SUMMARY_FILE"
echo "  - 비교: $COMPARISON_FILE"
echo ""
echo "📁 상세 결과:"
for CHUNK_TYPE in "paragraph" "item" "article"; do
    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(echo "$MODEL" | sed 's/[\/\-]/_/g')
        RESULT_FILE="$RESULTS_DIR/${CHUNK_TYPE}_${MODEL_NAME}.json"
        REPORT_FILE="$RESULTS_DIR/${CHUNK_TYPE}_${MODEL_NAME}.txt"
        if [ -f "$RESULT_FILE" ]; then
            echo "  - $CHUNK_TYPE / $MODEL:"
            echo "    JSON: $RESULT_FILE"
            echo "    리포트: $REPORT_FILE"
        fi
    done
done
echo ""
echo "📈 비교 결과 미리보기:"
cat "$COMPARISON_FILE"
echo ""
echo "=========================================="

