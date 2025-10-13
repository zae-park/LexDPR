import json
import random
from pathlib import Path
from typing import List, Dict

# 법령/규제 도메인 템플릿
QUERY_TEMPLATES = [
    "{키워드}에 대한 규정은 무엇인가요?",
    "{키워드} 관련 법령 조항을 알려주세요",
    "{키워드}의 예외 적용 조건은?",
    "{키워드}를 위한 내부통제 기준은?",
    "{키워드} 시 준수해야 할 사항은?",
    "{키워드}에 대한 감독기관 입장은?",
    "{키워드} 도입 시 고려사항은?",
    "{키워드}의 법적 근거는 무엇인가요?",
]

# 법령 도메인 키워드
KEYWORDS = [
    "망분리",
    "VDI",
    "클라우드",
    "가상자산",
    "커스터디",
    "핫월렛",
    "콜드월렛",
    "마이데이터",
    "API 캐싱",
    "개인정보보호",
    "접근통제",
    "MFA",
    "전자금융",
    "비조치의견",
    "신용정보",
    "OAuth",
    "로그 모니터링",
    "보안통제",
    "데이터 동의",
    "스코프 관리",
    "멀티시그",
]


# 가상 문서 생성
def generate_passages(num_docs=30) -> List[Dict]:
    passages = []

    for i in range(num_docs):
        keyword = random.choice(KEYWORDS)
        related_kw = random.choice([k for k in KEYWORDS if k != keyword])

        doc_types = [
            f"제{i+1}조 {keyword} 관련 규정\n\n본 조항은 {keyword}의 운영 및 관리 기준을 규정한다. {related_kw}와의 연계를 고려하여 내부통제 체계를 구축해야 한다.",
            f"질의요지\n{keyword} 시스템 도입 시 법적 요건 검토\n\n사실관계\n당사는 {keyword}를 도입하려 하며, {related_kw} 규정 준수 여부를 확인하고자 함.\n\n검토\n{keyword}는 보안통제 강화가 필요하며, 관련 로그를 3년간 보관해야 함.\n\n결론\n조건부 허용. 단, {related_kw} 요건을 충족할 것.",
            f"{keyword} 가이드라인\n\n1. 목적: {keyword}의 안전한 운영\n2. 적용대상: {related_kw} 사용 기관\n3. 주요내용: 접근통제, 로그관리, 정기점검\n4. 위반 시 조치사항",
        ]

        text = random.choice(doc_types)

        passages.append(
            {
                "id": f"synth_{i:04d}",
                "text": text,
                "meta": {"keyword": keyword, "related": related_kw},
            }
        )

    return passages


def generate_queries_with_labels(passages: List[Dict], num_queries=100) -> List[Dict]:
    queries = []

    for i in range(num_queries):
        # 랜덤하게 positive passage 선택
        pos_passage = random.choice(passages)
        keyword = pos_passage["meta"]["keyword"]

        # Query 생성
        template = random.choice(QUERY_TEMPLATES)
        query_text = template.format(키워드=keyword)

        # Hard negative 샘플링 (같은 키워드 아닌 것)
        negative_candidates = [
            p
            for p in passages
            if p["id"] != pos_passage["id"] and p["meta"]["keyword"] != keyword
        ]

        hard_negatives = random.sample(
            negative_candidates, min(5, len(negative_candidates))
        )

        queries.append(
            {
                "id": f"q{i:04d}",
                "question": query_text,
                "positive_ids": [pos_passage["id"]],
                "hard_negative_ids": [n["id"] for n in hard_negatives],
            }
        )

    return queries


def main():
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 패시지 생성
    passages = generate_passages(num_docs=50)

    # Corpus 저장
    corpus_path = output_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(passages)} passages -> {corpus_path}")

    # 학습용 쿼리 생성
    train_queries = generate_queries_with_labels(passages, num_queries=150)
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for q in train_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(train_queries)} training queries -> {train_path}")

    # 평가용 쿼리 생성
    eval_queries = generate_queries_with_labels(passages, num_queries=30)
    eval_path = output_dir / "eval.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for q in eval_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(eval_queries)} eval queries -> {eval_path}")

    print("\n📊 Data Statistics:")
    print(f"   Passages: {len(passages)}")
    print(f"   Train queries: {len(train_queries)}")
    print(f"   Eval queries: {len(eval_queries)}")
    print(f"   Unique keywords: {len(set(KEYWORDS))}")


if __name__ == "__main__":
    main()
