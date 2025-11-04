from __future__ import annotations
import argparse, os
from typing import Dict, Any, Iterable, Iterator
from .utils_io import read_json, write_jsonl
from .filters import is_deleted_clause, normalize_whitespace

def law_iter_to_passages(js: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    law.go.kr 조문 트리 JSON을 passage JSONL로 변환.
    기대 키(예): 법령ID/법령명/조문[], 각 조문: 조문번호/조문내용/조문시행일자/항[]
    """
    law_id = str(js.get("법령ID") or js.get("law_id") or "").zfill(6)
    law_name = js.get("법령명") or js.get("law_name") or ""
    articles = js.get("조문") or js.get("articles") or []

    def aid(article_no: str) -> str:
        return f"LAW_{law_id}_제{article_no}조"

    for a in articles:
        art_no = str(a.get("조문번호") or a.get("article_no") or "").strip()
        eff = str(a.get("조문시행일자") or a.get("effective_date") or "")
        article_id = aid(art_no)
        # 조문 본문만 있는 경우
        body = normalize_whitespace(a.get("조문내용") or a.get("article_text") or "")
        if body and not is_deleted_clause(body) and not a.get("항"):
            yield {
                "id": article_id,
                "parent_id": article_id,
                "type": "법령",
                "law_id": law_id,
                "law_name": law_name,
                "article": f"제{art_no}조",
                "effective_date": eff,
                "text": body,
            }

        # 항/호 처리
        for para in a.get("항", []):
            pno = str(para.get("항번호") or para.get("paragraph_no") or "").strip()
            ptext = normalize_whitespace(para.get("항내용") or para.get("paragraph_text") or "")
            if ptext and not is_deleted_clause(ptext):
                pid = f"{article_id}_{pno}"
                yield {
                    "id": pid,
                    "parent_id": article_id,
                    "type": "법령",
                    "law_id": law_id,
                    "law_name": law_name,
                    "article": f"제{art_no}조",
                    "effective_date": eff,
                    "text": ptext,
                }
            for itm in para.get("호", []):
                hno = str(itm.get("호번호") or itm.get("item_no") or "").strip()
                htext = normalize_whitespace(itm.get("호내용") or itm.get("item_text") or "")
                if htext and not is_deleted_clause(htext):
                    hid = f"{article_id}_{pno}_{hno}"
                    yield {
                        "id": hid,
                        "parent_id": f"{article_id}_{pno}",
                        "type": "법령",
                        "law_id": law_id,
                        "law_name": law_name,
                        "article": f"제{art_no}조",
                        "effective_date": eff,
                        "text": htext,
                    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="law.go.kr 조문 트리 JSON 파일")
    ap.add_argument("--out", required=True, help="출력 JSONL 경로")
    args = ap.parse_args()

    js = read_json(args.src)
    rows = list(law_iter_to_passages(js))
    write_jsonl(args.out, rows)
    print(f"[preprocess_law] passages: {len(rows)} → {args.out}")

if __name__ == "__main__":
    main()
