# lex_dpr/data_processing/preprocess_law.py

from __future__ import annotations
import argparse
from typing import Dict, Any, Iterator, List, Tuple, Union
from ..utils.io import read_json, write_jsonl
from .filters import is_deleted_clause, normalize_whitespace

Json = Dict[str, Any]


def coerce_text(x: Any) -> str:
    """
    dict/list/str 어떤 타입이 와도 '가능한 문자열'로 변환.
    - dict이면 흔한 텍스트 키 후보를 우선적으로 시도
    - list면 문자열 요소만 취해 합침
    - 그 밖은 빈 문자열
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join([s for s in x if isinstance(s, str)])
    if isinstance(x, dict):
        for k in ("조문내용","항내용","호내용","내용","text","#text","value"):
            if k in x and isinstance(x[k], str):
                return x[k]
        # 값이 전부 문자열이면 합치기
        strs = [v for v in x.values() if isinstance(v, str)]
        if strs:
            return " ".join(strs)
    return ""

# -------------------------------
# 스키마 헬퍼
# -------------------------------
def _extract_law_fields(js: Json) -> Tuple[str, str, list, bool]:
    """
    두 스키마 모두 지원:
    A) {법령ID, 법령명, 조문:[...] }
    B) {법령:{ 기본정보:{법령ID, 법령명_한글}, 조문:{조문단위:[...]}}}
    returns: (law_id, law_name, articles_list, is_wrapped)
    """
    if "법령" in js:
        law = js["법령"]
        info = law.get("기본정보", {})
        law_id = str(info.get("법령ID") or "").zfill(6)
        law_name = info.get("법령명_한글") or info.get("법령명") or ""
        articles = (law.get("조문") or {}).get("조문단위") or []
        return law_id, law_name, articles, True
    else:
        law_id = str(js.get("법령ID") or js.get("law_id") or "").zfill(6)
        law_name = js.get("법령명") or js.get("law_name") or ""
        articles = js.get("조문") or js.get("articles") or []
        return law_id, law_name, articles, False

def _format_article_label(art_no: str) -> str:
    # 입력이 '제30조' 같은 전체 라벨로 오는 경우 있음 → 중복 방지
    t = art_no.strip()
    if t.startswith("제") and t.endswith("조"):
        return t
    return f"제{t}조" if t else ""

def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _to_paragraph_dicts(paras_raw) -> List[Dict[str, str]]:
    """
    para가 dict / str / list(any) 전부 수용하여
    [{항번호, 항내용}] 리스트로 통일.
    """
    out = []
    for i, para in enumerate(_as_list(paras_raw), 1):
        if isinstance(para, dict):
            pno = str(para.get("항번호") or para.get("paragraph_no") or i).strip()
            ptx = normalize_whitespace(coerce_text(para.get("항내용") or para.get("paragraph_text") or para))
        else:  # str or other
            pno = str(i)
            ptx = normalize_whitespace(coerce_text(para))
        if ptx:
            out.append({"항번호": pno, "항내용": ptx, "raw": para})
    return out

def _to_item_dicts(items_raw) -> List[Dict[str, str]]:
    """
    item(호)이 dict / str / list(any) 전부 수용하여
    [{호번호, 호내용}] 리스트로 통일.
    """
    out = []
    for j, itm in enumerate(_as_list(items_raw), 1):
        if isinstance(itm, dict):
            ino = str(itm.get("호번호") or itm.get("item_no") or j).strip()
            itx = normalize_whitespace(coerce_text(itm.get("호내용") or itm.get("item_text") or itm))
        else:
            ino = str(j)
            itx = normalize_whitespace(coerce_text(itm))
        if itx:
            out.append({"호번호": ino, "호내용": itx})
    return out

# -------------------------------
# 변환 본체
# -------------------------------
def law_iter_to_passages(js: Json, include_items: bool = False) -> Iterator[Json]:
    """
    법령 JSON을 passage 리스트로 변환
    
    Args:
        js: 법령 JSON
        include_items: True면 호(절) 단위까지 생성, False면 항 단위까지만 생성 (기본값: False)
    """
    law_id, law_name, articles, _ = _extract_law_fields(js)

    for a in _as_list(articles):
        if not isinstance(a, dict):
            continue
        art_no = str(a.get("조문번호") or a.get("article_no") or "").strip()
        if not art_no:
            continue
        article_label = _format_article_label(art_no)
        article_id = f"LAW_{law_id}_{article_label}"
        eff = str(a.get("조문시행일자") or a.get("effective_date") or "")

        # 조문 본문 (항이 없을 때만 조문 단위로 내보냄)
        body_raw = a.get("조문내용") or a.get("article_text")
        body = normalize_whitespace(coerce_text(body_raw))
        paras = _to_paragraph_dicts(a.get("항"))
        has_paras = len(paras) > 0
        if body and not has_paras and not is_deleted_clause(body):
            yield {
                "id": article_id,
                "parent_id": article_id,
                "type": "법령",
                "law_id": law_id,
                "law_name": law_name,
                "article": article_label,
                "effective_date": eff,
                "text": body,
            }

        # 항/호 처리
        for para in paras:
            pno = para["항번호"]
            ptx = para["항내용"]
            if not is_deleted_clause(ptx):
                pid = f"{article_id}_{pno}"
                
                # 호(절)가 있는 경우 처리 방식 결정
                raw = para.get("raw") if isinstance(para.get("raw"), dict) else {}
                items = _to_item_dicts(raw.get("호") if isinstance(raw, dict) else None)
                has_items = len(items) > 0
                
                if include_items and has_items:
                    # 호 단위로 세분화하는 경우: 각 호를 별도 passage로 생성
                    for itm in items:
                        ino = itm["호번호"]
                        itx = itm["호내용"]
                        if is_deleted_clause(itx):
                            continue
                        hid = f"{article_id}_{pno}_{ino}"
                        yield {
                            "id": hid,
                            "parent_id": f"{article_id}_{pno}",
                            "type": "법령",
                            "law_id": law_id,
                            "law_name": law_name,
                            "article": article_label,
                            "effective_date": eff,
                            "text": itx,
                        }
                else:
                    # 항 단위로 생성 (호가 있어도 항 전체를 하나의 passage로)
                    # 호가 있는 경우 항 내용에 호들을 포함시켜 더 긴 passage 생성
                    if has_items:
                        # 항 내용 + 호들을 합쳐서 더 긴 passage 생성
                        item_texts = []
                        for itm in items:
                            itx = itm["호내용"]
                            if not is_deleted_clause(itx):
                                item_texts.append(itx)
                        if item_texts:
                            # 항 내용과 호들을 합침
                            combined_text = f"{ptx}\n" + "\n".join(item_texts)
                            yield {
                                "id": pid,
                                "parent_id": article_id,
                                "type": "법령",
                                "law_id": law_id,
                                "law_name": law_name,
                                "article": article_label,
                                "effective_date": eff,
                                "text": normalize_whitespace(combined_text),
                            }
                        else:
                            # 호가 모두 삭제된 경우 항만 생성
                            yield {
                                "id": pid,
                                "parent_id": article_id,
                                "type": "법령",
                                "law_id": law_id,
                                "law_name": law_name,
                                "article": article_label,
                                "effective_date": eff,
                                "text": ptx,
                            }
                    else:
                        # 호가 없는 경우 항만 생성
                        yield {
                            "id": pid,
                            "parent_id": article_id,
                            "type": "법령",
                            "law_id": law_id,
                            "law_name": law_name,
                            "article": article_label,
                            "effective_date": eff,
                            "text": ptx,
                        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="law.go.kr 조문 트리 JSON 파일")
    ap.add_argument("--out", required=True, help="출력 JSONL 경로")
    ap.add_argument("--include-items", action="store_true", 
                    help="호(절) 단위까지 passage 생성 (기본값: False, 항 단위까지만 생성)")
    args = ap.parse_args()

    js = read_json(args.src)
    rows = list(law_iter_to_passages(js, include_items=args.include_items))
    write_jsonl(args.out, rows)
    print(f"[preprocess_law] passages: {len(rows)} → {args.out}")
    if args.include_items:
        print(f"[preprocess_law] 호(절) 단위까지 생성됨")
    else:
        print(f"[preprocess_law] 항 단위까지만 생성됨 (호는 항에 포함)")

if __name__ == "__main__":
    main()
