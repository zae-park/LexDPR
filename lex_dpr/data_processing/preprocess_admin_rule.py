# lex_dpr/data_processing/preprocess_admin_rule.py

from __future__ import annotations
import argparse, re
from typing import Dict, Any, Iterator, List, Tuple, Union
from .utils_io import read_json, write_jsonl

Json = Dict[str, Any]

# ─────────────────────────────
# 공통 유틸
# ─────────────────────────────
def _as_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _flatten_str_blocks(x: Any) -> List[str]:
    """리스트/리스트의 리스트/문자열을 모두 문자열 리스트로 평탄화."""
    out: List[str] = []
    if x is None:
        return out
    if isinstance(x, str):
        if x.strip():
            out.append(x.strip())
        return out
    if isinstance(x, list):
        for e in x:
            out.extend(_flatten_str_blocks(e))
        return out
    if isinstance(x, dict):
        # dict면 값들 중 문자열만 모아본다 (필요시 키 확장)
        for v in x.values():
            out.extend(_flatten_str_blocks(v))
        return out
    return out

def join_lines_keep_tables(lines: List[str]) -> str:
    """좌우 공백만 정리하고 줄단위로 이어붙임(박스문자/표 정렬 보존)."""
    out = []
    for s in lines:
        if s is None:
            continue
        t = str(s).strip("\r\n")
        if t != "":
            out.append(t)
    return "\n".join(out).strip()

# ─────────────────────────────
# 조문 라인 → 조문 단위 슬라이스
# ─────────────────────────────
_ARTICLE_RE = re.compile(r"^제\s*(\d+)\s*조(?:\(([^)]*)\))?")

def _ensure_lines(clauses: Union[str, List[str]]) -> List[str]:
    if clauses is None:
        return []
    if isinstance(clauses, str):
        # 줄바꿈 기준 분할
        return [ln for ln in clauses.splitlines()]
    if isinstance(clauses, list):
        # 리스트 안에 또 리스트/문자열 혼재 가능 → 문자열만 뽑아 1차원화
        return _flatten_str_blocks(clauses)
    # 기타 타입은 문자열화
    return [str(clauses)]

def split_articles(clauses: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    조문내용을 '제n조(제목) ...' 기준으로 조문 단위로 슬라이싱.
    항/호는 일단 같은 조문 텍스트에 포함(추후 필요 시 세분화 가능).
    """
    lines = _ensure_lines(clauses)

    arts: List[Dict[str, Any]] = []
    cur_no, cur_title, buf = None, None, []

    def flush():
        nonlocal cur_no, cur_title, buf
        if cur_no is not None and buf:
            arts.append({
                "article_no": cur_no,
                "article_label": f"제{cur_no}조",
                "title": cur_title or "",
                "text": join_lines_keep_tables(buf),
            })
        cur_no, cur_title, buf = None, None, []

    for raw in lines:
        ln = str(raw).strip()
        m = _ARTICLE_RE.match(ln)
        if m:
            flush()
            cur_no = m.group(1)
            cur_title = m.group(2) or ""
            rest = _ARTICLE_RE.sub("", ln).strip()
            if rest:
                buf.append(rest)
        else:
            if cur_no is None:
                # 조문 시작 전(제1장 총칙 등)은 스킵
                continue
            buf.append(ln)
    flush()
    return arts

# ─────────────────────────────
# 별표 처리
# ─────────────────────────────
def iter_appendices(appx_any: Any, rule_id: str) -> Iterator[Dict[str, Any]]:
    """
    '별표'는 dict 혹은 list, 그 안의 '별표단위'도 dict 혹은 list일 수 있다.
    각각을 순회하며 appendix passage 생성.
    """
    for appx in _as_list(appx_any):
        if not isinstance(appx, dict):
            # 비표준 구조는 스킵
            continue
        units_any = appx.get("별표단위", appx)  # 없으면 자기 자신을 유닛처럼 간주
        for unit in _as_list(units_any):
            if not isinstance(unit, dict):
                continue
            title = unit.get("별표제목") or unit.get("제목") or ""
            num = unit.get("별표번호") or unit.get("번호") or ""
            part_no = unit.get("별표가지번호") or unit.get("가지번호") or ""
            content_any = unit.get("별표내용") or unit.get("내용") or []
            # 내용 평탄화 후 줄로 합치기
            text = join_lines_keep_tables(_flatten_str_blocks(content_any))
            if not text:
                continue
            yield {
                "id": f"ADM_{rule_id}_APPX_{part_no or '0'}_1",
                "parent_id": f"ADM_{rule_id}_APPX_{part_no or '0'}",
                "type": "행정규칙-별표",
                "rule_id": rule_id,
                "appendix_no": num,
                "appendix_title": title,
                "text": text,
            }

# ─────────────────────────────
# 메인 변환
# ─────────────────────────────
def preprocess_admin_rule(js: Json) -> List[Dict[str, Any]]:
    svc = js.get("AdmRulService") or {}
    info = svc.get("행정규칙기본정보") or {}

    rule_id = str(info.get("행정규칙일련번호") or info.get("행정규칙ID") or "0").zfill(10)
    rule_name = info.get("행정규칙명") or info.get("제목") or ""
    eff = info.get("시행일자") or info.get("발령일자") or info.get("제정일자") or ""

    rows: List[Dict[str, Any]] = []

    # 1) 조문내용 → 조문 단위 passage
    clauses_any = svc.get("조문내용")  # 문자열 or 리스트(문자열/리스트 혼재)
    articles = split_articles(clauses_any) if clauses_any else []
    for a in articles:
        aid = f"ADM_{rule_id}_{a['article_label']}"
        txt = a["text"]
        if not txt:
            continue
        rows.append({
            "id": aid,
            "parent_id": aid,
            "type": "행정규칙",
            "rule_id": rule_id,
            "rule_name": rule_name,
            "article": a["article_label"],
            "title": a.get("title", ""),
            "effective_date": eff,
            "text": txt,
        })

    # 2) 별표 → appendix passage
    for rec in iter_appendices(svc.get("별표"), rule_id):
        rec["rule_name"] = rule_name
        rec["effective_date"] = eff
        rows.append(rec)

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="행정규칙 JSON (AdmRulService)")
    ap.add_argument("--out", required=True, help="출력 JSONL 경로")
    args = ap.parse_args()

    js = read_json(args.src)
    rows = preprocess_admin_rule(js)
    write_jsonl(args.out, rows)
    print(f"[preprocess_admin_rule] passages: {len(rows)} → {args.out}")

if __name__ == "__main__":
    main()
