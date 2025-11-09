# lex_dpr/data_processing/make_pairs.py
from __future__ import annotations
import argparse, json, random, re
from typing import Dict, Any, List, Optional, Tuple
from ..utils.io import read_jsonl, write_jsonl


# =========================
# Helpers: text normalize
# =========================
def _one_line(s: str, max_len: int = 120) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s[:max_len]

def _short(s: Optional[str], n: int = 80) -> str:
    return (s or "").strip()[:n]

def _valid_passage(p: Dict[str, Any], min_len: int = 50) -> bool:
    return len((p.get("text") or "").strip()) >= min_len


# =========================
# Reference law parsing (참조조문 파싱)
# =========================
def parse_reference_laws(ref_law_text: str) -> List[Dict[str, Any]]:
    """
    참조조문 문자열에서 법령명/행정규칙명, 조문번호, 의조번호, 항번호를 추출.
    
    입력 예시:
        "[1]형법 제355조 제1항,제356조 / [2]산업안전보건기준에 관한 규칙 제1조"
    
    출력 예시:
        [
            {"law_name": "형법", "article_num": "355", "sub_article": None, "paragraph": "1", "type": "law"},
            {"law_name": "형법", "article_num": "356", "sub_article": None, "paragraph": None, "type": "law"},
            {"law_name": "산업안전보건기준에 관한 규칙", "article_num": "1", "sub_article": None, "paragraph": None, "type": "admin"},
        ]
    """
    if not ref_law_text or not ref_law_text.strip():
        return []
    
    # HTML 태그 제거
    ref_law_text = re.sub(r'<br/?>', ' ', ref_law_text)
    ref_law_text = re.sub(r'<[^>]+>', '', ref_law_text)
    
    refs: List[Dict[str, Any]] = []
    seen = set()  # 중복 제거용
    
    # 법령 패턴: 끝에 "법" 또는 "법률"
    law_pattern = r'([가-힣A-Za-z0-9·\s]+(?:법|법률))\s*제?\s*([0-9]+)\s*조(?:\s*의\s*([0-9]+))?(?:\s*제?\s*([0-9]+)\s*항)?'
    
    for m in re.finditer(law_pattern, ref_law_text):
        law_name = m.group(1).strip()
        article_num = m.group(2)
        sub_article = m.group(3) if m.group(3) else None
        paragraph = m.group(4) if m.group(4) else None
        
        key = (law_name, article_num, sub_article, paragraph, "law")
        if key in seen:
            continue
        seen.add(key)
        
        refs.append({
            'law_name': law_name,
            'article_num': article_num,
            'sub_article': sub_article,
            'paragraph': paragraph,
            'type': 'law',
        })
    
    # 행정규칙 패턴: 끝에 "규칙", "고시", "훈령", "예규", "지침" 등
    admin_pattern = r'([가-힣A-Za-z0-9·\s]+(?:규칙|고시|훈령|예규|지침|규정))\s*제?\s*([0-9]+)\s*조(?:\s*의\s*([0-9]+))?(?:\s*제?\s*([0-9]+)\s*항)?'
    
    for m in re.finditer(admin_pattern, ref_law_text):
        rule_name = m.group(1).strip()
        article_num = m.group(2)
        sub_article = m.group(3) if m.group(3) else None
        paragraph = m.group(4) if m.group(4) else None
        
        key = (rule_name, article_num, sub_article, paragraph, "admin")
        if key in seen:
            continue
        seen.add(key)
        
        refs.append({
            'law_name': rule_name,  # 통일성을 위해 law_name 필드 사용
            'article_num': article_num,
            'sub_article': sub_article,
            'paragraph': paragraph,
            'type': 'admin',
        })
    
    return refs


# =========================
# Query builders (type-wise)
# =========================
def build_query_law(p: Dict[str, Any]) -> str:
    law_name = _short(p.get("law_name"), 60)
    article = _short(p.get("article"), 40)
    title = _short(p.get("title"), 60)
    if article and title and title not in article:
        return f"{law_name} {article}({_short(title,40)})의 내용은 무엇인가?"
    if article:
        return f"{law_name} {article}의 내용은 무엇인가?"
    return f"{law_name} 관련 내용은 무엇인가?"

def build_query_admin(p: Dict[str, Any]) -> str:
    rule = _short(p.get("rule_name"), 60)
    article = _short(p.get("article"), 40)
    title = _short(p.get("title"), 60)
    annex = _short(p.get("annex_title") or p.get("appendix_title"), 60)

    if article:
        if title and title not in article:
            return f"{rule} {article}({_short(title,40)})의 내용은 무엇인가?"
        return f"{rule} {article}의 내용은 무엇인가?"
    if annex:
        return f"{rule}의 '{annex}' 별표 내용은 무엇인가?"
    return f"{rule} 관련 내용은 무엇인가?"

def build_query_prec(p: Dict[str, Any]) -> str:
    """판례 passage에서 질의 생성 (기존 함수 - 판례 passage용)"""
    title = (p.get("title") or "").strip()
    if title:
        return f"{_one_line(title, 120)}의 요지는 무엇인가?"
    # fallback: headnote/summary에서 한 줄
    hs = (p.get("headnote") or p.get("summary") or "").strip()
    hs = _one_line(hs, 120)
    return f"{hs}의 요지는 무엇인가?" if hs else "이 판례의 요지는 무엇인가?"

def build_query_from_precedent_json(prec_json: Dict[str, Any]) -> Optional[str]:
    """
    판례 원본 JSON에서 질의 생성.
    
    전략:
    1. 우선순위 1: 판시사항 (법적 쟁점이 명확)
    2. 우선순위 2: 판결요지 요약 (사건+판결)
    3. 우선순위 3: 사건명 기반 질의
    
    Args:
        prec_json: 판례 원본 JSON (판시사항, 판결요지, 사건명 필드 포함)
    
    Returns:
        생성된 질의 문자열 또는 None
    """
    def clean_html(text: str) -> str:
        """HTML 태그 제거 및 공백 정규화"""
        if not text:
            return ""
        text = re.sub(r'<br/?>', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_first_section(text: str) -> str:
        """[번호]로 구분된 첫 번째 섹션 추출"""
        if not text:
            return ""
        sections = re.split(r'\[(\d+)\]', text)
        if len(sections) > 2:
            # 첫 번째 섹션 내용 (인덱스 2)
            first_content = sections[2].strip()
            # 너무 길면 자르기
            if len(first_content) > 200:
                first_content = first_content[:200] + "..."
            return first_content
        return text.strip()[:200] if text else ""
    
    # 전략 1: 판시사항 사용
    headnote = clean_html(prec_json.get("판시사항") or prec_json.get("headnote") or "")
    if headnote:
        first_headnote = extract_first_section(headnote)
        if first_headnote:
            # 질의 형식: "법적 쟁점에 대한 법적 판단은?"
            query = f"{first_headnote}에 대한 법적 판단은?"
            return _one_line(query, 200)
    
    # 전략 2: 판결요지 사용
    summary = clean_html(prec_json.get("판결요지") or prec_json.get("summary") or "")
    if summary:
        first_summary = extract_first_section(summary)
        if first_summary:
            # 질의 형식: "사건 내용에 대한 법적 근거는?"
            query = f"{first_summary}에 대한 법적 근거는?"
            return _one_line(query, 200)
    
    # 전략 3: 사건명 사용
    title = (prec_json.get("사건명") or prec_json.get("title") or "").strip()
    if title:
        title_short = _one_line(title, 100)
        query = f"{title_short}에 적용되는 법령은?"
        return query
    
    return None


# =========================
# Hard negative utilities
# =========================
def _sample_hard_negatives(
    target: Dict[str, Any],
    pool: List[Dict[str, Any]],
    n: int,
    group_key: Optional[str],
    avoid_same_parent: bool = True,
) -> List[str]:
    """
    1) 같은 group_key(예: 같은 law_name/rule_name/court_name)에서 우선 추출
    2) 부족하면 동일 타입 전체에서 보충
    3) 같은 parent_id(동일 문서의 다른 청크)는 제외하여 in-document leakage 방지
    """
    if n <= 0:
        return []

    tid = target.get("id")
    tparent = target.get("parent_id")

    def ok(x: Dict[str, Any]) -> bool:
        if x.get("id") == tid:
            return False
        if avoid_same_parent and tparent and x.get("parent_id") == tparent:
            return False
        return True

    same_group_ids: List[str] = []
    if group_key:
        gval = (target.get(group_key) or "").strip()
        if gval:
            same_group_ids = [
                x["id"] for x in pool
                if ok(x) and (x.get(group_key, "").strip() == gval)
            ]

    random.shuffle(same_group_ids)
    hn: List[str] = same_group_ids[:n]

    if len(hn) < n:
        rest = [x["id"] for x in pool if ok(x) and x.get("id") not in set(hn)]
        random.shuffle(rest)
        hn.extend(rest[: (n - len(hn))])

    return hn[:n]


# =========================
# Builders for each type
#  - meta 보존
#  - very short passage 필터
# =========================
def build_pairs_from_law(law: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    law = [p for p in law if _valid_passage(p)]
    for p in law:
        q = build_query_law(p)
        pos = [p["id"]]
        hn = _sample_hard_negatives(p, law, hn_per_q, group_key="law_name")
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn,
            "meta": {
                "type": "law",
                "law_name": p.get("law_name"),
                "article": p.get("article"),
                "parent_id": p.get("parent_id"),
            },
        })
    return rows

def build_pairs_from_admin(admin: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    admin = [p for p in admin if _valid_passage(p)]
    for p in admin:
        q = build_query_admin(p)
        pos = [p["id"]]
        hn = _sample_hard_negatives(p, admin, hn_per_q, group_key="rule_name")
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn,
            "meta": {
                "type": "admin",
                "rule_name": p.get("rule_name"),
                "article": p.get("article"),
                "parent_id": p.get("parent_id"),
            },
        })
    return rows

def build_pairs_from_prec(prec: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    """
    판례 passage에서 질의-판례 쌍 생성 (기존 방식).
    판례 passage 자체를 positive로 사용.
    """
    rows: List[Dict[str, Any]] = []
    prec = [p for p in prec if _valid_passage(p)]
    for p in prec:
        q = build_query_prec(p)
        pos = [p["id"]]
        hn = _sample_hard_negatives(p, prec, hn_per_q, group_key="court_name")
        # 판례의 headnote/summary/text를 meta에 보관 → cross positive 추출에 활용
        meta_source = " ".join([
            (p.get("headnote") or ""),
            (p.get("summary") or ""),
            (p.get("text") or ""),
        ]).strip()
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn,
            "meta": {
                "type": "prec",
                "court_name": p.get("court_name"),
                "case_number": p.get("case_number"),
                "parent_id": p.get("parent_id"),
                "source_text": _one_line(meta_source, 400),
            },
        })
    return rows

def build_pairs_from_precedent_jsons(
    prec_json_dir: str,
    law_passages: List[Dict[str, Any]],
    admin_passages: List[Dict[str, Any]] = None,
    max_positives: int = 5,
    hn_per_q: int = 2,
    glob_pattern: str = "**/*.json",
    use_admin: bool = False,
) -> List[Dict[str, Any]]:
    """
    판례 원본 JSON 파일들에서 질의-법령/행정규칙 쌍 생성 (새로운 방식).
    판례의 사건 내용을 질의로, 참조조문의 법령/행정규칙을 positive로 사용.
    
    Args:
        prec_json_dir: 판례 원본 JSON 파일들이 있는 디렉토리
        law_passages: 모든 법령 passage 리스트
        admin_passages: 모든 행정규칙 passage 리스트 (선택)
        max_positives: 최대 positive passage 개수
        hn_per_q: 질의당 hard negative 개수
        glob_pattern: 파일 검색 패턴
        use_admin: 행정규칙 사용 여부 (기본값: False, 법령만 사용)
    
    Returns:
        질의-법령/행정규칙 쌍 리스트
    """
    from pathlib import Path
    
    p = Path(prec_json_dir)
    if not p.exists():
        return []
    
    # 행정규칙 사용 여부에 따라 처리
    if use_admin:
        admin_passages = admin_passages or []
    else:
        admin_passages = []  # 행정규칙 사용 안 함
    
    # 법령 및 행정규칙 인덱스 생성
    law_index = build_law_index(law_passages)
    admin_index = build_admin_index(admin_passages) if use_admin else {}
    
    rows: List[Dict[str, Any]] = []
    files = sorted(p.glob(glob_pattern))
    
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                prec_json = json.load(f)
            
            # 질의-법령/행정규칙 쌍 생성
            pair = build_pair_from_precedent_json(
                prec_json,
                law_index,
                admin_index,
                law_passages,
                admin_passages,
                max_positives=max_positives,
                hn_per_q=hn_per_q,
            )
            
            if pair:
                rows.append(pair)
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")
            continue
    
    return rows


# =========================
# Cross-type positives (prec → law)
#  - 판례 요지/본문에서 "○○법 제n조(의m)" 인용 탐지
#  - 해당 법령 passage를 positive에 추가 (상한 2개)
# =========================
LAW_MENTION = re.compile(
    r"([가-힣A-Za-z0-9·\s]+법)\s*제?\s*([0-9]+)\s*조(?:\s*의\s*([0-9]+))?",
    flags=re.UNICODE
)

def _law_index_by_name(law_passages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """법령명으로 인덱싱 (기존 함수 - cross positive용)"""
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for lp in law_passages:
        name = (lp.get("law_name") or "").strip()
        if not name:
            continue
        by_name.setdefault(name, []).append(lp)
    return by_name

def build_admin_index(admin_passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    행정규칙 passage를 규칙명+조문번호로 인덱싱.
    
    반환 구조:
    {
        "산업안전보건기준에 관한 규칙": {
            "1": [passage1, passage2, ...],
            "2": [...],
        }
    }
    """
    index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    
    for ap in admin_passages:
        rule_name = (ap.get("rule_name") or "").strip()
        article = (ap.get("article") or "").strip()
        
        if not rule_name or not article:
            continue
        
        # article에서 조문번호 추출: "제1조" → "1"
        article_match = re.search(r'제\s*([0-9]+)\s*조', article)
        if not article_match:
            continue
        
        article_num = article_match.group(1)
        
        # 인덱스 구조 생성
        if rule_name not in index:
            index[rule_name] = {}
        if article_num not in index[rule_name]:
            index[rule_name][article_num] = []
        
        index[rule_name][article_num].append(ap)
    
    return index

def build_law_index(law_passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    법령 passage를 법령명+조문번호로 인덱싱.
    
    반환 구조:
    {
        "형법": {
            "355": [passage1, passage2, ...],  # 제355조 관련 passages
            "356": [...],
        },
        "특정경제범죄 가중처벌 등에 관한 법률": {
            "3": [...],
            "8": [...],
        }
    }
    
    article 필드에서 조문번호 추출:
    - "제355조" → "355"
    - "제355조의2" → "355" (의조는 무시하고 메인 조문번호만 사용)
    """
    index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    
    for lp in law_passages:
        law_name = (lp.get("law_name") or "").strip()
        article = (lp.get("article") or "").strip()
        
        if not law_name or not article:
            continue
        
        # article에서 조문번호 추출: "제355조" → "355", "제355조의2" → "355"
        article_match = re.search(r'제\s*([0-9]+)\s*조', article)
        if not article_match:
            continue
        
        article_num = article_match.group(1)
        
        # 인덱스 구조 생성
        if law_name not in index:
            index[law_name] = {}
        if article_num not in index[law_name]:
            index[law_name][article_num] = []
        
        index[law_name][article_num].append(lp)
    
    return index

def find_law_passages(
    index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    law_name: str,
    article_num: str,
    sub_article: Optional[str] = None,
    paragraph: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    인덱스에서 법령 passage 검색.
    
    Args:
        index: build_law_index()로 생성한 인덱스
        law_name: 법령명
        article_num: 조문번호 (문자열)
        sub_article: 의조번호 (선택, 현재는 무시)
        paragraph: 항번호 (선택, 현재는 무시)
    
    Returns:
        매칭된 passage 리스트
    """
    if law_name not in index:
        return []
    
    if article_num not in index[law_name]:
        return []
    
    # 현재는 조문번호만으로 매칭 (항번호, 의조번호는 나중에 정밀화 가능)
    passages = index[law_name][article_num]
    
    # 항번호가 지정된 경우 필터링 (선택적)
    if paragraph:
        filtered = [p for p in passages if paragraph in (p.get("id") or "")]
        if filtered:
            return filtered
    
    return passages

def find_admin_passages(
    index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    rule_name: str,
    article_num: str,
    sub_article: Optional[str] = None,
    paragraph: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    인덱스에서 행정규칙 passage 검색.
    
    Args:
        index: build_admin_index()로 생성한 인덱스
        rule_name: 행정규칙명
        article_num: 조문번호 (문자열)
        sub_article: 의조번호 (선택, 현재는 무시)
        paragraph: 항번호 (선택, 현재는 무시)
    
    Returns:
        매칭된 passage 리스트
    """
    if rule_name not in index:
        return []
    
    if article_num not in index[rule_name]:
        return []
    
    passages = index[rule_name][article_num]
    
    # 항번호가 지정된 경우 필터링 (선택적)
    if paragraph:
        filtered = [p for p in passages if paragraph in (p.get("id") or "")]
        if filtered:
            return filtered
    
    return passages

def build_pair_from_precedent_json(
    prec_json: Dict[str, Any],
    law_index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    admin_index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    all_law_passages: List[Dict[str, Any]],
    all_admin_passages: List[Dict[str, Any]],
    max_positives: int = 5,
    hn_per_q: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    판례 원본 JSON에서 질의-법령/행정규칙 쌍 생성.
    
    Args:
        prec_json: 판례 원본 JSON
        law_index: build_law_index()로 생성한 법령 인덱스
        admin_index: build_admin_index()로 생성한 행정규칙 인덱스
        all_law_passages: 모든 법령 passage 리스트
        all_admin_passages: 모든 행정규칙 passage 리스트
        max_positives: 최대 positive passage 개수
        hn_per_q: 질의당 hard negative 개수
    
    Returns:
        {
            "query_text": "...",
            "positive_passages": ["LAW_...", "ADM_...", ...],
            "hard_negatives": [...],
            "meta": {...}
        } 또는 None (질의 생성 실패 또는 매칭된 법령/행정규칙 없음)
    """
    # 1. 질의 생성
    query_text = build_query_from_precedent_json(prec_json)
    if not query_text:
        return None
    
    # 2. 참조조문 파싱 (법령 + 행정규칙)
    ref_law_text = prec_json.get("참조조문") or prec_json.get("ref_law") or ""
    refs = parse_reference_laws(ref_law_text)
    
    if not refs:
        return None
    
    # 3. 법령/행정규칙 인덱스에서 passage 검색
    positive_ids: List[str] = []
    seen_ids = set()
    law_refs = []
    admin_refs = []
    
    for ref in refs:
        name = ref["law_name"]
        article_num = ref["article_num"]
        sub_article = ref.get("sub_article")
        paragraph = ref.get("paragraph")
        ref_type = ref.get("type", "law")
        
        passages = []
        if ref_type == "law":
            passages = find_law_passages(law_index, name, article_num, sub_article, paragraph)
            law_refs.append(ref)
        elif ref_type == "admin":
            passages = find_admin_passages(admin_index, name, article_num, sub_article, paragraph)
            admin_refs.append(ref)
        
        for passage in passages:
            passage_id = passage.get("id")
            if passage_id and passage_id not in seen_ids:
                positive_ids.append(passage_id)
                seen_ids.add(passage_id)
                
                # 최대 개수 제한
                if len(positive_ids) >= max_positives:
                    break
        
        if len(positive_ids) >= max_positives:
            break
    
    # positive passage가 없으면 None 반환
    if not positive_ids:
        return None
    
    # 4. Hard negative 샘플링 (법령과 행정규칙 모두 포함)
    all_passages = all_law_passages + all_admin_passages
    hard_negatives = sample_hard_negatives_for_prec_law_pair(
        positive_ids,
        refs,  # 법령과 행정규칙 모두 포함
        {**law_index, **admin_index},  # 통합 인덱스
        all_passages,
        n=hn_per_q,
    )
    
    # 5. 메타데이터 구성
    case_id = str(prec_json.get("판례일련번호") or prec_json.get("case_id") or "").zfill(6)
    case_number = prec_json.get("사건번호") or prec_json.get("case_number") or ""
    court_name = prec_json.get("법원명") or prec_json.get("court_name") or ""
    
    return {
        "query_text": query_text,
        "positive_passages": positive_ids,
        "hard_negatives": hard_negatives,
        "meta": {
            "type": "prec_to_law_admin",
            "precedent_id": case_id,
            "case_number": case_number,
            "court_name": court_name,
            "matched_laws": len(law_refs),
            "matched_admin": len(admin_refs),
            "matched_passages": len(positive_ids),
        }
    }

def sample_hard_negatives_for_prec_law_pair(
    positive_passages: List[str],
    refs: List[Dict[str, Any]],  # 법령과 행정규칙 모두 포함
    combined_index: Dict[str, Dict[str, List[Dict[str, Any]]]],  # 통합 인덱스
    all_passages: List[Dict[str, Any]],  # 법령과 행정규칙 모두 포함
    n: int = 2,
) -> List[str]:
    """
    판례→법령/행정규칙 쌍에 대한 hard negative 샘플링.
    
    전략:
    1. 같은 법령/행정규칙의 다른 조문에서 우선 샘플링
    2. 부족하면 다른 법령/행정규칙에서 랜덤 샘플링
    3. positive passage는 제외
    
    Args:
        positive_passages: positive로 선택된 passage ID 리스트
        refs: 참조조문에서 파싱한 법령/행정규칙 리스트
        combined_index: 법령과 행정규칙 통합 인덱스
        all_passages: 모든 법령/행정규칙 passage 리스트
        n: 샘플링할 hard negative 개수
    
    Returns:
        hard negative passage ID 리스트
    """
    if n <= 0:
        return []
    
    positive_set = set(positive_passages)
    hard_negatives: List[str] = []
    seen_hn = set()
    
    # 전략 1: 같은 법령/행정규칙의 다른 조문에서 샘플링
    for ref in refs:
        name = ref["law_name"]
        article_num = ref["article_num"]
        
        if name not in combined_index:
            continue
        
        # 같은 법령/행정규칙의 다른 조문들 찾기
        other_articles = [
            art_num for art_num in combined_index[name].keys()
            if art_num != article_num
        ]
        
        random.shuffle(other_articles)
        
        for other_art_num in other_articles:
            passages = combined_index[name][other_art_num]
            
            for passage in passages:
                passage_id = passage.get("id")
                if (passage_id and 
                    passage_id not in positive_set and 
                    passage_id not in seen_hn):
                    hard_negatives.append(passage_id)
                    seen_hn.add(passage_id)
                    
                    if len(hard_negatives) >= n:
                        return hard_negatives[:n]
    
    # 전략 2: 다른 법령/행정규칙에서 랜덤 샘플링 (부족한 경우)
    if len(hard_negatives) < n:
        # positive에 사용된 법령/행정규칙명 수집
        positive_names = {ref["law_name"] for ref in refs}
        
        # 다른 법령/행정규칙의 passage들 수집
        other_passages = [
            p for p in all_passages
            if (p.get("id") not in positive_set and
                p.get("id") not in seen_hn and
                ((p.get("law_name") or "").strip() not in positive_names and
                 (p.get("rule_name") or "").strip() not in positive_names))
        ]
        
        random.shuffle(other_passages)
        
        for passage in other_passages:
            passage_id = passage.get("id")
            if passage_id and passage_id not in seen_hn:
                hard_negatives.append(passage_id)
                seen_hn.add(passage_id)
                
                if len(hard_negatives) >= n:
                    break
    
    # 전략 3: 그래도 부족하면 전체에서 랜덤 샘플링
    if len(hard_negatives) < n:
        all_other_passages = [
            p for p in all_passages
            if p.get("id") not in positive_set and p.get("id") not in seen_hn
        ]
        random.shuffle(all_other_passages)
        
        for passage in all_other_passages:
            passage_id = passage.get("id")
            if passage_id:
                hard_negatives.append(passage_id)
                seen_hn.add(passage_id)
                
                if len(hard_negatives) >= n:
                    break
    
    return hard_negatives[:n]

def _article_has_number(art: str, num: str) -> bool:
    """article('제536조의2')에 num('536')이 포함되는지 간단 판정"""
    art = (art or "").replace(" ", "")
    return num in re.sub(r"[^0-9]", "", art)

def attach_cross_positives(rows: List[Dict[str, Any]], law_passages: List[Dict[str, Any]], max_add: int = 2) -> None:
    if not rows or not law_passages:
        return
    law_by_name = _law_index_by_name(law_passages)

    for r in rows:
        meta = r.get("meta") or {}
        if meta.get("type") != "prec":
            continue
        src = meta.get("source_text") or r.get("query_text", "")
        adds: List[str] = []

        # 여러 인용 가능 → 좌측부터 탐색
        for m in LAW_MENTION.finditer(src):
            law_name = _one_line(m.group(1), 80)
            num = (m.group(2) or "").strip()
            # 의조 번호(예: 조의2)는 여기선 우선 num만 사용
            cands = law_by_name.get(law_name)
            if not cands:
                continue
            for lp in cands:
                if _article_has_number(lp.get("article") or "", num):
                    adds.append(lp["id"])
                    if len(adds) >= max_add:
                        break
            if len(adds) >= max_add:
                break

        if adds:
            # 기존 positive와 합치되 중복 제거(순서 유지)
            existing = r.get("positive_passages", [])
            merged = list(dict.fromkeys(existing + adds))
            r["positive_passages"] = merged


# =========================
# Dedup by query_text
# =========================
def dedup_by_query(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = r.get("query_text", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# =========================
# Main maker
# =========================
def make_pairs(
    law_path: Optional[str],
    admin_path: Optional[str],
    prec_path: Optional[str],
    prec_json_dir: Optional[str] = None,
    out_path: str = "",
    hn_per_q: int = 2,
    seed: int = 42,
    enable_cross_positive: bool = True,
    max_positives_per_prec: int = 5,
    prec_json_glob: str = "**/*.json",
    use_admin_for_prec: bool = False,
) -> None:
    """
    질의-passage 쌍 생성.
    
    Args:
        law_path: 법령 passage JSONL 경로
        admin_path: 행정규칙 passage JSONL 경로
        prec_path: 판례 passage JSONL 경로 (기존 방식)
        prec_json_dir: 판례 원본 JSON 디렉토리 (새로운 방식, prec_path보다 우선)
        out_path: 출력 JSONL 경로
        hn_per_q: 질의당 hard negative 개수
        seed: 랜덤 시드
        enable_cross_positive: 판례→법령 cross positive 활성화
        max_positives_per_prec: 판례당 최대 positive passage 개수
        prec_json_glob: 판례 JSON 파일 검색 패턴
        use_admin_for_prec: 판례→법령/행정규칙 쌍 생성 시 행정규칙 사용 여부 (기본값: False)
    """
    random.seed(seed)

    law = list(read_jsonl(law_path)) if law_path else []
    admin = list(read_jsonl(admin_path)) if admin_path else []
    prec = list(read_jsonl(prec_path)) if prec_path else []

    rows: List[Dict[str, Any]] = []
    rows.extend(build_pairs_from_law(law, hn_per_q) if law else [])
    rows.extend(build_pairs_from_admin(admin, hn_per_q) if admin else [])
    
    # 판례 처리: 새로운 방식(원본 JSON) 우선, 없으면 기존 방식(passage)
    if prec_json_dir:
        # 새로운 방식: 판례 원본 JSON → 법령/행정규칙 passage
        prec_rows = build_pairs_from_precedent_jsons(
            prec_json_dir,
            law,
            admin_passages=admin if use_admin_for_prec else None,
            max_positives=max_positives_per_prec,
            hn_per_q=hn_per_q,
            glob_pattern=prec_json_glob,
            use_admin=use_admin_for_prec,
        )
        rows.extend(prec_rows)
        admin_status = "law+admin" if use_admin_for_prec else "law only"
        print(f"[make_pairs] prec→{admin_status} pairs: {len(prec_rows)} (from {prec_json_dir})")
    elif prec:
        # 기존 방식: 판례 passage → 판례 passage
        prec_rows = build_pairs_from_prec(prec, hn_per_q)
        rows.extend(prec_rows)
        print(f"[make_pairs] prec→prec pairs: {len(prec_rows)} (from prec_passages.jsonl)")

    # 판례 → 법령 cross positive 부여
    if enable_cross_positive and law:
        attach_cross_positives(rows, law, max_add=2)

    # dedup by query_text
    rows = dedup_by_query(rows)

    # assign query_id sequentially for stability
    for i, r in enumerate(rows, 1):
        r["query_id"] = f"Q_{i:05d}"

    write_jsonl(out_path, rows)
    print(f"[make_pairs] total queries: {len(rows)} → {out_path}")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", required=False, help="law_passages.jsonl")
    ap.add_argument("--admin", required=False, help="admin_passages.jsonl")
    ap.add_argument("--prec", required=False, help="prec_passages.jsonl (기존 방식)")
    ap.add_argument("--prec-json-dir", required=False, help="판례 원본 JSON 디렉토리 (새로운 방식, --prec보다 우선)")
    ap.add_argument("--prec-json-glob", default="**/*.json", help="판례 JSON 파일 검색 패턴")
    ap.add_argument("--out", required=True, help="pairs_train.jsonl")
    ap.add_argument("--hn_per_q", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-positives-per-prec", type=int, default=5, help="판례당 최대 positive passage 개수")
    ap.add_argument("--no_cross", action="store_true", help="disable prec→law cross positives")
    ap.add_argument("--use-admin-for-prec", action="store_true", help="판례→법령/행정규칙 쌍 생성 시 행정규칙도 사용 (기본값: 법령만 사용)")
    args = ap.parse_args()

    make_pairs(
        law_path=args.law,
        admin_path=args.admin,
        prec_path=args.prec,
        prec_json_dir=getattr(args, 'prec_json_dir', None),
        out_path=args.out,
        hn_per_q=args.hn_per_q,
        seed=args.seed,
        enable_cross_positive=(not args.no_cross),
        max_positives_per_prec=args.max_positives_per_prec,
        prec_json_glob=args.prec_json_glob,
        use_admin_for_prec=getattr(args, 'use_admin_for_prec', False),
    )

if __name__ == "__main__":
    main()
