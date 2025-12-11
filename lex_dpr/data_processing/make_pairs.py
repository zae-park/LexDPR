# lex_dpr/data_processing/make_pairs.py
from __future__ import annotations
import argparse, json, random, re, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
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
        
        # 법령명 정규화 (공백 정규화, 괄호 제거)
        # normalize_law_name 함수는 나중에 정의되므로 여기서 직접 처리
        law_name_normalized = re.sub(r'\s+', ' ', law_name.strip())
        law_name_normalized = re.sub(r'\([^)]*\)', '', law_name_normalized).strip()
        
        key = (law_name_normalized, article_num, sub_article, paragraph, "law")
        if key in seen:
            continue
        seen.add(key)
        
        refs.append({
            'law_name': law_name_normalized,  # 정규화된 법령명 사용
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
    for p in tqdm(law, desc="  법령 쌍 생성", unit="passage"):
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
    for p in tqdm(admin, desc="  행정규칙 쌍 생성", unit="passage"):
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
    for p in tqdm(prec, desc="  판례 passage 쌍 생성", unit="passage"):
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

def _process_single_precedent_json(
    fp: str,
    law_index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    admin_index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    law_passages: List[Dict[str, Any]],
    admin_passages: List[Dict[str, Any]],
    max_positives: int,
    hn_per_q: int,
    error_log: Optional[List[Tuple[str, str]]] = None,
    failure_reason: Optional[Dict[str, int]] = None,
    failure_samples: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """단일 판례 JSON 파일 처리 (병렬화용 워커 함수)"""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            prec_json = json.load(f)
        
        pair = build_pair_from_precedent_json(
            prec_json,
            law_index,
            admin_index,
            law_passages,
            admin_passages,
            max_positives=max_positives,
            hn_per_q=hn_per_q,
            failure_reason=failure_reason,
            failure_samples=failure_samples,
        )
        return pair
    except json.JSONDecodeError as e:
        if error_log is not None:
            error_log.append((fp, f"JSON 파싱 에러: {str(e)}"))
        return None
    except Exception as e:
        if error_log is not None:
            error_log.append((fp, f"처리 에러: {str(e)}"))
        return None


def build_pairs_from_precedent_jsons(
    prec_json_dir: str,
    law_passages: List[Dict[str, Any]],
    admin_passages: List[Dict[str, Any]] = None,
    max_positives: int = 5,
    hn_per_q: int = 2,
    glob_pattern: str = "**/*.json",
    use_admin: bool = False,
    max_workers: Optional[int] = None,
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
        max_workers: 병렬 처리 워커 수 (None이면 CPU 코어 수)
    
    Returns:
        질의-법령/행정규칙 쌍 리스트
    """
    from pathlib import Path
    import os
    
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
    
    files = sorted(p.glob(glob_pattern))
    if not files:
        print(f"[make_pairs]   경고: {prec_json_dir}에서 JSON 파일을 찾을 수 없습니다 (패턴: {glob_pattern})")
        return []
    
    print(f"[make_pairs]   발견된 판례 JSON 파일: {len(files):,}개")
    
    # 병렬 처리 워커 수 결정
    if max_workers is None:
        max_workers = min(len(files), os.cpu_count() or 4)
    
    rows: List[Dict[str, Any]] = []
    error_log: List[Tuple[str, str]] = []  # 에러 로그 (파일 경로, 에러 메시지)
    failure_reason: Dict[str, int] = {}  # 실패 원인 통계
    failure_samples: List[Dict[str, Any]] = []  # 실패 케이스 샘플 (법령명 매칭 실패)
    
    # 병렬 처리: ThreadPoolExecutor 사용 (I/O + CPU 혼합 작업)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_file = {
            executor.submit(
                _process_single_precedent_json,
                str(fp),
                law_index,
                admin_index,
                law_passages,
                admin_passages,
                max_positives,
                hn_per_q,
                error_log,  # 에러 로그 전달
                failure_reason,  # 실패 원인 통계 전달
                failure_samples,  # 실패 케이스 샘플 전달
            ): fp
            for fp in files
        }
        
        # 진행 상황 표시
        with tqdm(total=len(files), desc="  판례 JSON 처리", unit="file") as pbar:
            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                pbar.update(1)
                try:
                    pair = future.result()
                    if pair:
                        rows.append(pair)
                except Exception as e:
                    error_log.append((str(fp), f"예외 발생: {str(e)}"))
    
    # 통계 출력
    total_files = len(files)
    success_count = len(rows)
    failure_count = total_files - success_count
    
    print(f"[make_pairs]   성공적으로 처리된 판례: {success_count:,}개 ({success_count/max(1, total_files)*100:.1f}%)")
    print(f"[make_pairs]   처리 실패한 판례: {failure_count:,}개 ({failure_count/max(1, total_files)*100:.1f}%)")
    
    # 실패 원인 통계 출력
    if failure_reason:
        print(f"[make_pairs]   실패 원인 분석:")
        total_failures = sum(failure_reason.values())
        for reason, count in sorted(failure_reason.items(), key=lambda x: x[1], reverse=True):
            reason_name = {
                "no_query": "질의 생성 실패 (판시사항/판결요지/사건명 없음)",
                "no_ref_law": "참조조문 없음",
                "no_matched_passage": "법령 인덱스에서 매칭 실패"
            }.get(reason, reason)
            print(f"      - {reason_name}: {count:,}개 ({count/max(1, total_failures)*100:.1f}%)")
    
    # 법령명 매칭 실패 샘플 출력
    if failure_samples:
        print(f"\n[make_pairs]   법령명 매칭 실패 샘플 (최대 20개):")
        unique_failures = {}
        for sample in failure_samples[:100]:  # 최대 100개까지 확인
            key = (sample.get("original_name", ""), sample.get("article_num", ""))
            if key not in unique_failures:
                unique_failures[key] = sample
        
        for i, (key, sample) in enumerate(list(unique_failures.items())[:20], 1):
            print(f"      [{i}] 원본: '{sample.get('original_name', '')}' → 정규화: '{sample.get('normalized_name', '')}'")
            print(f"          조문: 제{sample.get('article_num', '')}조")
            print(f"          실패 이유: {sample.get('reason', '알 수 없음')}")
            if sample.get('available_laws'):
                print(f"          인덱스에 있는 법령 예시: {sample['available_laws'][:3]}")
            if sample.get('available_articles'):
                print(f"          해당 법령의 조문 예시: {sample['available_articles']}")
        
        if len(unique_failures) > 20:
            print(f"      ... 외 {len(unique_failures) - 20}개 실패 케이스")
    
    # 에러 로그 출력
    if error_log:
        print(f"[make_pairs]   경고: {len(error_log):,}개 파일에서 예외 발생")
        if len(error_log) <= 10:
            for fp, err_msg in error_log:
                print(f"      - {Path(fp).name}: {err_msg}")
        else:
            for fp, err_msg in error_log[:10]:
                print(f"      - {Path(fp).name}: {err_msg}")
            print(f"      ... 외 {len(error_log) - 10}개 파일")
    
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

def normalize_law_name(name: str) -> str:
    """
    법령명을 정규화하여 매칭 성공률 향상.
    
    정규화 규칙:
    1. 공백 정규화 (연속 공백 → 단일 공백)
    2. 괄호 내용 제거 (예: "형법(2023.12.31. 시행)" → "형법")
    3. 앞뒤 공백 제거
    """
    if not name:
        return ""
    
    # 공백 정규화
    name = re.sub(r'\s+', ' ', name.strip())
    
    # 괄호 내용 제거 (예: "형법(2023.12.31. 시행)" → "형법")
    name = re.sub(r'\([^)]*\)', '', name)
    name = name.strip()
    
    return name


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
        
        # 법령명 정규화
        normalized_name = normalize_law_name(law_name)
        if not normalized_name:
            continue
        
        # article에서 조문번호 추출: "제355조" → "355", "제355조의2" → "355"
        article_match = re.search(r'제\s*([0-9]+)\s*조', article)
        if not article_match:
            continue
        
        article_num = article_match.group(1)
        
        # 인덱스 구조 생성 (정규화된 법령명 사용)
        if normalized_name not in index:
            index[normalized_name] = {}
        if article_num not in index[normalized_name]:
            index[normalized_name][article_num] = []
        
        index[normalized_name][article_num].append(lp)
    
    return index

def find_law_passages(
    index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    law_name: str,
    article_num: str,
    sub_article: Optional[str] = None,
    paragraph: Optional[str] = None,
    failure_samples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    인덱스에서 법령 passage 검색 (법령명 정규화 적용).
    
    Args:
        index: build_law_index()로 생성한 인덱스
        law_name: 법령명
        article_num: 조문번호 (문자열)
        sub_article: 의조번호 (선택, 현재는 무시)
        paragraph: 항번호 (선택, 현재는 무시)
        failure_samples: 실패 케이스 샘플 수집용 리스트 (선택)
    
    Returns:
        매칭된 passage 리스트
    """
    # 법령명 정규화
    normalized_name = normalize_law_name(law_name)
    
    # 정규화된 법령명으로 검색
    if normalized_name not in index:
        # 실패 케이스 샘플 수집
        if failure_samples is not None and len(failure_samples) < 100:
            failure_samples.append({
                "original_name": law_name,
                "normalized_name": normalized_name,
                "article_num": article_num,
                "reason": "법령명 불일치",
                "available_laws": list(index.keys())[:5] if index else [],  # 샘플만
            })
        return []
    
    if article_num not in index[normalized_name]:
        # 실패 케이스 샘플 수집
        if failure_samples is not None and len(failure_samples) < 100:
            available_articles = list(index[normalized_name].keys())[:5] if normalized_name in index else []
            failure_samples.append({
                "original_name": law_name,
                "normalized_name": normalized_name,
                "article_num": article_num,
                "reason": "조문번호 불일치",
                "available_articles": available_articles,
            })
        return []
    
    # 현재는 조문번호만으로 매칭 (항번호, 의조번호는 나중에 정밀화 가능)
    passages = index[normalized_name][article_num]
    
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
    failure_reason: Optional[Dict[str, int]] = None,
    failure_samples: Optional[List[Dict[str, Any]]] = None,
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
        if failure_reason is not None:
            failure_reason["no_query"] = failure_reason.get("no_query", 0) + 1
        return None
    
    # 2. 참조조문 파싱 (법령 + 행정규칙)
    ref_law_text = prec_json.get("참조조문") or prec_json.get("ref_law") or ""
    refs = parse_reference_laws(ref_law_text)
    
    if not refs:
        if failure_reason is not None:
            failure_reason["no_ref_law"] = failure_reason.get("no_ref_law", 0) + 1
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
            passages = find_law_passages(law_index, name, article_num, sub_article, paragraph, failure_samples=failure_samples)
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
        if failure_reason is not None:
            failure_reason["no_matched_passage"] = failure_reason.get("no_matched_passage", 0) + 1
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

    # cross positive 적용 대상만 필터링
    prec_rows = [r for r in rows if (r.get("meta") or {}).get("type") == "prec"]
    
    for r in tqdm(prec_rows, desc="  cross positive 부여", unit="pair"):
        meta = r.get("meta") or {}
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
# 데이터 쌍 구조 검증 및 통계
# =========================
def validate_pair_structure(
    rows: List[Dict[str, Any]], 
    all_passages: Dict[str, Dict[str, Any]],
    sample_size: int = 5
) -> Dict[str, Any]:
    """
    생성된 쌍의 구조를 검증하고 통계를 반환.
    
    Args:
        rows: 생성된 쌍 리스트
        all_passages: 모든 passage 딕셔너리 {passage_id: passage_dict}
        sample_size: 출력할 샘플 개수
    
    Returns:
        검증 결과 및 통계 딕셔너리
    """
    stats = {
        "total_pairs": len(rows),
        "valid_pairs": 0,
        "invalid_pairs": 0,
        "errors": [],
        "type_distribution": {},
        "positive_count_distribution": {},
        "hard_negative_count_distribution": {},
        "samples": [],
    }
    
    for i, row in enumerate(rows):
        errors = []
        
        # 1. 필수 필드 확인
        if not row.get("query_text"):
            errors.append("query_text 없음")
        if not row.get("positive_passages"):
            errors.append("positive_passages 없음")
        elif not isinstance(row["positive_passages"], list):
            errors.append("positive_passages가 리스트가 아님")
        elif len(row["positive_passages"]) == 0:
            errors.append("positive_passages가 비어있음")
        
        # 2. Positive passage 존재 여부 확인
        if row.get("positive_passages"):
            missing_positives = []
            for pid in row["positive_passages"]:
                if pid not in all_passages:
                    missing_positives.append(pid)
            if missing_positives:
                errors.append(f"존재하지 않는 positive passages: {missing_positives[:3]}")
        
        # 3. Hard negative 존재 여부 확인
        if row.get("hard_negatives"):
            if not isinstance(row["hard_negatives"], list):
                errors.append("hard_negatives가 리스트가 아님")
            else:
                missing_negatives = []
                for nid in row["hard_negatives"]:
                    if nid not in all_passages:
                        missing_negatives.append(nid)
                if missing_negatives:
                    errors.append(f"존재하지 않는 hard negatives: {missing_negatives[:3]}")
        
        # 통계 수집
        if errors:
            stats["invalid_pairs"] += 1
            if len(stats["errors"]) < 10:  # 최대 10개 에러만 저장
                stats["errors"].append({
                    "index": i,
                    "query_text": row.get("query_text", "")[:100],
                    "errors": errors
                })
        else:
            stats["valid_pairs"] += 1
            
            # 타입별 분포
            meta_type = (row.get("meta") or {}).get("type", "unknown")
            stats["type_distribution"][meta_type] = stats["type_distribution"].get(meta_type, 0) + 1
            
            # Positive 개수 분포
            num_positives = len(row.get("positive_passages", []))
            stats["positive_count_distribution"][num_positives] = stats["positive_count_distribution"].get(num_positives, 0) + 1
            
            # Hard negative 개수 분포
            num_negatives = len(row.get("hard_negatives", []))
            stats["hard_negative_count_distribution"][num_negatives] = stats["hard_negative_count_distribution"].get(num_negatives, 0) + 1
            
            # 샘플 수집 (각 타입별로 최대 sample_size개)
            if len([s for s in stats["samples"] if (s.get("meta") or {}).get("type") == meta_type]) < sample_size:
                stats["samples"].append({
                    "query_text": row.get("query_text", ""),
                    "positive_passages": row.get("positive_passages", [])[:5],  # 최대 5개만
                    "hard_negatives": row.get("hard_negatives", [])[:3],  # 최대 3개만
                    "meta": row.get("meta", {})
                })
    
    return stats


def print_validation_report(stats: Dict[str, Any]) -> None:
    """검증 결과를 출력"""
    print("\n" + "="*80)
    print("[make_pairs] 데이터 쌍 구조 검증 결과")
    print("="*80)
    
    print(f"\n📊 전체 통계:")
    print(f"  총 쌍 수: {stats['total_pairs']:,}")
    print(f"  유효한 쌍: {stats['valid_pairs']:,} ({stats['valid_pairs']/max(1, stats['total_pairs'])*100:.1f}%)")
    print(f"  무효한 쌍: {stats['invalid_pairs']:,} ({stats['invalid_pairs']/max(1, stats['total_pairs'])*100:.1f}%)")
    
    if stats['errors']:
        print(f"\n⚠️  에러 사례 (최대 10개):")
        for err in stats['errors'][:10]:
            print(f"  [{err['index']}] {err['query_text']}")
            for e in err['errors']:
                print(f"      - {e}")
    
    if stats['type_distribution']:
        print(f"\n📋 타입별 분포:")
        for type_name, count in sorted(stats['type_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {type_name}: {count:,} ({count/max(1, stats['valid_pairs'])*100:.1f}%)")
    
    if stats['positive_count_distribution']:
        print(f"\n✅ Positive 개수 분포:")
        for count, num_pairs in sorted(stats['positive_count_distribution'].items()):
            print(f"  {count}개: {num_pairs:,} 쌍 ({num_pairs/max(1, stats['valid_pairs'])*100:.1f}%)")
    
    if stats['hard_negative_count_distribution']:
        print(f"\n❌ Hard Negative 개수 분포:")
        for count, num_pairs in sorted(stats['hard_negative_count_distribution'].items()):
            print(f"  {count}개: {num_pairs:,} 쌍 ({num_pairs/max(1, stats['valid_pairs'])*100:.1f}%)")
    
    if stats['samples']:
        print(f"\n📝 샘플 데이터 (각 타입별 최대 5개):")
        for i, sample in enumerate(stats['samples'][:20], 1):  # 최대 20개 출력
            meta_type = (sample.get("meta") or {}).get("type", "unknown")
            print(f"\n  [{i}] 타입: {meta_type}")
            print(f"      질의: {sample['query_text'][:150]}...")
            print(f"      Positive ({len(sample['positive_passages'])}개): {sample['positive_passages']}")
            if sample.get('hard_negatives'):
                print(f"      Hard Negative ({len(sample['hard_negatives'])}개): {sample['hard_negatives']}")
    
    print("\n" + "="*80)


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
    max_workers: Optional[int] = None,
) -> None:
    """
    질의-passage 쌍 생성.
    
    이 함수는 법령, 행정규칙, 판례 데이터로부터 학습용 질의-passage 쌍을 생성합니다.
    생성된 쌍은 Train/Valid/Test로 자동 분할되어 저장됩니다.
    
    Args:
        law_path (Optional[str]):
            법령 passage JSONL 파일 경로.
            예: "data/processed/law_passages.jsonl"
            - 각 passage는 {"id": "LAW_xxx_제n조", "text": "...", ...} 형태
            - None이면 법령 기반 쌍을 생성하지 않음
            
        admin_path (Optional[str]):
            행정규칙 passage JSONL 파일 경로.
            예: "data/processed/admin_passages.jsonl"
            - 각 passage는 {"id": "ADM_xxx_제n조", "text": "...", ...} 형태
            - None이면 행정규칙 기반 쌍을 생성하지 않음
            
        prec_path (Optional[str]):
            판례 passage JSONL 파일 경로 (기존 방식).
            예: "data/processed/prec_passages.jsonl"
            - 각 passage는 {"id": "PREC_xxx_1", "text": "...", ...} 형태
            - prec_json_dir이 지정되면 무시됨 (prec_json_dir 우선)
            - None이면 판례 passage 기반 쌍을 생성하지 않음
            
        prec_json_dir (Optional[str]):
            판례 원본 JSON 파일들이 있는 디렉토리 경로 (새로운 방식, 권장).
            예: "data/precedents"
            - 이 디렉토리 내의 JSON 파일들을 재귀적으로 검색
            - 각 JSON 파일은 판례 원본 데이터 (판시사항, 판결요지, 참조조문 등 포함)
            - prec_path보다 우선순위가 높음 (둘 다 지정되면 이 방식 사용)
            - None이면 판례 원본 JSON 기반 쌍을 생성하지 않음
            
        out_path (str):
            출력 JSONL 파일 경로 (Train 세트).
            예: "data/processed/pairs_train.jsonl"
            - Valid/Test 세트는 자동으로 생성됨:
              - Train: {out_path}
              - Valid: {out_path}_valid.jsonl
              - Test: {out_path}_test.jsonl
            - 분할 비율: Train 80%, Valid 10%, Test 10%
            
        hn_per_q (int, default=2):
            질의당 Hard Negative 개수.
            - 각 질의에 대해 몇 개의 hard negative를 샘플링할지 결정
            - Hard Negative 샘플링 전략:
              * 법령/행정규칙: 같은 법령/규칙의 다른 조문에서 우선 샘플링
              * 판례: 같은 법원의 다른 판례에서 샘플링
            - 권장값: 2~5 (너무 많으면 학습이 어려워질 수 있음)
            
        seed (int, default=42):
            랜덤 시드.
            - Hard Negative 샘플링 및 데이터 셔플링에 사용
            - 재현 가능한 결과를 위해 동일한 시드 사용 권장
            
        enable_cross_positive (bool, default=True):
            판례→법령 Cross Positive 활성화 여부.
            - True: 판례 passage 기반 쌍에서 본문에 인용된 법령을 추가 positive로 연결
            - 예: 판례 본문에 "형법 제355조"가 언급되면 해당 법령 passage를 positive에 추가
            - 최대 2개까지 추가
            - 판례 원본 JSON 방식에서는 이미 참조조문을 사용하므로 효과가 제한적
            
        max_positives_per_prec (int, default=5):
            판례당 최대 Positive Passage 개수 (판례 원본 JSON 방식에서만 사용).
            - 판례의 참조조문에서 파싱한 법령/행정규칙 passage 개수 제한
            - 참조조문이 많아도 이 개수만큼만 positive로 사용
            - 권장값: 3~10 (너무 많으면 학습이 어려워질 수 있음)
            
        prec_json_glob (str, default="**/*.json"):
            판례 JSON 파일 검색 패턴 (glob 패턴).
            - prec_json_dir 내에서 어떤 파일을 검색할지 결정
            - 예: "**/*.json" (모든 하위 디렉토리의 JSON 파일)
            - 예: "*.json" (현재 디렉토리의 JSON 파일만)
            - 예: "**/prec_*.json" (prec_로 시작하는 파일만)
            
        use_admin_for_prec (bool, default=False):
            판례→법령/행정규칙 쌍 생성 시 행정규칙 사용 여부.
            - True: 판례의 참조조문에서 법령과 행정규칙 모두 사용
            - False: 법령만 사용 (기본값)
            - admin_path가 None이면 무시됨
            
        max_workers (Optional[int], default=None):
            병렬 처리 워커 수 (판례 원본 JSON 처리 시).
            - None이면 CPU 코어 수만큼 자동 설정
            - 판례 JSON 파일이 많을 때 처리 속도 향상
            - I/O 집약적 작업이므로 CPU 코어 수보다 많게 설정해도 무방
            
    Returns:
        None (결과는 파일로 저장됨)
        
    출력 파일:
        - {out_path}: Train 세트 (80%)
        - {out_path}_valid.jsonl: Valid 세트 (10%)
        - {out_path}_test.jsonl: Test 세트 (10%)
        
    생성되는 쌍 타입:
        1. law: 법령 기반 쌍
           - 질의: "법령명 제n조의 내용은 무엇인가?"
           - Positive: 해당 법령 passage
           
        2. admin: 행정규칙 기반 쌍
           - 질의: "규칙명 제n조의 내용은 무엇인가?"
           - Positive: 해당 행정규칙 passage
           
        3. prec (기존 방식): 판례 passage 기반 쌍
           - 질의: "사건명의 요지는 무엇인가?"
           - Positive: 해당 판례 passage
           
        4. prec_to_law_admin (새로운 방식, 권장): 판례 원본 JSON 기반 쌍
           - 질의: "판시사항에 대한 법적 판단은?" (판시사항 기반)
           - Positive: 참조조문에서 파싱한 법령/행정규칙 passage들 (최대 max_positives_per_prec개)
           
    사용 예시:
        # 기본 사용 (법령 + 판례 원본 JSON)
        make_pairs(
            law_path="data/processed/law_passages.jsonl",
            prec_json_dir="data/precedents",
            out_path="data/processed/pairs_train.jsonl",
            hn_per_q=2,
            max_positives_per_prec=5
        )
        
        # 모든 타입 포함
        make_pairs(
            law_path="data/processed/law_passages.jsonl",
            admin_path="data/processed/admin_passages.jsonl",
            prec_json_dir="data/precedents",
            out_path="data/processed/pairs_train.jsonl",
            use_admin_for_prec=True,
            hn_per_q=3,
            max_positives_per_prec=5
        )
        
        # 기존 방식 (판례 passage 사용)
        make_pairs(
            law_path="data/processed/law_passages.jsonl",
            prec_path="data/processed/prec_passages.jsonl",
            out_path="data/processed/pairs_train.jsonl",
            enable_cross_positive=True
        )
    """
    t0 = time.time()
    random.seed(seed)

    print("[make_pairs] ===== 질의-passage 쌍 생성 시작 =====")
    print(f"[make_pairs] law_path        = {law_path}")
    print(f"[make_pairs] admin_path      = {admin_path}")
    print(f"[make_pairs] prec_path       = {prec_path}")
    print(f"[make_pairs] prec_json_dir   = {prec_json_dir}")
    print(f"[make_pairs] out_path        = {out_path}")
    print(f"[make_pairs] hn_per_q       = {hn_per_q}")
    print(f"[make_pairs] seed           = {seed}")
    print(f"[make_pairs] use_admin_for_prec = {use_admin_for_prec}")

    # 1) Passage 로드
    law = list(read_jsonl(law_path)) if law_path else []
    admin = list(read_jsonl(admin_path)) if admin_path else []
    prec = list(read_jsonl(prec_path)) if prec_path else []

    print(f"[make_pairs] 로드된 법령 passages: {len(law):,}")
    print(f"[make_pairs] 로드된 행정규칙 passages: {len(admin):,}")
    print(f"[make_pairs] 로드된 판례 passages: {len(prec):,}")

    rows: List[Dict[str, Any]] = []

    # 2) 법령/행정규칙 기반 쌍 생성
    if law:
        print("[make_pairs] 법령 기반 쌍 생성 중...")
        law_rows = build_pairs_from_law(law, hn_per_q)
        rows.extend(law_rows)
        print(f"[make_pairs]   생성된 law pairs: {len(law_rows):,}")

    if admin:
        print("[make_pairs] 행정규칙 기반 쌍 생성 중...")
        admin_rows = build_pairs_from_admin(admin, hn_per_q)
        rows.extend(admin_rows)
        print(f"[make_pairs]   생성된 admin pairs: {len(admin_rows):,}")
    
    # 3) 판례 기반 쌍 생성: 새로운 방식(원본 JSON) 우선, 없으면 기존 방식(passage)
    if prec_json_dir:
        print("[make_pairs] 판례 원본 JSON 기반 쌍 생성 중...")
        prec_rows = build_pairs_from_precedent_jsons(
            prec_json_dir,
            law,
            admin_passages=admin if use_admin_for_prec else None,
            max_positives=max_positives_per_prec,
            hn_per_q=hn_per_q,
            glob_pattern=prec_json_glob,
            use_admin=use_admin_for_prec,
            max_workers=max_workers,
        )
        rows.extend(prec_rows)
        admin_status = "law+admin" if use_admin_for_prec else "law only"
        print(f"[make_pairs] prec→{admin_status} pairs: {len(prec_rows):,} (from {prec_json_dir})")
    elif prec:
        print("[make_pairs] 판례 passage 기반 쌍 생성 중...")
        prec_rows = build_pairs_from_prec(prec, hn_per_q)
        rows.extend(prec_rows)
        print(f"[make_pairs] prec→prec pairs: {len(prec_rows):,} (from prec_passages.jsonl)")

    # 4) 판례 → 법령 cross positive 부여
    if enable_cross_positive and law:
        print("[make_pairs] 판례→법령 cross positive 부여 중...")
        before_pos = sum(len(r.get("positive_passages", [])) for r in rows)
        attach_cross_positives(rows, law, max_add=2)
        after_pos = sum(len(r.get("positive_passages", [])) for r in rows)
        print(f"[make_pairs]   cross positive 적용 전/후 positive 개수 합계: {before_pos:,} → {after_pos:,}")

    # 5) dedup by query_text
    print("[make_pairs] query_text 기준 중복 제거 중...")
    before = len(rows)
    rows = dedup_by_query(rows)
    after = len(rows)
    print(f"[make_pairs]   중복 제거: {before:,} → {after:,}")

    # 6) 데이터 쌍 구조 검증 및 통계
    print("[make_pairs] 데이터 쌍 구조 검증 중...")
    all_passages_dict = {}
    for p in law + admin + prec:
        pid = p.get("id")
        if pid:
            all_passages_dict[pid] = p
    
    validation_stats = validate_pair_structure(rows, all_passages_dict, sample_size=5)
    print_validation_report(validation_stats)
    
    # 검증 실패 시 경고
    if validation_stats["invalid_pairs"] > 0:
        print(f"\n⚠️  경고: {validation_stats['invalid_pairs']:,}개의 무효한 쌍이 발견되었습니다.")
        if validation_stats["invalid_pairs"] / max(1, validation_stats["total_pairs"]) > 0.1:
            print("⚠️  무효한 쌍 비율이 10%를 초과합니다. 데이터를 확인해주세요.")

    # -------------------------
    # Train / Valid / Test split
    # -------------------------
    # query_id 의 마지막 숫자를 기준으로 분할:
    # - 마지막 숫자 8  → valid
    # - 마지막 숫자 9  → test
    # - 나머지        → train
    print("[make_pairs] Train/Valid/Test 분할 중...")
    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for i, r in enumerate(tqdm(rows, desc="  분할 진행", unit="pair"), 1):
        r["query_id"] = f"Q_{i:05d}"
        d = i % 10
        if d == 8:
            valid_rows.append(r)
        elif d == 9:
            test_rows.append(r)
        else:
            train_rows.append(r)

    from pathlib import Path

    out_path_obj = Path(out_path)
    parent = out_path_obj.parent
    stem = out_path_obj.stem
    suffix = out_path_obj.suffix or ".jsonl"

    train_path = out_path_obj
    valid_path = parent / f"{stem}_valid{suffix}"
    test_path = parent / f"{stem}_test{suffix}"

    write_jsonl(str(train_path), train_rows)
    write_jsonl(str(valid_path), valid_rows)
    write_jsonl(str(test_path), test_rows)

    elapsed = time.time() - t0
    
    # 최종 요약 통계
    print("\n" + "="*80)
    print("[make_pairs] 최종 요약")
    print("="*80)
    print(f"\n📊 생성된 쌍 통계:")
    print(f"  총 쌍 수: {len(rows):,}")
    print(f"  Train: {len(train_rows):,} ({len(train_rows)/max(1, len(rows))*100:.1f}%) → {train_path}")
    print(f"  Valid: {len(valid_rows):,} ({len(valid_rows)/max(1, len(rows))*100:.1f}%) → {valid_path}")
    print(f"  Test : {len(test_rows):,} ({len(test_rows)/max(1, len(rows))*100:.1f}%) → {test_path}")
    
    # 타입별 통계
    type_counts = {}
    for row in rows:
        meta_type = (row.get("meta") or {}).get("type", "unknown")
        type_counts[meta_type] = type_counts.get(meta_type, 0) + 1
    
    if type_counts:
        print(f"\n📋 타입별 쌍 수:")
        for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {type_name}: {count:,} ({count/max(1, len(rows))*100:.1f}%)")
    
    # Positive/Hard Negative 통계
    total_positives = sum(len(r.get("positive_passages", [])) for r in rows)
    total_negatives = sum(len(r.get("hard_negatives", [])) for r in rows)
    avg_positives = total_positives / max(1, len(rows))
    avg_negatives = total_negatives / max(1, len(rows))
    
    print(f"\n✅ Positive 통계:")
    print(f"  총 Positive 개수: {total_positives:,}")
    print(f"  쌍당 평균 Positive 개수: {avg_positives:.2f}")
    
    print(f"\n❌ Hard Negative 통계:")
    print(f"  총 Hard Negative 개수: {total_negatives:,}")
    print(f"  쌍당 평균 Hard Negative 개수: {avg_negatives:.2f}")
    
    print(f"\n⏱️  소요 시간: {elapsed:.1f}초")
    print("="*80)
    print("[make_pairs] 완료 ✅")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="질의-passage 쌍 생성 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (법령 + 판례 원본 JSON)
  python -m lex_dpr.data_processing.make_pairs \\
    --law data/processed/law_passages.jsonl \\
    --prec-json-dir data/precedents \\
    --out data/processed/pairs_train.jsonl

  # 모든 타입 포함
  python -m lex_dpr.data_processing.make_pairs \\
    --law data/processed/law_passages.jsonl \\
    --admin data/processed/admin_passages.jsonl \\
    --prec-json-dir data/precedents \\
    --out data/processed/pairs_train.jsonl \\
    --use-admin-for-prec \\
    --hn_per_q 3 \\
    --max-positives-per-prec 5

  # 기존 방식 (판례 passage 사용)
  python -m lex_dpr.data_processing.make_pairs \\
    --law data/processed/law_passages.jsonl \\
    --prec data/processed/prec_passages.jsonl \\
    --out data/processed/pairs_train.jsonl
        """
    )
    
    # 입력 파일 경로
    ap.add_argument(
        "--law",
        required=False,
        help="법령 passage JSONL 파일 경로 (예: data/processed/law_passages.jsonl)"
    )
    ap.add_argument(
        "--admin",
        required=False,
        help="행정규칙 passage JSONL 파일 경로 (예: data/processed/admin_passages.jsonl)"
    )
    ap.add_argument(
        "--prec",
        required=False,
        help="판례 passage JSONL 파일 경로 (기존 방식, 예: data/processed/prec_passages.jsonl). "
             "--prec-json-dir이 지정되면 무시됨"
    )
    ap.add_argument(
        "--prec-json-dir",
        required=False,
        help="판례 원본 JSON 파일들이 있는 디렉토리 경로 (새로운 방식, 권장). "
             "예: data/precedents. --prec보다 우선순위가 높음"
    )
    ap.add_argument(
        "--prec-json-glob",
        default="**/*.json",
        help="판례 JSON 파일 검색 패턴 (glob 패턴, 기본값: **/*.json). "
             "예: '*.json' (현재 디렉토리만), '**/prec_*.json' (prec_로 시작하는 파일만)"
    )
    
    # 출력 경로
    ap.add_argument(
        "--out",
        required=True,
        help="출력 JSONL 파일 경로 (Train 세트). "
             "Valid/Test 세트는 자동으로 생성됨: {out_path}_valid.jsonl, {out_path}_test.jsonl"
    )
    
    # 하이퍼파라미터
    ap.add_argument(
        "--hn_per_q",
        type=int,
        default=2,
        help="질의당 Hard Negative 개수 (기본값: 2). "
             "권장값: 2~5. 너무 많으면 학습이 어려워질 수 있음"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42). 재현 가능한 결과를 위해 동일한 시드 사용 권장"
    )
    ap.add_argument(
        "--max-positives-per-prec",
        type=int,
        default=5,
        help="판례당 최대 Positive Passage 개수 (판례 원본 JSON 방식에서만 사용, 기본값: 5). "
             "판례의 참조조문에서 파싱한 법령/행정규칙 passage 개수 제한. 권장값: 3~10"
    )
    
    # 옵션
    ap.add_argument(
        "--no_cross",
        action="store_true",
        help="판례→법령 Cross Positive 비활성화. "
             "기본적으로 판례 본문에 인용된 법령을 추가 positive로 연결하지만, "
             "이 옵션으로 비활성화 가능"
    )
    ap.add_argument(
        "--use-admin-for-prec",
        action="store_true",
        help="판례→법령/행정규칙 쌍 생성 시 행정규칙도 사용 (기본값: False, 법령만 사용). "
             "--admin이 지정되어 있어야 함"
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="병렬 처리 워커 수 (판례 원본 JSON 처리 시, 기본값: CPU 코어 수). "
             "판례 JSON 파일이 많을 때 처리 속도 향상"
    )
    
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
        max_workers=getattr(args, 'max_workers', None),
    )

if __name__ == "__main__":
    main()
