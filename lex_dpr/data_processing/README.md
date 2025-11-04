# `lex_dpr/data_processing` – Legal Data Pre-processing Pipeline

> 목적: **법령·판례 원시 JSON 데이터를 Sentence-Transformers 파인튜닝에 적합한 형식(`law_passages.jsonl`, `pairs_train.jsonl`)으로 변환**한다.

---

## 📁 디렉토리 구조 제안

```
lex_dpr/data_processing/
├─ __init__.py
├─ README.md                  # ← 본 파일
├─ preprocess_law.py          # 법령 JSON(조문 트리) → passage JSONL
├─ preprocess_prec.py         # 판례 JSON → passage JSONL
├─ make_pairs.py              # 질의-패시지 쌍(pair_train.jsonl) 생성
├─ filters.py                 # “삭제” 조문 필터, 중복/공백 정제
├─ utils_io.py                # 입출력(read/write json, jsonl)
├─ merge_corpus.py            # 법령+판례 passage 병합
└─ validate_dataset.py        # id 정합성 검증(pairs ↔ passages)
```

---

## 🎯 설계 목표

| 목표                               | 설명                                                                      |
| -------------------------------- | ----------------------------------------------------------------------- |
| **표준화된 passage 포맷**              | 모든 문서를 `{"id": ..., "parent_id": ..., "type": ..., "text": ...}` 형태로 변환 |
| **삭제/공백 제거**                     | “삭제”, “삭제됨”, “(삭제)” 등 노이즈 문장을 필터링                                       |
| **ID 규칙 일관화**                    | `LAW_<법령ID>_제xx조_항_호`, `PREC_<판례ID>_n` 등                                |
| **최신 시행본 선택**                    | 중복 조문이 있을 경우 `조문시행일자` 기준 최신 버전 사용                                       |
| **law/passages & pair_train 연결** | `pair_train.jsonl`의 `positive_passages`가 실제 passage id와 매칭되도록           |
| **확장성 확보**                       | JSON 구조가 조금 달라도 공통 Entry point(`convert_*`) 함수에서 통합 처리                  |

---

## 🧩 주요 컴포넌트 설명

### 1️⃣ `preprocess_law.py` – 법령 조문 트리 평탄화

입력: law.go.kr 원본(조문/항/호 트리 JSON)
출력: `law_passages.jsonl`

**핵심 기능**

* 중첩 구조(`조문` → `항` → `호`)를 전개(flatten)
* `삭제` 조문/항/호 필터링
* `조문시행일자` 최신본 선택
* `id` 규칙:
  `LAW_<법령ID>_제{조문번호}조_[항번호]_[호번호]`
* `text` 결합 규칙(항/호 단위 세분화 우선)
* 메타 정보(`law_id`, `law_name`, `article`, `effective_date`) 유지

---

### 2️⃣ `preprocess_prec.py` – 판례 JSON 평탄화

입력: 법원 판례 API 응답(사건 단위 JSON)
출력: `prec_passages.jsonl`

**핵심 기능**

* `판시사항`/`판결요지`/`본문` 등 주요 필드에서 패시지 추출
* 사건번호(`case_number`), 법원명(`court_name`), 선고일자(`judgment_date`) 유지
* `id` 규칙: `PREC_<판례ID>_<n>`
* 짧은 조각/중복 텍스트 제거

---

### 3️⃣ `filters.py` – 노이즈 정리 모듈

* `is_deleted_clause(text)` : “삭제”, “삭제됨”, “(삭제)” 등 패턴 감지
* `normalize_whitespace(text)` : 유니코드/공백 정규화
* `dedup_texts(passages)` : 동일 `text` 중복 제거

---

### 4️⃣ `make_pairs.py` – 학습용 query-passage 쌍 생성

입력: `law_passages.jsonl`, `prec_passages.jsonl`
출력: `pairs_train.jsonl`

**생성 전략**

* (약지도) 조문/판례 제목(`title`/`headnote`)을 query로, 본문을 positive로
* Hard negative: 같은 주제 태그지만 다른 문서, 동일 키워드 포함 오답

---

### 5️⃣ `merge_corpus.py`

* 두 passage 파일 병합 → `merged_corpus.jsonl`
* 중복 id 검사 및 로그 출력

---

### 6️⃣ `validate_dataset.py`

* `pairs_train.jsonl`의 모든 passage id가 `merged_corpus.jsonl`에 존재하는지 검증
* 누락 id, 중복 query id, 빈 텍스트 감지

---

## ⚙️ 전체 워크플로우 예시

```bash
# 1. 법령 전처리
poetry run python -m lex_dpr.data_processing.preprocess_law \
  --src data/statutes/000030.json \
  --out data/processed/law_passages.jsonl

# 2. 판례 전처리
poetry run python -m lex_dpr.data_processing.preprocess_prec \
  --src data/no_action_letters/2015da12345.json \
  --out data/processed/prec_passages.jsonl

# 3. pair 생성
poetry run python -m lex_dpr.data_processing.make_pairs \
  --law data/processed/law_passages.jsonl \
  --prec data/processed/prec_passages.jsonl \
  --out data/processed/pairs_train.jsonl

# 4. 병합 + 검증
poetry run python -m lex_dpr.data_processing.merge_corpus \
  --law data/processed/law_passages.jsonl \
  --prec data/processed/prec_passages.jsonl \
  --out data/processed/merged_corpus.jsonl

poetry run python -m lex_dpr.data_processing.validate_dataset \
  --corpus data/processed/merged_corpus.jsonl \
  --pairs data/processed/pairs_train.jsonl
```

---

## ✅ 베스트 프랙티스

* “삭제”/“개정” 표기 필터 필수 (`filters.is_deleted_clause`)
* passage 단위는 **항/호 수준** 세분화 권장 (Recall@K 유리)
* `조문시행일자` 최신만 사용
* 템플릿(BGE/NONE) 규칙은 학습/평가/서빙에서 일관하게 유지
* 학습 전 `validate_dataset.py`로 id 정합성 검증

---

## 🔮 확장 로드맵

* `preprocess_regulation.py`: 하위법령(시행령/규칙)
* `link_law_prec.py`: 법령–판례 자동 참조 매핑
* `augment_pairs.py`: 질의 패러프레이즈/동의어 증강
* `text_cleaner.py`: 한국어 특화 정제(불용어/형태소)
* `meta_index.py`: 메타 인덱스(법령명/시행일자/조문키)
