# LexDPR 학습 가이드

이 문서는 LexDPR의 학습 데이터 처리 과정과 모델 구조를 상세히 설명합니다.

## 목차

1. [개요](#개요)
2. [학습 데이터 구조](#학습-데이터-구조)
3. [데이터 변환 과정](#데이터-변환-과정)
4. [모델 구조 (BiEncoder)](#모델-구조-biencoder)
5. [학습 과정](#학습-과정)
6. [템플릿의 역할](#템플릿의-역할)
7. [사전 학습된 인코더 사용](#사전-학습된-인코더-사용)

---

## 개요

LexDPR는 **BiEncoder (Dual Encoder)** 구조를 사용하는 법률 문서 검색 모델입니다. 질의(자연어)와 패시지(법률 조항)를 각각 독립적으로 인코딩하여 임베딩 벡터로 변환하고, 코사인 유사도를 계산하여 관련 패시지를 검색합니다.

**핵심 포인트:**
- 학습 데이터는 **텍스트 형태**로 저장됩니다 (ID가 아닌 실제 텍스트)
- 모델은 학습 중 **실시간으로 임베딩을 생성**합니다 (사전 계산된 임베딩이 아님)
- 사전 학습된 인코더(예: `BAAI/bge-m3`)를 초기 가중치로 사용하고, 법률 도메인에 맞게 **파인튜닝**합니다

---

## 학습 데이터 구조

학습에는 두 가지 주요 데이터 파일이 필요합니다:

### 1. `pairs_train.jsonl` - 학습 쌍 데이터

각 행은 하나의 질의와 관련된 양성 패시지 ID, 음성 패시지 ID를 포함합니다.

**실제 예시:**

```json
{
  "query_text": "정보통신망 이용촉진 및 정보보호 등에 관한 법률 제1조의 내용은 무엇인가?",
  "positive_passages": ["LAW_000030_제1조"],
  "hard_negatives": [
    "LAW_000030_제44조_①",
    "LAW_000030_제47조_①",
    "LAW_000030_제73조"
  ],
  "meta": {
    "type": "law",
    "law_name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "article": "제1조",
    "parent_id": "LAW_000030_제1조"
  },
  "query_id": "Q_00001"
}
```

**필드 설명:**
- `query_text`: 자연어 질의 텍스트
- `positive_passages`: 관련된 패시지 ID 리스트 (양성 샘플)
- `hard_negatives`: 관련 없는 패시지 ID 리스트 (음성 샘플, 선택사항)
- `meta`: 질의 생성에 사용된 메타데이터

### 2. `merged_corpus.jsonl` - 패시지 텍스트 매핑

각 행은 패시지 ID와 실제 텍스트를 매핑합니다.

**실제 예시:**

```json
{
  "id": "LAW_000030_제1조",
  "parent_id": "LAW_000030_제1조",
  "type": "법령",
  "law_id": "000030",
  "law_name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
  "article": "제1조",
  "effective_date": "20251001",
  "text": "제1장 총칙"
}
```

```json
{
  "id": "LAW_000030_제2조_①",
  "parent_id": "LAW_000030_제2조",
  "type": "법령",
  "law_id": "000030",
  "law_name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
  "article": "제2조",
  "effective_date": "20251001",
  "text": "1 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2004.1.29, 2007.1.26, 2007.12.21, 2008.6.13, 2010.3.22, 2014.5.28, 2020.6.9>"
}
```

**필드 설명:**
- `id`: 패시지 고유 ID (pairs_train.jsonl의 `positive_passages`/`hard_negatives`와 매칭)
- `text`: 실제 패시지 텍스트 (모델에 입력되는 텍스트)
- 기타 필드: 메타데이터 (법령명, 조항 번호 등)

---

## 데이터 변환 과정

학습 시작 시, `BiEncoderTrainer`는 다음과 같이 데이터를 변환합니다:

### 1단계: ID → 텍스트 변환

```python
# base_trainer.py의 _build_examples() 메서드

for row in self.pairs:
    # 질의 텍스트 (템플릿 적용)
    q_text = tq(row["query_text"], self.template_mode)
    # 예: "Represent this sentence for searching relevant passages: 정보통신망 이용촉진..."
    
    # 양성 패시지 ID로 실제 텍스트 조회
    for pid in row["positive_passages"]:
        passage = self.passages.get(pid)  # merged_corpus.jsonl에서 조회
        if not passage:
            continue
        
        # 패시지 텍스트 (템플릿 적용)
        p_text = tp(passage["text"], self.template_mode)
        # 예: "Represent this sentence for retrieving relevant passages: 제1장 총칙"
        
        # InputExample 생성
        examples.append(InputExample(texts=[q_text, p_text]))
```

**변환 예시:**

| 단계         | 입력                                                                                                  | 출력                                                                                                                                                                                          |
| ------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 원본 데이터  | `query_text`: "정보통신망 이용촉진... 제1조의 내용은?"<br>`positive_passages`: `["LAW_000030_제1조"]` | -                                                                                                                                                                                             |
| ID 조회      | `passages.get("LAW_000030_제1조")`                                                                    | `{"text": "제1장 총칙", ...}`                                                                                                                                                                 |
| 템플릿 적용  | 질의: `tq("정보통신망 이용촉진...", BGE)`<br>패시지: `tp("제1장 총칙", BGE)`                          | 질의: `"Represent this sentence for searching relevant passages: 정보통신망 이용촉진... 제1조의 내용은?"`<br>패시지: `"Represent this sentence for retrieving relevant passages: 제1장 총칙"` |
| InputExample | `InputExample(texts=[q_text, p_text])`                                                                | 학습 가능한 형태                                                                                                                                                                              |

### 2단계: 배치 구성

```python
# sentence-transformers의 DataLoader가 배치로 구성
# 각 배치는 여러 (query, passage) 쌍을 포함
batch = [
    InputExample(texts=[q1, p1]),
    InputExample(texts=[q2, p2]),
    InputExample(texts=[q3, p3]),
    ...
]
```

---

## 모델 구조 (BiEncoder)

### 핵심 개념

LexDPR는 **단일 SentenceTransformer 모델**을 사용하여 질의와 패시지를 모두 인코딩합니다. 

**중요한 점:**
- ❌ **분리된 인코더가 아닙니다**: 질의용 모델과 패시지용 모델이 따로 있지 않습니다
- ✅ **단일 모델 + 템플릿 구분**: 하나의 모델을 사용하되, **템플릿으로 역할을 구분**합니다
- ✅ **bge-m3 계열 모델**: `BAAI/bge-m3` 같은 사전 학습된 단일 모델을 사용합니다

### 다른 BiEncoder 모델과의 차이

일부 BiEncoder 모델(예: **DPR, ColBERT**)은 질의 인코더와 패시지 인코더가 **완전히 분리**되어 있습니다:

```
DPR 구조 (분리된 인코더):
질의 → [Query Encoder (BERT)] → query_embedding
패시지 → [Passage Encoder (BERT)] → passage_embedding
```

하지만 **bge-m3**는 단일 모델을 사용합니다:

```
bge-m3 구조 (단일 모델):
질의 + 템플릿 → [단일 모델] → query_embedding
패시지 + 템플릿 → [단일 모델] → passage_embedding
```

### 아키텍처 다이어그램

```
질의 (자연어)
    ↓
[템플릿 적용]
"Represent this sentence for searching relevant passages: {질의}"
    ↓
[BiEncoder 모델]
    ↓
query_embedding (768차원 벡터)

패시지 (조항 텍스트)
    ↓
[템플릿 적용]
"Represent this sentence for retrieving relevant passages: {패시지}"
    ↓
[BiEncoder 모델]
    ↓
passage_embedding (768차원 벡터)

유사도 계산
    ↓
cosine_similarity(query_embedding, passage_embedding)
```

### 코드 구조

```python
# lex_dpr/models/encoders.py

class BiEncoder:
    def __init__(self, name_or_path: str, template: TemplateMode = TemplateMode.BGE, ...):
        # ⚠️ 하나의 SentenceTransformer 모델만 로드
        # 질의용/패시지용 모델이 분리되어 있지 않음
        self.model = SentenceTransformer(name_or_path)  # 단일 모델
        self.template = template
    
    def encode_queries(self, queries: Iterable[str], batch_size=64):
        # 질의 템플릿 적용
        texts = [tq(q, self.template) for q in queries]
        # ⚠️ 같은 모델(self.model)을 사용하여 인코딩
        return self.model.encode(texts, ...)
    
    def encode_passages(self, passages: Iterable[str], batch_size=64):
        # 패시지 템플릿 적용
        texts = [tp(p, self.template) for p in passages]
        # ⚠️ 같은 모델(self.model)을 사용하여 인코딩
        return self.model.encode(texts, ...)
```

**핵심 포인트:**
1. ✅ **단일 모델**: `self.model`은 하나의 `SentenceTransformer` 인스턴스입니다
2. ✅ **템플릿으로 구분**: 질의와 패시지에 다른 템플릿을 적용하여 모델이 역할을 구분합니다
3. ✅ **같은 가중치 공유**: 질의와 패시지 인코딩이 같은 모델 파라미터를 사용합니다

### 실제 동작 예시

```python
# 모델 초기화 (단일 모델만 로드)
encoder = BiEncoder("BAAI/bge-m3", template=TemplateMode.BGE)

# 질의 인코딩
query = "형법 제355조 위반에 대한 법적 판단은?"
query_with_template = "Represent this sentence for searching relevant passages: 형법 제355조 위반에 대한 법적 판단은?"
query_embedding = encoder.model.encode([query_with_template])  # 단일 모델 사용

# 패시지 인코딩
passage = "제355조(절도) ① 타인의 재물을 절취한 자는..."
passage_with_template = "Represent this sentence for retrieving relevant passages: 제355조(절도) ① 타인의 재물을 절취한 자는..."
passage_embedding = encoder.model.encode([passage_with_template])  # 같은 모델 사용

# 유사도 계산
similarity = cosine_similarity(query_embedding, passage_embedding)
```

**중요:** `encoder.model`은 질의와 패시지 모두에 대해 **동일한 모델 인스턴스**입니다. 템플릿만 다르게 적용하여 모델이 각각의 역할을 이해하도록 합니다.

### 단일 모델 vs 분리된 인코더: 임베딩 품질 비교

#### 단일 모델 (bge-m3 방식)의 장단점

**장점:**
1. ✅ **파라미터 효율성**: 하나의 모델만 학습/저장하므로 메모리와 저장 공간 절약
2. ✅ **학습 데이터 효율성**: 질의와 패시지가 같은 임베딩 공간을 공유하여 데이터 활용도 높음
3. ✅ **일관된 임베딩 공간**: 질의와 패시지가 동일한 공간에 매핑되어 비교가 직관적
4. ✅ **일반화 능력**: 다양한 도메인에서 좋은 성능 (BGE-m3, E5 등이 이를 입증)
5. ✅ **템플릿으로 역할 구분**: 템플릿만으로 질의/패시지 역할을 효과적으로 구분 가능

**단점:**
1. ❌ **특화 제한**: 질의와 패시지의 특성이 매우 다를 때 최적화 어려움
2. ❌ **템플릿 의존성**: 템플릿 설계가 성능에 큰 영향

#### 분리된 인코더 (DPR 방식)의 장단점

**장점:**
1. ✅ **입력별 최적화**: 질의와 패시지 각각에 특화된 인코더로 더 정밀한 학습 가능
2. ✅ **유연성**: 질의 인코더와 패시지 인코더를 독립적으로 개선/교체 가능
3. ✅ **도메인 특화**: 질의와 패시지의 특성이 매우 다를 때 유리

**단점:**
1. ❌ **파라미터 수 증가**: 두 배의 모델 파라미터 필요
2. ❌ **학습 데이터 요구**: 각 인코더를 효과적으로 학습하려면 더 많은 데이터 필요
3. ❌ **복잡성**: 두 모델의 학습/관리/배포가 복잡
4. ❌ **임베딩 공간 불일치**: 질의와 패시지가 다른 공간에 매핑되어 비교가 덜 직관적

#### 실제 성능 비교

**최근 연구 동향 (2023-2024):**

1. **BGE-m3, E5-v2 등 단일 모델**: 
   - MS MARCO, BEIR 벤치마크에서 **최고 성능** 기록
   - 템플릿 기반 접근으로 분리된 인코더와 유사하거나 더 나은 성능

2. **DPR (분리된 인코더)**:
   - 대규모 데이터셋(수백만 쌍)에서 강점
   - 질의와 패시지의 특성이 매우 다를 때 유리 (예: 질의는 짧고, 패시지는 매우 긴 경우)

3. **실무적 선택 기준**:
   - **단일 모델 선택 시기**: 
     - 데이터가 중간 규모 이하 (수만~수십만 쌍)
     - 질의와 패시지의 길이/특성이 유사
     - 리소스 제약 (GPU 메모리, 저장 공간)
     - 빠른 프로토타이핑 및 배포
   
   - **분리된 인코더 선택 시기**:
     - 대규모 데이터셋 (수백만 쌍 이상)
     - 질의와 패시지의 특성이 매우 다름 (예: 짧은 질의 vs 긴 문서)
     - 충분한 리소스와 학습 시간

#### LexDPR의 선택: 단일 모델

LexDPR는 **단일 모델(bge-m3) + 템플릿** 방식을 선택한 이유:

1. ✅ **법률 도메인 특성**: 질의와 패시지 모두 법률 텍스트로 특성이 유사
2. ✅ **데이터 규모**: 수만~수십만 쌍 규모로 단일 모델이 효율적
3. ✅ **검증된 성능**: BGE-m3가 다양한 검색 태스크에서 최고 성능
4. ✅ **실용성**: 배포와 유지보수가 간단

**결론**: 일반적으로 **단일 모델 + 템플릿** 방식이 더 효율적이고 실용적이며, 최근 연구에서도 우수한 성능을 보입니다. 분리된 인코더는 특수한 경우(매우 다른 특성, 대규모 데이터)에만 고려하는 것이 좋습니다.

---

## 학습 과정

### 1. 모델 초기화

```python
# base_trainer.py의 _build_encoder()

encoder = get_bi_encoder(
    self.cfg.model.bi_model,  # 예: "BAAI/bge-m3" 또는 "BAAI/bge-m3-ko"
    template=self.template_mode.value,  # "bge"
    max_len=max_len if max_len > 0 else None,
)
```

- 사전 학습된 모델을 HuggingFace에서 로드
- 템플릿 모드 설정 (BGE 권장)
- 최대 시퀀스 길이 설정

### 2. Loss 함수 구성

```python
# MultipleNegativesRankingLoss 사용
loss = losses.MultipleNegativesRankingLoss(
    model=encoder.model,
    scale=temperature  # 기본값: 0.05
)
```

**MultipleNegativesRankingLoss 동작 방식:**
- 배치 내에서 각 질의에 대해:
  - 양성 패시지와의 유사도 ↑ (maximize)
  - 배치 내 다른 샘플들의 패시지를 negative로 사용하여 유사도 ↓ (minimize)
- In-batch negative sampling으로 별도의 negative 샘플이 없어도 학습 가능

### 3. 학습 루프

```python
# sentence-transformers의 fit() 메서드 사용

encoder.model.fit(
    train_objectives=[(dataloader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    scheduler="warmupcosine",
    optimizer_params={"lr": learning_rate},
    use_amp=True,  # Mixed precision training
    evaluator=evaluator,  # 선택사항
    evaluation_steps=eval_steps,  # 선택사항
)
```

### 4. 학습 중 데이터 흐름

```
1. DataLoader에서 배치 로드
   ↓
   batch = [
       InputExample(texts=[q1, p1]),
       InputExample(texts=[q2, p2]),
       ...
   ]

2. 모델 Forward Pass
   ↓
   query_embeddings = model.encode([q1, q2, ...])  # 실시간 임베딩 생성
   passage_embeddings = model.encode([p1, p2, ...])  # 실시간 임베딩 생성

3. Loss 계산
   ↓
   loss = MultipleNegativesRankingLoss(
       query_embeddings,
       passage_embeddings,
       # 배치 내 다른 샘플들을 negative로 사용
   )

4. Backward Pass
   ↓
   loss.backward()
   optimizer.step()
   # 모델 파라미터 업데이트
```

**핵심:** 학습 중에는 **텍스트를 입력받아 실시간으로 임베딩을 생성**합니다. 사전 계산된 임베딩을 사용하지 않습니다.

---

## 템플릿의 역할

템플릿은 질의와 패시지를 구분하고, 모델이 각각의 역할을 이해하도록 돕습니다.

### BGE 템플릿 (권장)

```python
# lex_dpr/models/templates.py

BGE_Q = "Represent this sentence for searching relevant passages: {q}"
BGE_P = "Represent this sentence for retrieving relevant passages: {p}"
```

**예시:**

| 원본 텍스트                                                                       | 템플릿 적용 후                                                                                                                       |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 질의: `"정보통신망 이용촉진 및 정보보호 등에 관한 법률 제1조의 내용은 무엇인가?"` | `"Represent this sentence for searching relevant passages: 정보통신망 이용촉진 및 정보보호 등에 관한 법률 제1조의 내용은 무엇인가?"` |
| 패시지: `"제1장 총칙"`                                                            | `"Represent this sentence for retrieving relevant passages: 제1장 총칙"`                                                             |

### 템플릿이 중요한 이유

1. **역할 구분**: 질의와 패시지를 명확히 구분
2. **성능 향상**: bge-m3 계열 모델은 BGE 템플릿을 사용했을 때 최적 성능
3. **일관성**: 학습과 추론 시 동일한 템플릿 사용으로 재현성 보장

**주의:** 학습과 추론 시 **반드시 동일한 템플릿**을 사용해야 합니다.

---

## 사전 학습된 인코더 사용

LexDPR는 사전 학습된 인코더를 초기 가중치로 사용하고, 법률 도메인에 맞게 파인튜닝합니다.

### 지원 모델

```python
# lex_dpr/models/factory.py

ALIASES = {
    "bge-m3": "BAAI/bge-m3",
    "bge-m3-ko": "BAAI/bge-m3",  # 한국어 최적화
    "ko-simcse": "jhgan/ko-sroberta-multitask",
    "multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ...
}
```

### 파인튜닝 과정

1. **초기화**: 사전 학습된 모델 가중치 로드
2. **학습**: 법률 도메인 데이터로 파인튜닝
   - 질의-패시지 쌍 데이터로 학습
   - 법률 용어와 문맥에 맞게 임베딩 공간 조정
3. **평가**: 검색 성능 메트릭 (MRR@k, NDCG@k, Recall@k 등)으로 평가

### PEFT (LoRA) 지원

메모리 제약이 있는 경우, LoRA 어댑터를 사용하여 일부 파라미터만 학습할 수 있습니다:

```yaml
# configs/base.yaml
model:
  bi_model: BAAI/bge-m3
  peft:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
```

---

## 요약

### 데이터 흐름

| 단계             | 입력                         | 처리                                  | 출력                         |
| ---------------- | ---------------------------- | ------------------------------------- | ---------------------------- |
| **데이터 준비**  | Passage ID                   | `merged_corpus.jsonl`에서 텍스트 조회 | Passage 텍스트               |
| **템플릿 적용**  | 질의 텍스트 + Passage 텍스트 | BGE 템플릿 적용                       | `InputExample(texts=[q, p])` |
| **모델 Forward** | 텍스트 쌍                    | 모델이 실시간 임베딩 생성             | 임베딩 벡터                  |
| **Loss 계산**    | 임베딩                       | 배치 내 negative 샘플링               | Loss 값                      |
| **Backward**     | Loss                         | 역전파                                | 모델 파라미터 업데이트       |

### 핵심 포인트

1. ✅ 학습 데이터는 **텍스트 형태**로 저장 (ID가 아닌 실제 텍스트)
2. ✅ 모델은 학습 중 **실시간으로 임베딩 생성** (사전 계산 아님)
3. ✅ 사전 학습된 인코더를 초기 가중치로 사용하고 **파인튜닝**
4. ✅ **단일 모델 사용**: 질의와 패시지가 **같은 모델**을 사용 (분리된 인코더 아님)
5. ✅ **템플릿으로 구분**: 질의와 패시지에 **다른 템플릿** 적용하여 역할 구분
6. ✅ 학습과 추론 시 **동일한 템플릿** 사용 필수

### 다음 단계

- 학습 실행: `poetry run lex-dpr train`
- 설정 확인: `poetry run lex-dpr config show`
- 모델 평가: 학습 중 자동 평가 또는 별도 평가 스크립트 실행
- 임베딩 추출: `poetry run lex-dpr embed`

---

## 참고 자료

- [sentence-transformers 문서](https://www.sbert.net/)
- [BGE 모델 논문](https://arxiv.org/abs/2309.07597)
- [Multiple Negatives Ranking Loss 설명](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss)

