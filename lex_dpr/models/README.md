# `lex_dpr/models` – Architecture & Usage Guide

> 목적: **bge-m3(-ko) 기반 BI 파이프라인**을 중심으로, 학습(파인튜닝)과 추론(임베딩)에서 재사용 가능한 모델 레이어를 정리합니다.

---

## 디렉토리 개요

```
lex_dpr/models/
├─ factory.py          # 모델 로더(레지스트리/alias) – BiEncoder 생성 진입점
├─ encoders.py         # BiEncoder 래퍼(쿼리/패시지 템플릿, normalize, max_len)
└─ templates.py        # BGE/None 템플릿(쿼리/패시지 프롬프트)

lex_dpr/training/
├─ bi_encoder.py       # 학습 루틴 (train_bi)
├─ augment.py          # 질의/패시지 간단 증강 유틸
├─ collators.py        # DataLoader collate 함수
├─ datasets.py         # PairDataset, HN 옵션
├─ distill.py          # 지식 증류 loss
├─ grad.py             # gradient clipping 등
├─ losses.py           # MultipleNegativesRankingLoss 래핑
├─ miner.py            # hard negative 마이너(옵션)
├─ optim.py            # AdamW 등 optimizer 빌더
├─ peft.py             # LoRA/IA3 어댑터 부착
├─ scheduler.py        # Warmup + cosine 스케줄러 빌더
└─ types.py            # (선택) Pydantic 기반 설정 스키마

lex_dpr/utils/
├─ checkpoint.py       # 베스트 스코어 저장 헬퍼
├─ export.py           # SentenceTransformer 저장/내보내기
├─ textnorm.py         # 텍스트 정규화
└─ tokenizer.py        # 토크나이저 로더
```

> **원칙**: 모델 래퍼/템플릿은 `models/`, 학습 파이프라인은 `training/`, 범용 헬퍼는 `utils/`, 서빙/인덱싱은 `embed/`로 분리합니다. (FAISS는 `embed/`)

---

## 설계 철학

* **단일 진입점**: `factory.get_bi_encoder()` 하나로 모델 초기화(템플릿/정규화/길이 설정 포함)
* **템플릿 일관성**: bge 계열 권장 프롬프트를 학습·평가·추론에 동일 적용
* **안전 가드**: `max_seq_length`, 텍스트 정규화(옵션), 데이터 검증으로 재현성 강화
* **확장 포인트**: LoRA(PEFT), 하드 네거티브, 스케줄러, ONNX export 등은 모듈 단위로 온디맨드 도입

---

## 파일별 상세 설명

### 1) `templates.py`

* 역할: 임베딩 품질 안정화를 위한 **프롬프트 템플릿** 제공
* 모드:

  * `TemplateMode.BGE` (권장):

    * Query: `"Represent this sentence for searching relevant passages: {q}"`
    * Passage: `"Represent this sentence for retrieving relevant passages: {p}"`
  * `TemplateMode.NONE`: 템플릿 미적용
* 헬퍼: `tq(q, mode)`, `tp(p, mode)`로 문자열에 모드 적용

**언제 쓰나?**

* bge-m3(-ko) 파인튜닝/추론 시 **항상 동일한 템플릿**을 쓰는 것이 핵심. 미적용 대비 Recall/MRR이 안정적으로 나옴.

---

### 2) `encoders.py`

* 역할: Sentence-Transformers 래퍼 **`BiEncoder`**
* 주요 기능:

  * `encode_queries(queries, batch_size)` / `encode_passages(passages, batch_size)`
  * 내부에서 템플릿 적용 + `normalize_embeddings=True` 인코딩
  * `max_seq_length` 설정 지원(긴 조문/판례에 대비)
* 생성자 옵션:

  * `name_or_path`: HF 모델 경로(예: `BAAI/bge-m3`)
  * `template`: `TemplateMode` (기본 BGE)
  * `normalize`: 코사인/IP 대비 정규화 여부(기본 True)
  * `max_seq_length`: 토크나이저 컷오프 길이(기본 모델 상한)

**팁**: 평가/서빙에서도 동일한 `template`/`max_seq_length`를 유지하세요.

---

### 3) `factory.py`

* 역할: 모델 **레지스트리/alias** → `BiEncoder` 생성 진입점
* 제공 함수:

  * `get_bi_encoder(name: str, template: str = "bge", normalize: bool = True, max_len: int | None = None) -> BiEncoder`
* alias 예시:

  * `bge-m3-ko` → `BAAI/bge-m3` (ko 가중치가 별도 레포에 없다면 기본 m3를 사용)

**언제 쓰나?**

* `scripts/train_cfg.py` 또는 CLI에서 모델명을 문자열로만 넘기고 싶을 때 안정적인 초기화를 보장.

---

### 4) `datasets.py`

* 역할: 파인튜닝용 샘플 생성
* `PairDataset` (기본):

  * 입력: `pairs.jsonl` + `passages: Dict[id, {text,...}]`
  * 출력 샘플:

    * 기본 모드: `(q, p_pos)`
    * HN 모드(`use_hard_negatives=True`): `(q, p_pos, [p_neg...])`
  * 옵션:

    * `use_bge_template`: 쿼리/패시지에 템플릿 적용
    * `use_hard_negatives`: pairs의 `hard_negatives` 필드 사용
    * `normalize_text`: 유니코드/공백 정규화(옵션; 필요 시 `lex_dpr/utils/text.py` 참조)

**권장 흐름**: 데이터가 적을 때는 **in-batch negatives**(기본 MNRLoss)만으로 충분. HN은 성능이 정체될 때 도입.

---

### 5) `collators.py`

* 역할: `DataLoader`에서 배치를 **Sentence-Transformers `InputExample`**로 변환
* 제공:

  * `mnr_collate(batch)`: `(q, p)` → `[InputExample(texts=[q, p])]`
  * `mnr_with_hn_collate(batch)`: `(q, p, [neg...])` → `texts=[q, p, *neg]`

> 주의: 기본 `MultipleNegativesRankingLoss`는 (q,p) 2-튜플을 전제로 설계됨.
> HN을 명시적으로 쓰려면 커스텀 로스(Triplet, InfoNCE 변형) 또는 ST의 호환 기능을 확인하세요.

---

### 6) `losses.py`

* 역할: 로스 구성 간소화
* 제공:

  * `build_mnr_loss(model, temperature=0.05)` → `MultipleNegativesRankingLoss`
* 검증: temperature > 0 확인

**왜 필요?**

* 로스 초기화/검증을 한 곳에서 통일해 실수 방지, 이후 커스텀 로스(예: HN 가중)로 교체 용이.

---

### 7) `peft.py` (선택)

* 역할: **LoRA/IA3 등 어댑터**를 Sentence-Transformers 내부 Transformer에 부착
* 제공:

  * `attach_lora_to_st(st_model, r=16, alpha=32, dropout=0.05, target_modules=[...])` → ST 내 HF 모델에 LoRA 장착
  * `enable_lora_only_train(st_model)` → LoRA 파라미터만 학습하도록 동결

**언제 쓰나?**

* VRAM/시간 제약이 있거나, bge-m3-ko를 소량 도메인 데이터로 빠르게 파인튜닝할 때.
* `target_modules`는 모델 구조에 맞게(`q_proj`, `v_proj` 혹은 `query`,`value`) 조정.

---

### 8) `export.py`

* 역할: **저장/로드 표준화** + (선택) ONNX 내보내기
* 제공:

  * `save_sentence_transformer(model, out_dir)` / `load_sentence_transformer(path)`
  * `export_transformer_to_onnx(model, onnx_path, opset=17, max_seq_len=512)`

**권장**: ONNX는 보통 **Transformer encoder만** 내보내고, mean-pooling은 런타임(PyTorch/NumPy)에서 처리.

---

### 9) `schedulers.py` (선택)

* 역할: 직접 학습 루프 도입 시 스케줄러 생성
* 제공:

  * `build_warmup_cosine(optimizer, total_steps, warmup_ratio=0.1, min_warmup=10)`

**현재**: ST의 `fit()` 스케줄러로 충분. 커스텀 루프/혼합 전략이 필요해질 때 사용.

---

### 10) `types.py` (선택)

* 역할: Pydantic 기반 설정 스키마로 **IDE 지원/유효성 검사** 향상
* 예시:

  * `BiModelCfg(name, template, normalize, max_len)`
  * `TrainerCfg(epochs, lr, batch_size, temperature, eval_steps)`

**현재**: OmegaConf로도 충분. 정적 타입 지원이 필요하면 추가.

---

## 학습 스크립트와의 연결 (예: `scripts/train_cfg.py`)

* **모델 초기화**

  ```python
  from lex_dpr.models.factory import get_bi_encoder
  enc = get_bi_encoder(cfg.model.bi_model, template="bge", max_len=cfg.model.max_len)
  model = enc.model  # SentenceTransformer 인스턴스
  ```

* **데이터셋/로더**

  ```python
  from lex_dpr.training.datasets import PairDataset
  from lex_dpr.training.collators import mnr_collate

  ds = PairDataset(cfg.data.pairs, passages, use_bge_template=True, use_hard_negatives=False)
  loader = DataLoader(ds, batch_size=cfg.data.batches.bi, shuffle=True, collate_fn=mnr_collate)
  ```

* **로스**

  ```python
  from lex_dpr.training.losses import build_mnr_loss
  loss = build_mnr_loss(model, temperature=cfg.trainer.temperature)
  ```

* **LoRA(옵션)**

  ```python
  from lex_dpr.training.peft import attach_lora_to_st, enable_lora_only_train
  model = attach_lora_to_st(model, r=16, alpha=32, dropout=0.05, target_modules=["q_proj","v_proj"])
  enable_lora_only_train(model)
  ```

* **저장/로드**

  ```python
  from lex_dpr.utils.export import save_sentence_transformer
  save_sentence_transformer(model, os.path.join(cfg.out_dir, "bi_encoder"))
  ```

---

## 모범 설정 (bge-m3-ko)

```yaml
# configs/model.yaml
model:
  bi_model: BAAI/bge-m3
  use_bge_template: true
  max_len: 512
```

```yaml
# configs/base.yaml (요지)
trainer:
  epochs: 3
  lr: 2e-5
  temperature: 0.05
  eval_pairs: data/pairs_eval.jsonl
  eval_steps: 300
  k: 10
  k_values: [1,3,5,10]

data:
  passages: data/merged_corpus.jsonl
  pairs: data/pairs_train.jsonl
  batches:
    bi: 64
```

---

## 베스트 프랙티스 & 체크리스트

* **템플릿 일관성**: 학습/평가/서빙에서 동일 모드(BGE/NONE) 유지
* **정규화 유지**: `normalize_embeddings=True` (코사인/IP 안정화)
* **길이 관리**: `max_seq_length`를 모델/도메인에 맞게 설정(법령·판례는 512 권장)
* **in-batch negatives 우선**: 데이터 적을 때는 기본 MNRLoss로 충분
* **평가 주기**: `InformationRetrievalEvaluator`를 `evaluation_steps`로 주기 실행
* **베스트 저장**: (옵션) evaluator 결과를 기준으로 스냅샷 저장
* **HN 도입 시**: 먼저 오프라인 마이닝 → 성능 정체일 때 점진 도입

---

## 향후 확장 로드맵

* `miners.py`(오프라인 HN 마이너) → 데이터 커질 때 도입
* `validators.py`(pairs/passages 교차 검증) → 학습 전 빠른 실패 패턴
* `distill.py`(teacher→student) → 경량화/서빙비용 최적화
* `export.py` ONNX + mean-pooling 내장 버전(필요 시)

---

## FAQ

**Q. CE(크로스 인코더)는 어디에?**
A. 본 레이어는 BI 중심입니다. CE는 `models/rerankers.py` 등으로 분리해 선택적으로 붙이세요.

**Q. bge-m3-ko를 꼭 써야 하나요?**
A. ko 전용 체크포인트가 없다면 `BAAI/bge-m3` + BGE 템플릿으로도 좋은 성능이 납니다.

**Q. HN 샘플이 많은데 바로 써도 되나요?**
A. 초기에는 in-batch negatives로 학습–검증을 먼저 완료하고, 이후 HN을 점진 도입하는 것을 권장합니다.

---

## 변경 이력

* v0.1: 기본 파일 세트 도입(templates/encoders/factory/datasets/collators/losses) + 선택(peft/export/schedulers/types)
* v0.1.1: PairDataset에 HN/정규화 옵션, BiEncoder에 max_seq_length 옵션.
