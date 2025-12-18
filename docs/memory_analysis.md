# GPU 메모리 사용량 분석 (bge-m3-ko)

## 모델 정보
- Base Model 크기: ~1.2GB (가중치만)
- Hidden Size: 1024
- Layers: 24
- Max Sequence Length: 8192 (지원), 실제 사용: 256-512

## 메모리 사용량 구성 요소

### 1. Base Model 가중치
- **크기**: ~1.2GB
- **설명**: LoRA를 사용해도 base model은 메모리에 로드됨

### 2. Activation Memory (Forward Pass)
```
Activation ≈ Batch Size × Seq Length × Hidden Size × Layers × 4 bytes
```

**예시 계산**:
- Batch Size: 256
- Sequence Length: 512
- Hidden Size: 1024
- Layers: 24
- Activation ≈ 256 × 512 × 1024 × 24 × 4 bytes ≈ **12.6 GB**

### 3. Attention Memory (Self-Attention)
```
Attention ≈ Batch Size × Seq Length² × Heads × 4 bytes
```

**예시 계산**:
- Batch Size: 256
- Sequence Length: 512
- Attention Heads: 16
- Attention ≈ 256 × 512² × 16 × 4 bytes ≈ **2.1 GB**

### 4. Gradient Memory (Backward Pass)
- LoRA만 학습하지만 activation gradient는 저장됨
- Activation과 유사한 크기: **~12.6 GB**

### 5. Optimizer States (AdamW)
- LoRA 파라미터만 저장: **~50-100 MB** (매우 작음)

## 총 메모리 사용량 추정

### 배치 크기 256, 시퀀스 길이 512
```
Base Model:        1.2 GB
Activation:       12.6 GB
Attention:         2.1 GB
Gradient:         12.6 GB
Optimizer:         0.1 GB
─────────────────────────
총합:              ~28.6 GB
```

### 배치 크기 128, 시퀀스 길이 512
```
Base Model:        1.2 GB
Activation:        6.3 GB  (절반)
Attention:         1.1 GB  (절반)
Gradient:          6.3 GB  (절반)
Optimizer:         0.1 GB
─────────────────────────
총합:              ~15.0 GB
```

### 배치 크기 64, 시퀀스 길이 256
```
Base Model:        1.2 GB
Activation:        1.6 GB  (1/4 배치, 1/2 시퀀스)
Attention:         0.3 GB  (1/4 배치, 1/4 시퀀스²)
Gradient:          1.6 GB
Optimizer:         0.1 GB
─────────────────────────
총합:              ~4.8 GB
```

## OOM 발생 원인

1. **배치 크기가 너무 큼**: 256은 H100에서도 부담스러울 수 있음
2. **시퀀스 길이가 길음**: 512는 attention 메모리를 제곱으로 증가시킴
3. **AMP가 활성화되어 있지만**: 일부 연산은 여전히 FP32로 수행됨
4. **Gradient Checkpointing 비활성화**: 메모리 절약 기법이 꺼져있음

## 해결 방안

### 1. 배치 크기 줄이기
```yaml
data.batches.bi:
  values: [16, 32, 64]  # 128, 256 제거
```

### 2. 시퀀스 길이 줄이기
```yaml
model.max_len:
  values: [128, 256]  # 384, 512 제거
```

### 3. Gradient Checkpointing 활성화
```yaml
trainer.gradient_checkpointing: true
```
- 메모리 사용량을 ~50% 감소시킬 수 있음
- 학습 속도는 약간 느려짐

### 4. Gradient Accumulation 사용
- sentence-transformers는 지원하지 않지만, 배치 크기를 줄이고 여러 번 forward 후 backward를 수행하면 유사한 효과

### 5. AMP 최적화 확인
- `use_amp: true`가 제대로 작동하는지 확인
- 일부 연산이 FP32로 수행되지 않는지 확인

## 권장 설정 (H100 100GB 기준)

```yaml
data.batches.bi: 64  # 또는 32
model.max_len: 256   # 또는 128
trainer.gradient_checkpointing: true
trainer.use_amp: true
```

이 설정으로 메모리 사용량을 ~10-15GB로 줄일 수 있습니다.

