#!/usr/bin/env python3
"""
배치 크기 256, max_len 512에서 필요한 GPU 메모리 사용량 계산

bge-m3 모델 사양 (추정):
- Hidden size: 1024
- Layers: 24
- Attention heads: 16
- Vocab size: 250000
- Max position embeddings: 8192
"""

def estimate_memory(batch_size: int, seq_len: int, hidden_size: int = 1024, 
                   num_layers: int = 24, num_heads: int = 16, 
                   vocab_size: int = 250000, use_lora: bool = True):
    """
    GPU 메모리 사용량 추정
    
    Args:
        batch_size: 배치 크기
        seq_len: 시퀀스 길이
        hidden_size: Hidden dimension
        num_layers: Transformer 레이어 수
        num_heads: Attention head 수
        use_lora: LoRA 사용 여부 (True면 base model은 frozen)
    """
    
    # FP32 기준 (AMP 사용 시 절반으로 감소 가능)
    bytes_per_float = 4
    
    print(f"=" * 80)
    print(f"메모리 사용량 추정 (배치 크기: {batch_size}, 시퀀스 길이: {seq_len})")
    print(f"=" * 80)
    
    # 1. Base Model 가중치 (항상 메모리에 로드)
    # bge-m3-ko: ~1.2GB
    base_model_memory = 1.2  # GB
    print(f"\n1. Base Model 가중치: {base_model_memory:.2f} GB")
    print(f"   - LoRA 사용 시에도 base model은 메모리에 로드됨")
    
    # 2. Forward Pass Activation Memory
    # 각 레이어의 activation 저장 (backward pass를 위해)
    # Shape: [batch_size, seq_len, hidden_size] per layer
    activation_per_layer = batch_size * seq_len * hidden_size * bytes_per_float / (1024**3)
    total_activation = activation_per_layer * num_layers
    print(f"\n2. Forward Pass Activation: {total_activation:.2f} GB")
    print(f"   - 레이어당: {activation_per_layer:.4f} GB")
    print(f"   - 계산: {batch_size} × {seq_len} × {hidden_size} × {num_layers} × 4 bytes")
    
    # 3. Attention Memory (Self-Attention)
    # 실제로는 Flash Attention이나 최적화된 attention을 사용하면 메모리가 크게 줄어듦
    # 하지만 backward pass를 위해 일부 activation은 저장되어야 함
    head_dim = hidden_size // num_heads
    
    # Q, K, V matrices (forward pass에서만 필요, backward를 위해 저장)
    qkv_memory = batch_size * num_heads * seq_len * head_dim * 3 * bytes_per_float / (1024**3)
    
    # Attention scores (Flash Attention 사용 시 메모리 효율적)
    # 실제로는 chunk 단위로 계산하므로 전체를 저장하지 않음
    # 하지만 backward를 위해 일부는 저장 필요
    # 보수적으로 전체의 1/4만 저장한다고 가정 (Flash Attention 효과)
    attention_scores_raw = batch_size * num_heads * seq_len * seq_len * bytes_per_float / (1024**3)
    attention_scores = attention_scores_raw * 0.25  # Flash Attention 효과
    
    # Attention output
    attention_output = batch_size * num_heads * seq_len * head_dim * bytes_per_float / (1024**3)
    
    total_attention = (qkv_memory + attention_scores + attention_output) * num_layers
    print(f"\n3. Attention Memory: {total_attention:.2f} GB")
    print(f"   - QKV matrices: {qkv_memory * num_layers:.2f} GB")
    print(f"   - Attention scores (Flash Attention 효과): {attention_scores * num_layers:.2f} GB")
    print(f"     (원래: {attention_scores_raw * num_layers:.2f} GB, Flash Attention으로 75% 절감)")
    print(f"   - Attention output: {attention_output * num_layers:.2f} GB")
    print(f"   - 참고: Flash Attention 미사용 시 {attention_scores_raw * num_layers:.2f} GB 추가 필요")
    
    # 4. Gradient Memory (Backward Pass)
    # LoRA만 학습하지만, activation gradient는 저장됨
    # Activation과 유사한 크기
    gradient_memory = total_activation * 1.0  # Activation과 비슷
    print(f"\n4. Gradient Memory: {gradient_memory:.2f} GB")
    print(f"   - Activation gradient 저장 (LoRA만 학습해도 필요)")
    
    # 5. Optimizer States (AdamW)
    if use_lora:
        # LoRA 파라미터만 저장
        # LoRA rank = 8, target_modules = ["query", "value"] (2개)
        # 각 레이어당: r × hidden_size × 2 (query, value) × 2 (A, B matrices)
        lora_params_per_layer = 8 * hidden_size * 2 * 2  # query, value 각각 A, B
        total_lora_params = lora_params_per_layer * num_layers
        # AdamW: momentum + variance = 파라미터의 2배
        optimizer_memory = total_lora_params * bytes_per_float * 2 / (1024**3)
        print(f"\n5. Optimizer States (LoRA만): {optimizer_memory:.4f} GB")
        print(f"   - LoRA 파라미터: {total_lora_params:,} 개")
        print(f"   - AdamW (momentum + variance): 파라미터의 2배")
    else:
        # 전체 모델 학습 시
        optimizer_memory = base_model_memory * 2  # momentum + variance
        print(f"\n5. Optimizer States (전체 모델): {optimizer_memory:.2f} GB")
    
    # 6. 임베딩 레이어 메모리
    embedding_memory = vocab_size * hidden_size * bytes_per_float / (1024**3)
    print(f"\n6. Embedding Layer: {embedding_memory:.2f} GB")
    print(f"   - Vocab size: {vocab_size:,}")
    
    # 7. 기타 오버헤드 (PyTorch, CUDA 등)
    overhead = 2.0  # GB (추정)
    print(f"\n7. 기타 오버헤드: {overhead:.2f} GB")
    print(f"   - PyTorch, CUDA, 데이터 로더 등")
    
    # 총 메모리 사용량
    total_memory = (base_model_memory + total_activation + total_attention + 
                    gradient_memory + optimizer_memory + embedding_memory + overhead)
    
    print(f"\n" + "=" * 80)
    print(f"총 메모리 사용량: {total_memory:.2f} GB")
    print(f"=" * 80)
    
    # AMP 사용 시 메모리 절감
    if use_lora:
        # AMP는 activation과 gradient를 절반으로 줄임
        amp_savings = (total_activation + gradient_memory) * 0.5
        total_memory_amp = total_memory - amp_savings
        print(f"\nAMP 사용 시 예상 절감: {amp_savings:.2f} GB")
        print(f"AMP 사용 시 총 메모리: {total_memory_amp:.2f} GB")
    
    # Gradient Checkpointing 사용 시
    # Activation을 거의 저장하지 않음 (계산 시에만 사용)
    checkpointing_savings = total_activation * 0.8  # 약 80% 절감
    total_memory_checkpoint = total_memory - checkpointing_savings
    print(f"\nGradient Checkpointing 사용 시 예상 절감: {checkpointing_savings:.2f} GB")
    print(f"Gradient Checkpointing 사용 시 총 메모리: {total_memory_checkpoint:.2f} GB")
    
    # 둘 다 사용 시
    if use_lora:
        total_memory_both = total_memory - amp_savings - checkpointing_savings
        print(f"\nAMP + Gradient Checkpointing 사용 시 총 메모리: {total_memory_both:.2f} GB")
    
    return total_memory


if __name__ == "__main__":
    import sys
    
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    
    print("bge-m3-ko 모델 메모리 사용량 추정")
    print(f"배치 크기: {batch_size}, 시퀀스 길이: {seq_len}\n")
    
    estimate_memory(batch_size=batch_size, seq_len=seq_len, use_lora=True)
    
    print("\n" + "=" * 80)
    print("참고:")
    print("- 실제 메모리 사용량은 모델 구현과 PyTorch 버전에 따라 다를 수 있습니다.")
    print("- Attention 메모리는 시퀀스 길이의 제곱에 비례하여 증가합니다.")
    print("- Gradient Checkpointing을 사용하면 메모리를 크게 절약할 수 있습니다.")
    print("=" * 80)

