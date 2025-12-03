# lex_dpr/visualization/embedding_quality.py
"""
임베딩 품질 시각화 모듈

임베딩 품질을 증명하고 시각화하는 다양한 도구를 제공합니다:
1. 임베딩 공간 시각화 (t-SNE, UMAP)
2. 유사도 분포 분석 (Positive vs Negative)
3. 히트맵 시각화
4. 학습 전후 비교
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn이 설치되지 않았습니다. UMAP 시각화를 사용할 수 없습니다.")

warnings.filterwarnings("ignore", category=UserWarning)

from lex_dpr.data import load_passages
from lex_dpr.models.encoders import BiEncoder
from lex_dpr.models.templates import TemplateMode, tp, tq
from lex_dpr.utils.io import read_jsonl


def visualize_embedding_space(
    encoder: BiEncoder,
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    output_dir: Path,
    *,
    method: str = "umap",  # "tsne" or "umap"
    n_samples: int = 1000,
    n_components: int = 2,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    임베딩 공간을 2D/3D로 시각화
    
    Args:
        encoder: BiEncoder 모델
        passages: Passage 딕셔너리
        eval_pairs_path: 평가 쌍 경로
        output_dir: 출력 디렉토리
        method: 차원 축소 방법 ("tsne" or "umap")
        n_samples: 시각화할 샘플 수
        n_components: 차원 수 (2 or 3)
        random_state: 랜덤 시드
        figsize: 그림 크기
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 평가 쌍 로드
    eval_pairs = list(read_jsonl(eval_pairs_path))
    if not eval_pairs:
        print("⚠️ 평가 쌍이 없습니다.")
        return
    
    # 샘플링
    if len(eval_pairs) > n_samples:
        indices = np.random.choice(len(eval_pairs), n_samples, replace=False)
        eval_pairs = [eval_pairs[i] for i in indices]
    
    # 쿼리와 관련 패시지 임베딩 추출
    query_texts = []
    query_labels = []
    passage_ids = []
    passage_texts = []
    passage_labels = []
    
    for pair in eval_pairs:
        query_text = pair["query_text"]
        query_texts.append(query_text)
        query_labels.append("Query")
        
        # Positive passages
        positive_ids = pair.get("positive_passages", [])
        for pid in positive_ids[:3]:  # 최대 3개만
            if pid in passages:
                passage_ids.append(pid)
                passage_texts.append(passages[pid]["text"])
                passage_labels.append("Positive")
    
    # 임베딩 생성
    print(f"[시각화] 임베딩 생성 중... (쿼리: {len(query_texts)}, 패시지: {len(passage_texts)})")
    query_embeddings = encoder.encode_queries(query_texts, batch_size=64)
    passage_embeddings = encoder.encode_passages(passage_texts, batch_size=64)
    
    # 통합 임베딩
    all_embeddings = np.vstack([query_embeddings, passage_embeddings])
    all_labels = query_labels + passage_labels
    
    # 차원 축소
    print(f"[시각화] {method.upper()}로 차원 축소 중...")
    if method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
        embeddings_2d = reducer.fit_transform(all_embeddings)
    elif method == "umap":
        if not UMAP_AVAILABLE:
            print("⚠️ UMAP을 사용할 수 없습니다. t-SNE로 대체합니다.")
            reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
            embeddings_2d = reducer.fit_transform(all_embeddings)
        else:
            reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=15)
            embeddings_2d = reducer.fit_transform(all_embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 시각화
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    if n_components == 2:
        # 2D 시각화
        for label in ["Query", "Positive"]:
            mask = np.array(all_labels) == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=label,
                alpha=0.6,
                s=50,
            )
        
        plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
        plt.title(f"임베딩 공간 시각화 ({method.upper()})", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        output_path = output_dir / f"embedding_space_{method}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ 시각화 저장: {output_path}")
        plt.close()
    
    elif n_components == 3:
        # 3D 시각화
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        
        for label in ["Query", "Positive"]:
            mask = np.array(all_labels) == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                embeddings_2d[mask, 2],
                label=label,
                alpha=0.6,
                s=50,
            )
        
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.set_zlabel(f"{method.upper()} Component 3", fontsize=12)
        ax.set_title(f"임베딩 공간 시각화 ({method.upper()}, 3D)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        
        output_path = output_dir / f"embedding_space_{method}_3d.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ 시각화 저장: {output_path}")
        plt.close()


def visualize_similarity_distribution(
    encoder: BiEncoder,
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    output_dir: Path,
    *,
    n_samples: int = 500,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Positive vs Negative 유사도 분포 시각화
    
    Args:
        encoder: BiEncoder 모델
        passages: Passage 딕셔너리
        eval_pairs_path: 평가 쌍 경로
        output_dir: 출력 디렉토리
        n_samples: 샘플 수
        figsize: 그림 크기
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 평가 쌍 로드
    eval_pairs = list(read_jsonl(eval_pairs_path))
    if not eval_pairs:
        print("⚠️ 평가 쌍이 없습니다.")
        return
    
    # 샘플링
    if len(eval_pairs) > n_samples:
        indices = np.random.choice(len(eval_pairs), n_samples, replace=False)
        eval_pairs = [eval_pairs[i] for i in indices]
    
    # 유사도 계산
    positive_similarities = []
    negative_similarities = []
    
    print(f"[시각화] 유사도 계산 중... ({len(eval_pairs)}개 쌍)")
    corpus_ids = list(passages.keys())
    corpus_texts = [passages[pid]["text"] for pid in corpus_ids]
    
    corpus_embeddings = encoder.encode_passages(corpus_texts, batch_size=64)
    corpus_tensor = np.array(corpus_embeddings)
    
    for pair in eval_pairs:
        query_text = pair["query_text"]
        positive_ids = set(pair.get("positive_passages", []))
        
        # 쿼리 임베딩
        query_emb = encoder.encode_queries([query_text], batch_size=1)[0]
        
        # Positive 유사도
        for pid in positive_ids:
            if pid in corpus_ids:
                idx = corpus_ids.index(pid)
                similarity = np.dot(query_emb, corpus_tensor[idx])
                positive_similarities.append(similarity)
        
        # Negative 유사도 (상위 100개 중 positive가 아닌 것들)
        similarities = np.dot(query_emb, corpus_tensor.T)
        top_indices = np.argsort(similarities)[::-1][:100]
        
        for idx in top_indices:
            pid = corpus_ids[idx]
            if pid not in positive_ids:
                negative_similarities.append(similarities[idx])
                if len(negative_similarities) >= len(positive_similarities):
                    break
    
    # 시각화
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    plt.hist(
        positive_similarities,
        bins=50,
        alpha=0.7,
        label="Positive",
        color="green",
        density=True,
    )
    plt.hist(
        negative_similarities,
        bins=50,
        alpha=0.7,
        label="Negative",
        color="red",
        density=True,
    )
    
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Positive vs Negative 유사도 분포", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 통계 정보 추가
    pos_mean = np.mean(positive_similarities)
    neg_mean = np.mean(negative_similarities)
    separation = pos_mean - neg_mean
    
    plt.axvline(pos_mean, color="green", linestyle="--", linewidth=2, label=f"Positive 평균: {pos_mean:.3f}")
    plt.axvline(neg_mean, color="red", linestyle="--", linewidth=2, label=f"Negative 평균: {neg_mean:.3f}")
    
    plt.text(
        0.05, 0.95,
        f"분리도 (Separation): {separation:.3f}\n"
        f"Positive 평균: {pos_mean:.3f}\n"
        f"Negative 평균: {neg_mean:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    plt.tight_layout()
    
    output_path = output_dir / "similarity_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 시각화 저장: {output_path}")
    print(f"   Positive 평균: {pos_mean:.4f}")
    print(f"   Negative 평균: {neg_mean:.4f}")
    print(f"   분리도: {separation:.4f}")
    plt.close()


def visualize_similarity_heatmap(
    encoder: BiEncoder,
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    output_dir: Path,
    *,
    n_queries: int = 20,
    n_passages: int = 50,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    쿼리-패시지 유사도 히트맵 시각화
    
    Args:
        encoder: BiEncoder 모델
        passages: Passage 딕셔너리
        eval_pairs_path: 평가 쌍 경로
        output_dir: 출력 디렉토리
        n_queries: 시각화할 쿼리 수
        n_passages: 시각화할 패시지 수
        figsize: 그림 크기
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 평가 쌍 로드
    eval_pairs = list(read_jsonl(eval_pairs_path))
    if not eval_pairs:
        print("⚠️ 평가 쌍이 없습니다.")
        return
    
    # 샘플링
    if len(eval_pairs) > n_queries:
        indices = np.random.choice(len(eval_pairs), n_queries, replace=False)
        eval_pairs = [eval_pairs[i] for i in indices]
    
    # 관련 패시지 수집
    all_positive_ids = set()
    for pair in eval_pairs:
        all_positive_ids.update(pair.get("positive_passages", []))
    
    # 패시지 샘플링
    positive_ids = list(all_positive_ids)[:n_passages]
    if len(positive_ids) < n_passages:
        # 추가 패시지 선택
        remaining = n_passages - len(positive_ids)
        other_ids = [pid for pid in passages.keys() if pid not in all_positive_ids]
        if len(other_ids) > remaining:
            other_ids = np.random.choice(other_ids, remaining, replace=False).tolist()
        positive_ids.extend(other_ids)
    
    # 임베딩 생성
    query_texts = [pair["query_text"] for pair in eval_pairs]
    passage_texts = [passages[pid]["text"] for pid in positive_ids]
    
    print(f"[시각화] 임베딩 생성 중... (쿼리: {len(query_texts)}, 패시지: {len(passage_texts)})")
    query_embeddings = encoder.encode_queries(query_texts, batch_size=64)
    passage_embeddings = encoder.encode_passages(passage_texts, batch_size=64)
    
    # 유사도 행렬 계산
    similarity_matrix = np.dot(query_embeddings, passage_embeddings.T)
    
    # 히트맵 시각화
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    # Positive 마스크 생성
    positive_mask = np.zeros_like(similarity_matrix, dtype=bool)
    for i, pair in enumerate(eval_pairs):
        positive_ids_set = set(pair.get("positive_passages", []))
        for j, pid in enumerate(positive_ids):
            if pid in positive_ids_set:
                positive_mask[i, j] = True
    
    # 히트맵
    sns.heatmap(
        similarity_matrix,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Cosine Similarity"},
        xticklabels=[pid[:20] + "..." if len(pid) > 20 else pid for pid in positive_ids],
        yticklabels=[q[:30] + "..." if len(q) > 30 else q for q in query_texts],
    )
    
    # Positive 셀 강조
    for i in range(len(eval_pairs)):
        for j in range(len(positive_ids)):
            if positive_mask[i, j]:
                plt.gca().add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="blue", lw=2)
                )
    
    plt.xlabel("Passages", fontsize=12)
    plt.ylabel("Queries", fontsize=12)
    plt.title("쿼리-패시지 유사도 히트맵 (파란색 테두리 = Positive)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / "similarity_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 시각화 저장: {output_path}")
    plt.close()


def compare_embeddings_before_after(
    encoder_before: Optional[BiEncoder],
    encoder_after: BiEncoder,
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    output_dir: Path,
    *,
    n_samples: int = 200,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """
    학습 전후 임베딩 품질 비교
    
    Args:
        encoder_before: 학습 전 모델 (None이면 스킵)
        encoder_after: 학습 후 모델
        passages: Passage 딕셔너리
        eval_pairs_path: 평가 쌍 경로
        output_dir: 출력 디렉토리
        n_samples: 샘플 수
        figsize: 그림 크기
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if encoder_before is None:
        print("⚠️ 학습 전 모델이 제공되지 않았습니다. 학습 후 모델만 평가합니다.")
        visualize_similarity_distribution(encoder_after, passages, eval_pairs_path, output_dir)
        return
    
    # 평가 쌍 로드
    eval_pairs = list(read_jsonl(eval_pairs_path))
    if not eval_pairs:
        print("⚠️ 평가 쌍이 없습니다.")
        return
    
    # 샘플링
    if len(eval_pairs) > n_samples:
        indices = np.random.choice(len(eval_pairs), n_samples, replace=False)
        eval_pairs = [eval_pairs[i] for i in indices]
    
    # 유사도 계산
    def compute_similarities(encoder, pairs):
        positive_sims = []
        negative_sims = []
        
        corpus_ids = list(passages.keys())
        corpus_texts = [passages[pid]["text"] for pid in corpus_ids]
        corpus_embeddings = encoder.encode_passages(corpus_texts, batch_size=64)
        corpus_tensor = np.array(corpus_embeddings)
        
        for pair in pairs:
            query_text = pair["query_text"]
            positive_ids = set(pair.get("positive_passages", []))
            
            query_emb = encoder.encode_queries([query_text], batch_size=1)[0]
            
            # Positive
            for pid in positive_ids:
                if pid in corpus_ids:
                    idx = corpus_ids.index(pid)
                    positive_sims.append(np.dot(query_emb, corpus_tensor[idx]))
            
            # Negative
            similarities = np.dot(query_emb, corpus_tensor.T)
            top_indices = np.argsort(similarities)[::-1][:100]
            
            for idx in top_indices:
                pid = corpus_ids[idx]
                if pid not in positive_ids:
                    negative_sims.append(similarities[idx])
                    if len(negative_sims) >= len(positive_sims):
                        break
        
        return positive_sims, negative_sims
    
    print("[시각화] 학습 전 모델 평가 중...")
    pos_before, neg_before = compute_similarities(encoder_before, eval_pairs)
    
    print("[시각화] 학습 후 모델 평가 중...")
    pos_after, neg_after = compute_similarities(encoder_after, eval_pairs)
    
    # 비교 시각화
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.set_style("whitegrid")
    
    # Before
    axes[0].hist(pos_before, bins=50, alpha=0.7, label="Positive", color="green", density=True)
    axes[0].hist(neg_before, bins=50, alpha=0.7, label="Negative", color="red", density=True)
    axes[0].axvline(np.mean(pos_before), color="green", linestyle="--", linewidth=2)
    axes[0].axvline(np.mean(neg_before), color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Cosine Similarity", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("학습 전", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # After
    axes[1].hist(pos_after, bins=50, alpha=0.7, label="Positive", color="green", density=True)
    axes[1].hist(neg_after, bins=50, alpha=0.7, label="Negative", color="red", density=True)
    axes[1].axvline(np.mean(pos_after), color="green", linestyle="--", linewidth=2)
    axes[1].axvline(np.mean(neg_after), color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Cosine Similarity", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("학습 후", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("학습 전후 유사도 분포 비교", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "before_after_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 시각화 저장: {output_path}")
    
    # 통계 출력
    sep_before = np.mean(pos_before) - np.mean(neg_before)
    sep_after = np.mean(pos_after) - np.mean(neg_after)
    improvement = sep_after - sep_before
    
    print(f"\n📊 학습 전후 비교:")
    print(f"   학습 전 분리도: {sep_before:.4f}")
    print(f"   학습 후 분리도: {sep_after:.4f}")
    print(f"   개선도: {improvement:.4f} ({improvement/abs(sep_before)*100:.1f}% 개선)")
    
    plt.close()

