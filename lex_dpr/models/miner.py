# lex_dpr/models/miners.py
import numpy as np

def simple_hard_negative_miner(q_embs: np.ndarray, p_embs: np.ndarray, pos_idx: np.ndarray, topk: int = 5):
    """
    q_embs: [N, D], p_embs: [M, D], pos_idx: [N] (정답 passage의 인덱스)
    반환: 각 q마다 hard neg 인덱스 리스트
    """
    scores = q_embs @ p_embs.T  # normalized라면 IP=cos
    hard = []
    for i in range(len(q_embs)):
        order = np.argsort(-scores[i])  # desc
        # 정답 제외 topk
        hn = [j for j in order if j != pos_idx[i]][:topk]
        hard.append(hn)
    return hard
