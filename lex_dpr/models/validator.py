# lex_dpr/models/validators.py
def validate_pairs_and_passages(passages: dict, pairs_iter):
    missing = []
    for row in pairs_iter:
        for pid in row.get("positive_passages", []):
            if pid not in passages: missing.append(("pos", pid))
        for nid in row.get("hard_negatives", []):
            if nid not in passages: missing.append(("neg", nid))
    if missing:
        ex = ", ".join(f"{t}:{i}" for t,i in missing[:10])
        raise ValueError(f"Pairs refer to missing passages ({len(missing)}). e.g., {ex}")
