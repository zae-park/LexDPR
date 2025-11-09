# lex_dpr/models/augment.py

def duplicate_with_prefix(queries: list[str], prefix: str) -> list[str]:
    return [f"{prefix} {q}" for q in queries]
# 예: "질문:" 프리픽스 버전 추가, or "사용자:" 등 간단 변형
