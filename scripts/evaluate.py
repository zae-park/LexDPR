"""
학습이 완료된 LexDPR Bi-Encoder 체크포인트를 이용해
Retrieval 메트릭을 평가하는 스크립트.

내부적으로는 `lex_dpr.cli.eval_cli`의 main()을 그대로 호출한다.
"""

from lex_dpr.cli.eval_cli import main


if __name__ == "__main__":
    main()
