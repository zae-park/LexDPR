"""
LexDPR 메인 CLI 래퍼

사용 예시:
  poetry run lex-dpr train
  poetry run lex-dpr config init
  poetry run lex-dpr config show
  poetry run lex-dpr embed --model ...
  poetry run lex-dpr api --model ...
"""

import logging
import sys
import warnings
from typing import Optional

import typer

# FutureWarning 억제
warnings.filterwarnings("ignore", category=FutureWarning)

# 서브커맨드 모듈 import
from lex_dpr.cli import train, embed, api, config
from lex_dpr.crawler.crawl_precedents import PrecedentCrawler, REQUEST_DELAY

logger = logging.getLogger("lex_dpr.cli")

app = typer.Typer(
    name="lex-dpr",
    help="LexDPR: Legal Document Retriever & Reranker CLI",
    add_completion=False,
    no_args_is_help=True,
)


# Train 서브커맨드
@app.command("train")
def train_command():
    """
    모델 학습 실행
    
    예시:
      poetry run lex-dpr train
      poetry run lex-dpr train trainer.epochs=5 trainer.lr=3e-5
    """
    # train.py의 main 함수를 호출하되, sys.argv를 조작
    original_argv = sys.argv.copy()
    try:
        # 'lex-dpr train' 부분을 제거하고 나머지만 전달
        # sys.argv에서 'lex-dpr train' 이후의 모든 인자 가져오기
        remaining_args = sys.argv[2:] if len(sys.argv) > 2 else []
        sys.argv = ["train"] + remaining_args
        train.main()
    finally:
        sys.argv = original_argv


@app.command("smoke-train")
def smoke_train_command():
    """
    빠른 학습 SMOKE TEST 실행용 명령어.

    - test_run=true 로 고정 (최대 100 iteration 또는 1 epoch)
    - trainer.epochs=1 로 고정

    예시:
      poetry run lex-dpr smoke-train
      poetry run lex-dpr smoke-train trainer.lr=3e-5
    """
    original_argv = sys.argv.copy()
    try:
        # 사용자가 추가로 넘긴 오버라이드 인자 확보
        user_args = sys.argv[2:] if len(sys.argv) > 2 else []
        # SMOKE TEST 모드에서 강제할 인자
        forced_args = ["test_run=true", "trainer.epochs=1"]
        # 사용자가 같은 키를 덮어쓰지 못하도록 필터링
        filtered_user_args = [
            a
            for a in user_args
            if not (a.startswith("test_run=") or a.startswith("trainer.epochs="))
        ]
        sys.argv = ["train"] + forced_args + filtered_user_args
        train.main()
    finally:
        sys.argv = original_argv


@app.command("crawl-precedents")
def crawl_precedents_command(
    output: str = typer.Option(
        "data/precedents",
        "--output",
        "-o",
        help="판례 JSON 파일을 저장할 디렉토리 (기본값: data/precedents)",
    ),
    max_pages: int = typer.Option(
        0,
        "--max-pages",
        help="크롤링할 최대 페이지 수 (0이면 crawler 기본값 사용)",
    ),
    start_page: int = typer.Option(
        1,
        "--start-page",
        help="시작 페이지 번호 (기본값: 1)",
    ),
    delay: float = typer.Option(
        REQUEST_DELAY,
        "--delay",
        help=f"요청 간 지연 시간(초) (기본값: {REQUEST_DELAY})",
    ),
    max_workers: int = typer.Option(
        4,
        "--max-workers",
        help="병렬 처리 워커 수 (기본값: 4)",
    ),
):
    """
    law.go.kr에서 판례 데이터를 크롤링합니다.

    - PAGE 번호를 기준으로 페이지 범위를 지정할 수 있습니다.
    - `--start-page`, `--max-pages` 옵션으로 범위를 제어합니다.

    예시:
      poetry run lex-dpr crawl-precedents --max-pages 10
      poetry run lex-dpr crawl-precedents --start-page 5 --max-pages 20
    """
    crawler = PrecedentCrawler(output, delay=delay, max_workers=max_workers)
    crawler.crawl(max_pages=max_pages or None, start_page=start_page)


# Config 서브커맨드
config_app = typer.Typer(name="config", help="설정 관리")
app.add_typer(config_app)


@config_app.command("init")
def config_init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="기존 파일이 있어도 덮어쓰기",
    ),
):
    """
    기본 설정 파일을 configs/ 디렉토리에 초기화
    
    예시:
      poetry run lex-dpr config init
      poetry run lex-dpr config init --force
    """
    config.init_configs(force=force)


@config_app.command("show")
def config_show():
    """
    현재 설정된 config 출력
    
    예시:
      poetry run lex-dpr config show
    """
    config.show_config()


# Embed 서브커맨드
@app.command("embed")
def embed_command(
    model: str = typer.Option(..., "--model", "-m", help="학습된 모델 체크포인트 경로"),
    input: str = typer.Option(..., "--input", "-i", help="입력 JSONL 파일 (passages or queries)"),
    outdir: str = typer.Option(..., "--outdir", "-o", help="임베딩 출력 디렉토리"),
    prefix: str = typer.Option(..., "--prefix", "-p", help="출력 파일 접두사 (예: 'passages', 'queries')"),
    type: str = typer.Option(..., "--type", "-t", help="임베딩 타입: 'passage' or 'query'"),
    id_field: str = typer.Option("id", "--id-field", help="입력 JSONL의 ID 필드명"),
    text_field: str = typer.Option("text", "--text-field", help="입력 JSONL의 텍스트 필드명"),
    template: str = typer.Option("bge", "--template", help="템플릿 모드: 'bge' or 'none'"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="인코딩 배치 크기"),
    max_len: int = typer.Option(0, "--max-len", help="최대 시퀀스 길이 (0 = 모델 기본값)"),
    device: Optional[str] = typer.Option(None, "--device", help="디바이스 (cuda/cpu, 기본: 자동)"),
    output_format: str = typer.Option("npz", "--output-format", help="출력 형식: 'npz', 'npy', 'both'"),
    limit: Optional[int] = typer.Option(None, "--limit", help="인코딩할 행 수 제한 (테스트용)"),
    no_normalize: bool = typer.Option(False, "--no-normalize", help="임베딩 정규화 비활성화"),
    peft_adapter: Optional[str] = typer.Option(None, "--peft-adapter", help="PEFT 어댑터 경로"),
):
    """
    학습된 Bi-Encoder 모델로부터 임베딩 추출
    
    예시:
      poetry run lex-dpr embed \\
        --model checkpoint/lexdpr/bi_encoder \\
        --input data/processed/law_passages.jsonl \\
        --outdir embeds \\
        --prefix passages \\
        --type passage
    """
    # embed.py의 main 함수를 호출하되, sys.argv를 조작
    original_argv = sys.argv.copy()
    try:
        args = []
        args.extend(["--model", model])
        args.extend(["--input", input])
        args.extend(["--outdir", outdir])
        args.extend(["--prefix", prefix])
        args.extend(["--type", type])
        args.extend(["--id-field", id_field])
        args.extend(["--text-field", text_field])
        args.extend(["--template", template])
        args.extend(["--batch-size", str(batch_size)])
        args.extend(["--max-len", str(max_len)])
        if device:
            args.extend(["--device", device])
        args.extend(["--output-format", output_format])
        if limit:
            args.extend(["--limit", str(limit)])
        if no_normalize:
            args.append("--no-normalize")
        if peft_adapter:
            args.extend(["--peft-adapter", peft_adapter])
        
        sys.argv = ["embed"] + args
        embed.main()
    finally:
        sys.argv = original_argv


# API 서브커맨드
@app.command("api")
def api_command(
    model: str = typer.Option(..., "--model", "-m", help="학습된 모델 체크포인트 경로"),
    template: str = typer.Option("bge", "--template", help="템플릿 모드: 'bge' or 'none'"),
    max_len: int = typer.Option(0, "--max-len", help="최대 시퀀스 길이 (0 = 모델 기본값)"),
    device: Optional[str] = typer.Option(None, "--device", help="디바이스 (cuda/cpu, 기본: 자동)"),
    peft_adapter: Optional[str] = typer.Option(None, "--peft-adapter", help="PEFT 어댑터 경로"),
    host: str = typer.Option("0.0.0.0", "--host", help="바인딩할 호스트"),
    port: int = typer.Option(8000, "--port", "-p", help="바인딩할 포트"),
):
    """
    임베딩 API 서버 실행
    
    예시:
      poetry run lex-dpr api \\
        --model checkpoint/lexdpr/bi_encoder \\
        --host 0.0.0.0 \\
        --port 8000
    """
    # api.py의 main 함수를 호출하되, sys.argv를 조작
    original_argv = sys.argv.copy()
    try:
        args = []
        args.extend(["--model", model])
        args.extend(["--template", template])
        args.extend(["--max-len", str(max_len)])
        if device:
            args.extend(["--device", device])
        if peft_adapter:
            args.extend(["--peft-adapter", peft_adapter])
        args.extend(["--host", host])
        args.extend(["--port", str(port)])
        
        sys.argv = ["api"] + args
        api.main()
    finally:
        sys.argv = original_argv


def main():
    """메인 진입점"""
    app()


if __name__ == "__main__":
    main()

