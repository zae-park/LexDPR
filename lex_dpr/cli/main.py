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

