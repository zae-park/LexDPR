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
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import typer

# FutureWarning 억제
warnings.filterwarnings("ignore", category=FutureWarning)

# 서브커맨드 모듈 import
from lex_dpr.cli import train, embed, api, config, eval_cli
from lex_dpr.crawler.crawl_precedents import PrecedentCrawler, REQUEST_DELAY
from lex_dpr.data_processing import make_pairs as make_pairs_mod

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


@app.command("eval")
def eval_command():
    """
    학습된 Bi-Encoder 체크포인트를 이용해 Retrieval 메트릭을 평가합니다.

    scripts/evaluate.py 와 동일한 인자를 사용할 수 있습니다.

    예시:
      poetry run lex-dpr eval
      poetry run lex-dpr eval --model checkpoint/lexdpr/bi_encoder --eval-pairs data/pairs_eval.jsonl
      poetry run lex-dpr eval --k-values 1 3 5 10 --output eval_results.json
    """
    original_argv = sys.argv.copy()
    try:
        # 'lex-dpr eval' 이후의 인자를 그대로 전달
        remaining_args = sys.argv[2:] if len(sys.argv) > 2 else []
        sys.argv = ["evaluate"] + remaining_args
        eval_cli.main()
    finally:
        sys.argv = original_argv


@app.command("gen-data")
def gen_data_command(
    law: str = typer.Option(
        "data/processed/law_passages.jsonl",
        "--law",
        help="법령 passage JSONL 경로 (기본값: data/processed/law_passages.jsonl)",
    ),
    admin: str = typer.Option(
        "data/processed/admin_passages.jsonl",
        "--admin",
        help="행정규칙 passage JSONL 경로 (기본값: data/processed/admin_passages.jsonl)",
    ),
    prec: str = typer.Option(
        "data/processed/prec_passages.jsonl",
        "--prec",
        help="판례 passage JSONL 경로 (기본값: data/processed/prec_passages.jsonl)",
    ),
    prec_json_dir: str = typer.Option(
        "data/precedents",
        "--prec-json-dir",
        help="판례 원본 JSON 디렉토리 (기본값: data/precedents)",
    ),
    out: str = typer.Option(
        "data/pairs_train.jsonl",
        "--out",
        help="생성할 train pairs 경로 (기본값: data/pairs_train.jsonl)",
    ),
    hn_per_q: int = typer.Option(
        10,
        "--hn-per-q",
        help="질의당 hard negative 개수 (기본값: 10)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="랜덤 시드 (기본값: 42)",
    ),
    max_positives_per_prec: int = typer.Option(
        5,
        "--max-positives-per-prec",
        help="판례당 최대 positive passage 개수 (기본값: 5)",
    ),
    use_admin_for_prec: bool = typer.Option(
        False,
        "--use-admin-for-prec",
        help="판례→법령/행정규칙 쌍 생성 시 행정규칙도 포함할지 여부 (기본값: False)",
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        help="병렬 처리 워커 수 (기본값: CPU 코어 수)",
    ),
):
    """
    전처리된 passage들로부터 train/valid/test 질의-passage 쌍을 생성합니다.

    - 마지막 자리수가 8인 query_id → valid
    - 마지막 자리수가 9인 query_id → test
    - 나머지 → train

    결과:
      - data/pairs_train.jsonl
      - data/pairs_train_valid.jsonl
      - data/pairs_train_test.jsonl
      - data/pairs_eval.jsonl (valid 세트 복사본)
    """
    # make_pairs 모듈을 통해 실제 쌍 생성 및 split 수행
    make_pairs_mod.make_pairs(
        law_path=law,
        admin_path=admin,
        prec_path=prec,
        prec_json_dir=prec_json_dir,
        out_path=out,
        hn_per_q=hn_per_q,
        seed=seed,
        enable_cross_positive=True,
        max_positives_per_prec=max_positives_per_prec,
        prec_json_glob="**/*.json",
        use_admin_for_prec=use_admin_for_prec,
        max_workers=max_workers,
    )

    out_path_obj = Path(out)
    parent = out_path_obj.parent
    stem = out_path_obj.stem
    suffix = out_path_obj.suffix or ".jsonl"

    valid_path = parent / f"{stem}_valid{suffix}"
    eval_path = Path("data/pairs_eval.jsonl")

    if valid_path.exists():
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(valid_path, eval_path)
        logger.info(f"평가용 pairs_eval.jsonl 생성: {eval_path} (from {valid_path})")
    else:
        logger.warning(f"valid 파일을 찾을 수 없어 pairs_eval.jsonl을 생성하지 못했습니다: {valid_path}")


@app.command("analyze-pairs")
def analyze_pairs_command(
    pairs_dir: Optional[str] = typer.Option(
        None,
        "--pairs-dir",
        help="pairs 파일들이 있는 디렉토리 (자동으로 train/valid/test 파일 찾기)",
    ),
    train: Optional[str] = typer.Option(
        None,
        "--train",
        help="Train 데이터셋 경로 (pairs_train.jsonl)",
    ),
    valid: Optional[str] = typer.Option(
        None,
        "--valid",
        help="Valid 데이터셋 경로 (pairs_train_valid.jsonl)",
    ),
    test: Optional[str] = typer.Option(
        None,
        "--test",
        help="Test 데이터셋 경로 (pairs_train_test.jsonl)",
    ),
    passages: Optional[str] = typer.Option(
        "data/processed/merged_corpus.jsonl",
        "--passages",
        help="Passage 코퍼스 경로 (토큰 길이 계산용)",
    ),
    tokenizer: str = typer.Option(
        "BAAI/bge-m3",
        "--tokenizer",
        help="토크나이저 모델 이름 (기본값: BAAI/bge-m3). 'none'이면 단어 수로 계산",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="분석 리포트 출력 파일 경로 (텍스트 + JSON)",
    ),
):
    """
    데이터 품질 분석: train/valid/test 데이터셋의 통계 및 분포 분석
    
    분석 항목:
    - 데이터셋 크기 (train/valid/test)
    - Positive/Negative 비율 및 분포
    - 쿼리 타입별 분포 (law, admin, prec)
    - 질의(query) 토큰 길이 분포
    - Passage 토큰 길이 분포 (positive passages)
    
    예시:
      poetry run lex-dpr analyze-pairs --pairs-dir data
      poetry run lex-dpr analyze-pairs --train data/pairs_train.jsonl --valid data/pairs_train_valid.jsonl
    """
    from pathlib import Path
    from scripts.analyze_pairs import analyze_dataset, print_analysis_report
    
    # 파일 경로 결정
    train_path = None
    valid_path = None
    test_path = None
    
    if pairs_dir:
        pairs_dir_obj = Path(pairs_dir)
        train_path = pairs_dir_obj / "pairs_train.jsonl"
        valid_path = pairs_dir_obj / "pairs_train_valid.jsonl"
        test_path = pairs_dir_obj / "pairs_train_test.jsonl"
        
        # 파일 존재 확인
        if not train_path.exists():
            train_path = None
        if not valid_path.exists():
            valid_path = None
        if not test_path.exists():
            test_path = None
    else:
        train_path = train
        valid_path = valid
        test_path = test
    
    if not any([train_path, valid_path, test_path]):
        logger.error("분석할 데이터셋 파일을 찾을 수 없습니다. --pairs-dir 또는 --train/--valid/--test를 지정하세요.")
        raise typer.Exit(1)
    
    # 토크나이저 설정
    tokenizer_name = tokenizer if tokenizer.lower() != "none" else None
    
    # 분석 실행
    results = {}
    
    if train_path and Path(train_path).exists():
        logger.info(f"[분석 중] Train 데이터셋: {train_path}")
        results["train"] = analyze_dataset(
            str(train_path),
            passages_path=passages,
            tokenizer_name=tokenizer_name,
            dataset_name="train",
        )
    
    if valid_path and Path(valid_path).exists():
        logger.info(f"[분석 중] Valid 데이터셋: {valid_path}")
        results["valid"] = analyze_dataset(
            str(valid_path),
            passages_path=passages,
            tokenizer_name=tokenizer_name,
            dataset_name="valid",
        )
    
    if test_path and Path(test_path).exists():
        logger.info(f"[분석 중] Test 데이터셋: {test_path}")
        results["test"] = analyze_dataset(
            str(test_path),
            passages_path=passages,
            tokenizer_name=tokenizer_name,
            dataset_name="test",
        )
    
    if not results:
        logger.error("분석할 데이터가 없습니다.")
        raise typer.Exit(1)
    
    # 리포트 출력
    print_analysis_report(results, output_file=output)


def main():
    """메인 진입점"""
    app()


if __name__ == "__main__":
    main()

