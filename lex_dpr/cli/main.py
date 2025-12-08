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
from lex_dpr.cli import train, embed, api, config, eval_cli, sweep
from lex_dpr.crawler.crawl_precedents import PrecedentCrawler, REQUEST_DELAY
from lex_dpr.data_processing import make_pairs as make_pairs_mod
from lex_dpr.utils import gpu_utils

logger = logging.getLogger("lex_dpr.cli")

app = typer.Typer(
    name="lex-dpr",
    help="LexDPR: Legal Document Retriever & Reranker CLI",
    add_completion=False,
    no_args_is_help=True,
)


# Train 서브커맨드
train_app = typer.Typer(name="train", help="학습 관련 명령어")

@train_app.command("init")
def train_init_command(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="기존 파일이 있어도 덮어쓰기",
    ),
):
    """
    기본 학습 설정 파일 초기화
    
    configs/base.yaml, configs/data.yaml, configs/model.yaml 파일을 생성합니다.
    
    예시:
      poetry run lex-dpr train init
      poetry run lex-dpr train init --force
    """
    from lex_dpr.cli import config
    config.init_configs(force=force)

def _run_smoke_train():
    """smoke-train 실행 로직 (재사용)"""
    # 먼저 기본 config 파일이 없으면 생성
    from lex_dpr.cli import config as config_module
    user_configs_dir = Path.cwd() / "configs"
    base_path = user_configs_dir / "base.yaml"
    
    if not base_path.exists():
        logger.info("기본 설정 파일이 없습니다. 자동 생성합니다...")
        config_module.init_configs(force=False)
        logger.info("")
    
    original_argv = sys.argv.copy()
    try:
        # 사용자가 추가로 넘긴 오버라이드 인자 확보
        user_args = sys.argv[3:] if len(sys.argv) > 3 else []  # 'lex-dpr train smoke' 이후
        
        # SMOKE TEST 모드에서 강제할 인자
        # 1. 반복 횟수 제한 (epoch/step)
        # 2. 모든 기능 활성화 (gradient clipping, early stopping 등)
        forced_args = [
            "test_run=true",
            "trainer.epochs=1",
            # 모든 기능 활성화
            "trainer.gradient_clip_norm=1.0",
            "trainer.early_stopping.enabled=true",
            "trainer.early_stopping.patience=2",  # smoke-test에서는 patience를 낮게 설정
            "trainer.eval_steps=50",  # smoke-test에서는 더 자주 평가
        ]
        
        # 사용자가 같은 키를 덮어쓰지 못하도록 필터링
        filtered_user_args = [
            a
            for a in user_args
            if not (
                a.startswith("test_run=") or 
                a.startswith("trainer.epochs=") or
                a.startswith("trainer.gradient_clip_norm=") or
                a.startswith("trainer.early_stopping.enabled=") or
                a.startswith("trainer.early_stopping.patience=") or
                a.startswith("trainer.eval_steps=")
            )
        ]
        
        # 로그 출력을 위해 smoke-test용 config 정보 출력
        logger.info("=" * 80)
        logger.info("🧪 SMOKE TEST 모드: 모든 기능 활성화된 config 생성")
        logger.info("=" * 80)
        logger.info("📋 Smoke-test용 설정 (자동 생성):")
        logger.info("  ✅ 반복 횟수 제한:")
        logger.info("     - test_run: true (최대 100 iteration 또는 1 epoch)")
        logger.info("     - epochs: 1")
        logger.info("     - eval_steps: 50 (더 자주 평가)")
        logger.info("  ✅ 활성화된 기능:")
        logger.info("     - Learning Rate Scheduler: Warm-up + Cosine Annealing (전체 step의 10% warmup)")
        logger.info("     - Gradient Clipping: 활성화 (max_norm=1.0)")
        logger.info("     - Early Stopping: 활성화 (metric=cosine_ndcg@10, patience=2)")
        logger.info("")
        logger.info("💡 사용자 오버라이드:")
        if filtered_user_args:
            logger.info(f"     {', '.join(filtered_user_args)}")
        else:
            logger.info("     없음")
        logger.info("=" * 80)
        logger.info("")
        
        sys.argv = ["train"] + forced_args + filtered_user_args
        train.main()
    finally:
        sys.argv = original_argv

@train_app.command("smoke")
def train_smoke_command():
    """
    빠른 학습 SMOKE TEST 실행용 명령어.

    - 최소한의 config 파일을 자동 생성한 뒤 바로 실행
    - test_run=true 로 고정 (최대 100 iteration 또는 1 epoch)
    - trainer.epochs=1 로 고정
    - 모든 기능(learning rate scheduler, gradient clipping, early stopping 등) 활성화
    - epoch와 step 수만 제한하여 빠른 동작 테스트 수행

    예시:
      poetry run lex-dpr train smoke
      poetry run lex-dpr train smoke trainer.lr=3e-5
    """
    _run_smoke_train()

@train_app.callback(invoke_without_command=True)
def train_command(ctx: typer.Context):
    """
    모델 학습 실행
    
    config 파일이 없으면 smoke 모드와 동일하게 동작합니다.
    
    예시:
      poetry run lex-dpr train
      poetry run lex-dpr train trainer.epochs=5 trainer.lr=3e-5
    """
    # 서브커맨드가 지정된 경우 (init, smoke) 그대로 진행
    if ctx.invoked_subcommand is not None:
        return
    
    # config 파일이 없으면 smoke 모드와 동일하게 동작
    user_configs_dir = Path.cwd() / "configs"
    base_path = user_configs_dir / "base.yaml"
    
    if not base_path.exists():
        logger.info("설정 파일이 없습니다. smoke 모드로 실행합니다...")
        logger.info("")
        _run_smoke_train()
        return
    
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

app.add_typer(train_app)


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


# Sweep 서브커맨드
app.add_typer(sweep.app, name="sweep", help="WandB Sweep을 통한 하이퍼파라미터 튜닝")


@app.command("analyze-passages")
def analyze_passages_command(
    corpus: str = typer.Option(
        "data/processed/merged_corpus.jsonl",
        "--corpus",
        "-c",
        help="분석할 passage corpus JSONL 파일 경로 (기본값: data/processed/merged_corpus.jsonl)",
    ),
    tokenizer: Optional[str] = typer.Option(
        None,
        "--tokenizer",
        "-t",
        help="토큰 길이 계산용 토크나이저 (예: BAAI/bge-m3). 지정하지 않으면 단어 수로 추정",
    ),
    min_text_length: int = typer.Option(
        10,
        "--min-text-length",
        help="짧은 텍스트로 간주할 최소 길이 (기본값: 10)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="텍스트 리포트 저장 경로 (선택사항)",
    ),
    json_output: Optional[str] = typer.Option(
        None,
        "--json-output",
        help="JSON 리포트 저장 경로 (선택사항)",
    ),
):
    """
    Passage Corpus 품질 분석 스크립트
    
    Passage corpus의 통계 및 품질을 분석합니다:
    - 총 passage 개수 및 소스별 분포
    - 중복 passage 탐지 및 통계
    - 길이 분포 분석 (문자 수, 토큰 수)
    - 소스별(법령/행정규칙/판례) 통계
    
    예시:
      poetry run lex-dpr analyze-passages
      poetry run lex-dpr analyze-passages --corpus data/merged_corpus.jsonl
      poetry run lex-dpr analyze-passages --corpus data/merged_corpus.jsonl --tokenizer BAAI/bge-m3 --output report.txt
    """
    from scripts.analyze_passages import analyze_passages, print_analysis_report
    import json as json_module
    
    # 분석 실행
    logger.info(f"Passage corpus 분석 중: {corpus}")
    results = analyze_passages(
        corpus_path=corpus,
        tokenizer_name=tokenizer,
        min_text_length=min_text_length,
    )
    
    # 리포트 출력
    print_analysis_report(results, output_file=output)
    
    # JSON 출력
    if json_output:
        json_path = Path(json_output)
        json_path.write_text(
            json_module.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"✅ JSON 리포트 저장: {json_path}")


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


@app.command("visualize")
def visualize_command(
    model: str = typer.Option(..., "--model", "-m", help="모델 경로 (체크포인트 또는 HuggingFace 모델)"),
    passages: str = typer.Option("data/processed/merged_corpus.jsonl", "--passages", "-p", help="Passage corpus JSONL 경로"),
    eval_pairs: str = typer.Option("data/pairs_eval.jsonl", "--eval-pairs", "-e", help="평가 쌍 JSONL 경로"),
    output_dir: str = typer.Option("visualizations", "--output", "-o", help="시각화 결과 저장 디렉토리"),
    method: str = typer.Option("umap", "--method", help="차원 축소 방법 (tsne 또는 umap)"),
    visualization_type: str = typer.Option("all", "--type", "-t", help="시각화 타입 (all, space, similarity, heatmap, comparison)"),
    model_before: Optional[str] = typer.Option(None, "--model-before", help="학습 전 모델 경로 (비교용)"),
    n_samples: int = typer.Option(1000, "--n-samples", help="시각화할 샘플 수"),
    peft_adapter: Optional[str] = typer.Option(None, "--peft-adapter", help="PEFT 어댑터 경로"),
):
    """
    임베딩 품질 시각화
    
    다양한 방법으로 임베딩 품질을 시각화합니다:
    - embedding-space: 임베딩 공간 시각화 (t-SNE/UMAP)
    - similarity: Positive vs Negative 유사도 분포
    - heatmap: 쿼리-패시지 유사도 히트맵
    - comparison: 학습 전후 비교
    
    예시:
      poetry run lex-dpr visualize --model checkpoint/lexdpr/bi_encoder
      poetry run lex-dpr visualize --model checkpoint/lexdpr/bi_encoder --type similarity
      poetry run lex-dpr visualize --model checkpoint/lexdpr/bi_encoder --model-before ko-simcse --type comparison
    """
    from lex_dpr.data import load_passages
    from lex_dpr.models.encoders import BiEncoder
    from lex_dpr.models.templates import TemplateMode
    from lex_dpr.visualization import (
        compare_embeddings_before_after,
        visualize_embedding_space,
        visualize_similarity_distribution,
        visualize_similarity_heatmap,
    )
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    logger.info(f"[시각화] 모델 로딩 중: {model}")
    encoder = BiEncoder(
        model,
        template=TemplateMode.BGE,
        normalize=True,
        peft_adapter_path=peft_adapter,
    )
    
    encoder_before = None
    if model_before:
        logger.info(f"[시각화] 학습 전 모델 로딩 중: {model_before}")
        encoder_before = BiEncoder(
            model_before,
            template=TemplateMode.BGE,
            normalize=True,
        )
    
    # Passage 로드
    logger.info(f"[시각화] Passage 로딩 중: {passages}")
    passages_dict = load_passages(passages)
    logger.info(f"[시각화] {len(passages_dict)}개 Passage 로드 완료")
    
    # 시각화 실행
    if visualization_type in ["all", "space"]:
        logger.info("[시각화] 임베딩 공간 시각화 중...")
        visualize_embedding_space(
            encoder=encoder,
            passages=passages_dict,
            eval_pairs_path=eval_pairs,
            output_dir=output_dir_path,
            method=method,
            n_samples=n_samples,
        )
    
    if visualization_type in ["all", "similarity"]:
        logger.info("[시각화] 유사도 분포 시각화 중...")
        visualize_similarity_distribution(
            encoder=encoder,
            passages=passages_dict,
            eval_pairs_path=eval_pairs,
            output_dir=output_dir_path,
            n_samples=n_samples,
        )
    
    if visualization_type in ["all", "heatmap"]:
        logger.info("[시각화] 히트맵 시각화 중...")
        visualize_similarity_heatmap(
            encoder=encoder,
            passages=passages_dict,
            eval_pairs_path=eval_pairs,
            output_dir=output_dir_path,
        )
    
    if visualization_type in ["all", "comparison"]:
        if encoder_before:
            logger.info("[시각화] 학습 전후 비교 중...")
            compare_embeddings_before_after(
                encoder_before=encoder_before,
                encoder_after=encoder,
                passages=passages_dict,
                eval_pairs_path=eval_pairs,
                output_dir=output_dir_path,
                n_samples=n_samples,
            )
        else:
            logger.warning("⚠️ 학습 전 모델이 제공되지 않아 비교를 건너뜁니다. --model-before를 지정하세요.")
    
    logger.info(f"✅ 시각화 완료! 결과는 {output_dir_path}에 저장되었습니다.")


@app.command("gpu")
def gpu_command(
    action: str = typer.Argument(..., help="동작: list, kill, kill-all"),
    pid: Optional[int] = typer.Argument(None, help="종료할 프로세스 ID (kill 명령어 사용 시)"),
    force: bool = typer.Option(False, "--force", "-f", help="강제 종료"),
    sudo: bool = typer.Option(False, "--sudo", help="sudo 권한 사용 (다른 사용자의 프로세스 종료 시 필요)"),
):
    """
    GPU 프로세스 관리
    
    사용 예시:
      poetry run lex-dpr gpu list                    # GPU 프로세스 목록 확인
      poetry run lex-dpr gpu kill <PID>              # 특정 프로세스 종료
      poetry run lex-dpr gpu kill <PID> --sudo      # sudo 권한으로 종료
      poetry run lex-dpr gpu kill-all                # 모든 GPU 프로세스 종료
      poetry run lex-dpr gpu kill <PID> --force      # 강제 종료
    """
    if action == "list":
        gpu_utils.list_processes()
    elif action == "kill":
        if pid is None:
            logger.error("❌ kill 명령어는 PID가 필요합니다.")
            logger.error("사용법: poetry run lex-dpr gpu kill <PID>")
            raise typer.Exit(1)
        gpu_utils.kill_process_by_pid(pid, force=force, use_sudo=sudo)
    elif action == "kill-all":
        gpu_utils.kill_all_processes(force=force, use_sudo=sudo)
    else:
        logger.error(f"❌ 알 수 없는 동작: {action}")
        logger.error("사용 가능한 동작: list, kill, kill-all")
        raise typer.Exit(1)


def main():
    """메인 진입점"""
    app()


if __name__ == "__main__":
    main()

