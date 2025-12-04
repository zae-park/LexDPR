"""
WandB Sweep CLI 명령어

사용 예시:
  poetry run lex-dpr sweep init --output configs/my_sweep.yaml
  poetry run lex-dpr sweep start --config configs/my_sweep.yaml
  poetry run lex-dpr sweep agent <sweep-id>
  poetry run lex-dpr sweep run --config configs/my_sweep.yaml
"""

import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

# train.py의 함수들을 import하여 재사용

# FutureWarning 억제
warnings.filterwarnings("ignore", category=FutureWarning)

import typer
from omegaconf import OmegaConf, DictConfig

from lex_dpr.cli.train import _get_config_path
from lex_dpr.trainer.sweep_trainer import SweepTrainer

logger = logging.getLogger("lex_dpr.cli.sweep")


def _convert_to_dict(obj: Any) -> Any:
    """OmegaConf 객체를 재귀적으로 일반 Python 딕셔너리/리스트/값으로 변환"""
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        return {k: _convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_dict(item) for item in obj]
    else:
        return obj


def _check_time_window(
    time_window: Optional[Tuple[int, int]],
    timezone: str = "Asia/Seoul",
) -> Tuple[bool, Optional[datetime]]:
    """
    현재 시간이 허용된 시간 범위 내에 있는지 확인
    
    Args:
        time_window: (시작 시간, 종료 시간) 튜플 (예: (1, 8) = 1시~8시) 또는 None
        timezone: 타임존 (기본값: Asia/Seoul)
    
    Returns:
        (is_allowed, next_start_time): 허용 여부와 다음 시작 시간 (None이면 시간 제한 없음)
    """
    if time_window is None:
        return True, None
    
    try:
        import pytz
        start_hour, end_hour = time_window
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        current_hour = now.hour
        
        # 시간 범위 체크
        if start_hour <= end_hour:
            # 일반적인 경우 (예: 1시~8시)
            is_allowed = start_hour <= current_hour < end_hour
        else:
            # 자정을 넘어가는 경우 (예: 22시~6시)
            is_allowed = current_hour >= start_hour or current_hour < end_hour
        
        if is_allowed:
            return True, None
        
        # 다음 시작 시간 계산
        if start_hour <= end_hour:
            if current_hour < start_hour:
                # 오늘 시작 시간까지 대기
                next_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            else:
                # 내일 시작 시간까지 대기
                next_start = (now + timedelta(days=1)).replace(hour=start_hour, minute=0, second=0, microsecond=0)
        else:
            # 자정을 넘어가는 경우
            if current_hour < end_hour:
                # 지금은 종료 시간 이전이므로 시작 시간까지 대기 (오늘)
                next_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            elif current_hour >= start_hour:
                # 지금은 시작 시간 이후이므로 내일 시작 시간까지 대기
                next_start = (now + timedelta(days=1)).replace(hour=start_hour, minute=0, second=0, microsecond=0)
            else:
                # 종료 시간과 시작 시간 사이
                next_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        return False, next_start
    
    except ImportError:
        logger.warning("pytz가 설치되지 않았습니다. 시간 기반 제한 기능을 사용할 수 없습니다.")
        logger.info("설치: poetry add pytz")
        return True, None
    except Exception as e:
        logger.warning(f"시간 체크 실패: {e}. 시간 제한 없이 실행합니다.")
        return True, None

app = typer.Typer(
    name="sweep",
    help="WandB Sweep을 통한 하이퍼파라미터 튜닝",
    add_completion=False,
    no_args_is_help=False,  # 인자가 없을 때 자동 실행 허용
)


def _run_sweep_impl(
    config_path: Path,
    smoke_test: bool,
    run_agent: bool,
):
    """sweep 실행 로직 (재사용)"""
    try:
        import wandb
    except ImportError:
        logger.error("wandb가 설치되지 않았습니다. 'poetry install --extras wandb'로 설치하세요.")
        raise typer.Exit(1)
    
    if not config_path.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise typer.Exit(1)
    
    sweep_config = OmegaConf.load(config_path)
    
    if smoke_test:
        logger.info("🧪 SMOKE TEST 모드로 실행합니다.")
        if "fixed" not in sweep_config:
            sweep_config["fixed"] = {}
        sweep_config["fixed"]["test_run"] = True
        sweep_config["fixed"]["trainer.epochs"] = 1
        if "trainer.eval_steps" not in sweep_config.get("fixed", {}):
            sweep_config["fixed"]["trainer.eval_steps"] = 50
    
    wandb_project = sweep_config.get("project", "lexdpr")
    wandb_entity = sweep_config.get("entity", None)
    if smoke_test:
        wandb_project = f"{wandb_project}-smoke-test"
    
    # OmegaConf 객체를 일반 Python 딕셔너리로 변환
    method = _convert_to_dict(sweep_config.get("method", "random"))
    metric = _convert_to_dict(sweep_config.get("metric", {"name": "eval/ndcg@10", "goal": "maximize"}))
    parameters = _convert_to_dict(sweep_config.get("parameters", {}))
    
    sweep_dict = {
        "method": method,
        "metric": metric,
        "parameters": parameters or {},
    }
    
    # Early termination 설정 추가
    early_terminate = sweep_config.get("early_terminate")
    if early_terminate:
        early_terminate_dict = _convert_to_dict(early_terminate)
        sweep_dict["early_terminate"] = early_terminate_dict
        logger.info(f"Early termination 설정: {early_terminate_dict}")
    
    fixed_params = _convert_to_dict(sweep_config.get("fixed", {}))
    if fixed_params:
        logger.info(f"고정 파라미터 적용: {list(fixed_params.keys())}")
        for key, value in fixed_params.items():
            if key not in sweep_dict["parameters"]:
                sweep_dict["parameters"][key] = {"value": _convert_to_dict(value)}
    
    logger.info(f"WandB 프로젝트: {wandb_project}")
    if wandb_entity:
        logger.info(f"WandB 엔티티: {wandb_entity}")
    logger.info("스윕 생성 중... (WandB API 호출 중, 잠시만 기다려주세요)")
    logger.info("")
    logger.info(f"스윕 설정 요약:")
    logger.info(f"  - 방법: {method}")
    logger.info(f"  - 파라미터 수: {len(parameters)}")
    logger.info(f"  - 고정 파라미터 수: {len(fixed_params)}")
    logger.info("")
    
    try:
        import sys
        logger.info("WandB API에 연결 중...")
        sys.stdout.flush()  # 강제 출력
        
        sweep_id = wandb.sweep(sweep_dict, project=wandb_project, entity=wandb_entity)
        
        logger.info("")
        logger.info(f"✅ WandB API 호출 완료")
        logger.info(f"   스윕 ID: {sweep_id}")
        sys.stdout.flush()
    except Exception as e:
        logger.error("")
        logger.error(f"❌ WandB 스윕 생성 실패!")
        logger.error(f"   에러 타입: {type(e).__name__}")
        logger.error(f"   에러 메시지: {str(e)}")
        logger.error("")
        logger.error("가능한 원인:")
        logger.error("  1. WandB 설정 파일 스키마 오류 (위 경고 확인)")
        logger.error("  2. 네트워크 연결 문제")
        logger.error("  3. WandB 인증 문제 (wandb login 확인)")
        logger.error("")
        raise
    
    # 스윕 ID를 설정 파일에 저장
    sweep_config["sweep_id"] = sweep_id
    OmegaConf.save(sweep_config, config_path)
    logger.info(f"스윕 ID가 설정 파일에 저장되었습니다: {config_path}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 스윕 생성 완료!")
    logger.info(f"스윕 ID: {sweep_id}")
    logger.info("")
    logger.info("다음 단계:")
    logger.info(f"  poetry run lex-dpr sweep agent --config {config_path}")
    logger.info(f"  또는")
    logger.info(f"  poetry run lex-dpr sweep agent {sweep_id}")
    logger.info("")
    logger.info(f"WandB 대시보드: https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}/sweeps/{sweep_id}")
    logger.info("=" * 80)
    
    # 에이전트 자동 실행
    if run_agent:
        logger.info("")
        logger.info("에이전트를 시작합니다...")
        logger.info("")
        
        # 설정 파일에서 시간 제한 읽기
        time_window_config = sweep_config.get("time_window")
        time_window_tuple = None
        if time_window_config:
            if isinstance(time_window_config, str):
                parts = time_window_config.split("-")
                if len(parts) == 2:
                    try:
                        start_hour = int(parts[0].strip())
                        end_hour = int(parts[1].strip())
                        if 0 <= start_hour < 24 and 0 <= end_hour <= 24:
                            time_window_tuple = (start_hour, end_hour)
                            logger.info(f"⏰ 시간 제한 설정: {start_hour}시~{end_hour}시 (KST)")
                            
                            # 현재 시간 확인 및 대기 여부 안내
                            import pytz
                            tz = pytz.timezone(sweep_config.get("timezone", "Asia/Seoul"))
                            now = datetime.now(tz)
                            in_window, next_start_time = _check_time_window(time_window_tuple, sweep_config.get("timezone", "Asia/Seoul"))
                            if not in_window:
                                if next_start_time:
                                    wait_seconds = (next_start_time - now).total_seconds()
                                    wait_hours = wait_seconds / 3600
                                    logger.warning("")
                                    logger.warning("⚠️  현재 시간이 스윕 실행 시간 범위 밖입니다!")
                                    logger.warning(f"   현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')} ({timezone_config})")
                                    logger.warning(f"   실행 시간 범위: {start_hour}시~{end_hour}시")
                                    logger.warning(f"   다음 시작 시간: {next_start_time.strftime('%Y-%m-%d %H:%M:%S')} ({int(wait_hours)}시간 {int((wait_seconds % 3600) // 60)}분 후)")
                                    logger.warning("")
                                    logger.warning("에이전트가 다음 시작 시간까지 대기합니다...")
                                    logger.warning("(Ctrl+C로 중단하고 나중에 다시 실행할 수 있습니다)")
                                    logger.warning("")
                    except ValueError:
                        pass
            elif isinstance(time_window_config, (list, tuple)) and len(time_window_config) == 2:
                time_window_tuple = tuple(time_window_config)
                logger.info(f"⏰ 시간 제한 설정: {time_window_tuple[0]}시~{time_window_tuple[1]}시 (KST)")
        
        timezone_config = sweep_config.get("timezone", "Asia/Seoul")
        
        # 에이전트 실행 내부 함수 호출
        _run_agent_impl(sweep_id=sweep_id, count=None, time_window=time_window_tuple, timezone=timezone_config)
    else:
        logger.info("")
        logger.info("에이전트를 실행하려면:")
        logger.info(f"  poetry run lex-dpr sweep agent --config {config_path}")
        logger.info(f"  또는")
        logger.info(f"  poetry run lex-dpr sweep agent {sweep_id}")
    
    return sweep_id

@app.command("smoke")
def sweep_smoke_command(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="스윕 설정 파일 경로 (없으면 자동 생성)",
    ),
    run_agent: bool = typer.Option(
        True,
        "--run-agent/--no-run-agent",
        help="스윕 생성 후 에이전트 자동 실행 (기본값: True)",
    ),
):
    """
    빠른 Sweep SMOKE TEST 실행용 명령어.
    
    - 최소한의 sweep config 파일을 자동 생성한 뒤 바로 실행
    - test_run=true, epochs=1로 제한하여 빠른 테스트
    
    예시:
      poetry run lex-dpr sweep smoke
      poetry run lex-dpr sweep smoke --no-run-agent
    """
    # 설정 파일이 없으면 자동 생성
    if config is None:
        config = "configs/smoke_sweep.yaml"
    
    config_path = Path(config)
    
    if not config_path.exists():
        logger.info("설정 파일이 없습니다. SMOKE TEST 모드용 설정 파일을 자동 생성합니다...")
        logger.info("")
        sweep_init(output=str(config_path), smoke_test=True)
        logger.info("")
    else:
        logger.info(f"기존 설정 파일 사용: {config_path}")
        logger.info("")
    
    # 스윕 시작
    logger.info("스윕을 시작합니다...")
    logger.info("")
    
    try:
        _run_sweep_impl(config_path, smoke_test=True, run_agent=run_agent)
    except Exception as e:
        logger.error(f"스윕 시작 실패: {e}")
        raise typer.Exit(1)

@app.callback(invoke_without_command=True)
def sweep_main(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="스윕 설정 파일 경로 (없으면 자동 생성)",
    ),
    run_agent: bool = typer.Option(
        True,
        "--run-agent/--no-run-agent",
        help="스윕 생성 후 에이전트 자동 실행 (기본값: True)",
    ),
):
    """
    WandB Sweep을 실행합니다.
    
    config 파일이 없으면 smoke 모드와 동일하게 동작합니다.
    기본적으로 스윕 생성 후 에이전트를 자동으로 실행합니다.
    
    여러 날짜에 나눠서 실행하려면:
      1. 스윕 생성: poetry run lex-dpr sweep --no-run-agent
      2. 각 날짜마다: poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 5
    
    스윕 종료 조건:
      - 기본적으로 무한정 실행 (모든 파라미터 조합 탐색)
      - --count 옵션으로 실행 횟수 제한 가능
      - WandB 대시보드에서 수동으로 중단 가능
    
    예시:
      poetry run lex-dpr sweep
      poetry run lex-dpr sweep --config configs/my_sweep.yaml
      poetry run lex-dpr sweep --no-run-agent  # 스윕만 생성하고 에이전트는 실행하지 않음
    """
    # 서브커맨드가 지정된 경우 (init, smoke, start, agent 등) 그대로 진행
    if ctx.invoked_subcommand is not None:
        return
    
    # 설정 파일이 없으면 smoke 모드와 동일하게 동작
    if config is None:
        config = "configs/sweep.yaml"
    
    config_path = Path(config)
    
    if not config_path.exists():
        logger.info("설정 파일이 없습니다. smoke 모드로 실행합니다...")
        logger.info("")
        # smoke 모드로 실행
        sweep_smoke_command(config=None, run_agent=run_agent)
        return
    
    logger.info(f"기존 설정 파일 사용: {config_path}")
    logger.info("")
    
    # 스윕 시작
    logger.info("스윕을 시작합니다...")
    logger.info("")
    
    try:
        _run_sweep_impl(config_path, smoke_test=False, run_agent=run_agent)
    except Exception as e:
        logger.error(f"스윕 시작 실패: {e}")
        raise typer.Exit(1)


def _get_sweep_template(smoke_test: bool = False) -> str:
    """스윕 설정 템플릿 반환"""
    if smoke_test:
        fixed_section = """# 고정 파라미터 (스윕 설정 파일에 직접 정의)
# 이 값들은 모든 스윕 실행에서 동일하게 사용됩니다.
fixed:
  test_run: true  # SMOKE TEST 모드: 최대 100 iteration 또는 1 epoch만 실행
  trainer.epochs: 1  # SMOKE TEST 모드: 1 epoch로 고정
  trainer.eval_steps: 50  # SMOKE TEST 모드: 더 자주 평가
  data.pairs: data/pairs_train.jsonl
  data.passages: data/merged_corpus.jsonl"""
    else:
        fixed_section = """# 고정 파라미터 (스윕 설정 파일에 직접 정의)
# 이 값들은 모든 스윕 실행에서 동일하게 사용됩니다.
fixed:
  trainer.epochs: 3
  trainer.eval_steps: 300
  data.pairs: data/pairs_train.jsonl
  data.passages: data/merged_corpus.jsonl"""
    
    return f"""# WandB Sweep 설정 파일
# 이 파일은 WandB Sweep의 하이퍼파라미터 탐색 범위를 정의합니다.
{f"# SMOKE TEST 모드: 빠른 테스트를 위한 설정 (test_run=true, epochs=1)" if smoke_test else ""}

# 프로그램 경로 (학습 스크립트)
program: lex_dpr/cli/train.py

# 탐색 방법: grid, random, bayes
method: bayes

# 최적화할 메트릭
metric:
  name: eval/ndcg@10  # WandB에 로깅되는 메트릭 이름
  goal: maximize       # maximize 또는 minimize

# 탐색할 하이퍼파라미터
parameters:
  trainer.lr:
    distribution: log_uniform_values  # log_uniform_values, uniform, categorical
    min: 0.000001  # 1e-6
    max: 0.001     # 1e-3
  
  trainer.temperature:
    distribution: uniform
    min: 0.01
    max: 0.2
  
  trainer.gradient_accumulation_steps:
    values: [4, 8, 16]  # categorical (고정 값들 중 선택)
  
  # trainer.epochs:
  #   value: 3  # 고정 값 (스윕에서 변경하지 않음)

{fixed_section}

# WandB 프로젝트 설정 (선택사항)
# project: lexdpr
# entity: your-wandb-entity
"""


def _get_sweep_preset_template() -> str:
    """넉넉한 범위의 스윕 설정 템플릿 반환"""
    return """# WandB Sweep 설정 파일 (넉넉한 범위)
# 이 파일은 WandB Sweep의 하이퍼파라미터 탐색 범위를 정의합니다.
# 넉넉한 범위로 설정되어 있어 다양한 하이퍼파라미터 조합을 탐색할 수 있습니다.

# 프로그램 경로 (학습 스크립트)
program: lex_dpr/cli/train.py

# 탐색 방법: grid, random, bayes
# bayes: Bayesian optimization (효율적, 권장)
method: bayes

# 베이지안 최적화 초기 샘플 수 (랜덤 샘플을 먼저 실행하여 베이지안 모델 학습)
# 베이지안 최적화가 제대로 작동하려면 최소 5-10개의 초기 샘플이 필요합니다.
initial_runs: 10

# 최적화할 메트릭
metric:
  name: eval/ndcg@10  # WandB에 로깅되는 메트릭 이름
  goal: maximize       # maximize 또는 minimize

# Early termination 설정 (Bayesian search에서 수렴 시 자동 종료)
early_terminate:
  type: hyperband
  min_iter: 3  # 최소 3번 평가 후 종료 판단
  max_iter: 10  # 최대 10번 평가 후 종료
  s: 2  # Successive halving factor

# 탐색할 하이퍼파라미터 (넉넉한 범위)
parameters:
  # 학습률 (넉넉한 범위)
  trainer.lr:
    distribution: log_uniform_values
    min: 0.000001  # 1e-6
    max: 0.001     # 1e-3
  
  # Loss temperature (넉넉한 범위)
  trainer.temperature:
    distribution: uniform
    min: 0.01
    max: 0.3
  
  # Optimizer weight decay (넉넉한 범위, continuous)
  trainer.weight_decay:
    distribution: uniform
    min: 0.0
    max: 0.1
  
  # Warmup ratio (넉넉한 범위, continuous)
  trainer.warmup_ratio:
    distribution: uniform
    min: 0.0
    max: 0.3
  
  # Gradient accumulation steps (넉넉한 범위)
  trainer.gradient_accumulation_steps:
    values: [2, 4, 8, 16, 32]
  
  # Gradient clipping (넉넉한 범위, continuous)
  trainer.gradient_clip_norm:
    distribution: uniform
    min: 0.0
    max: 5.0
  
  # LoRA rank (integer, categorical 유지)
  model.peft.r:
    values: [4, 8, 16, 32, 64]
  
  # LoRA alpha (integer, categorical 유지)
  model.peft.alpha:
    values: [8, 16, 32, 64, 128]
  
  # LoRA dropout (넉넉한 범위, continuous)
  model.peft.dropout:
    distribution: uniform
    min: 0.0
    max: 0.3
  
  # 배치 크기 (integer, categorical 유지)
  data.batches.bi:
    values: [8, 16, 32, 64]
  
  # 데이터 증폭 (integer, categorical 유지)
  data.multiply:
    values: [0, 1, 2, 3]

# 고정 파라미터 (모든 스윕 실행에서 동일하게 사용)
fixed:
  # 학습 설정
  trainer.epochs: 50  # 넉넉한 에포크 수 (실제 학습에서는 충분한 에포크 필요)
  trainer.eval_steps: 300  # 평가 주기
  trainer.k: 10  # 평가 시 top-k
  trainer.k_values: [1, 3, 5, 10]  # 평가 메트릭 k 값들
  
  # Early Stopping 설정 (학습 효율성)
  trainer.early_stopping.enabled: true
  trainer.early_stopping.metric: "cosine_ndcg@10"
  trainer.early_stopping.patience: 3
  trainer.early_stopping.min_delta: 0.001
  trainer.early_stopping.mode: "max"
  trainer.early_stopping.restore_best_weights: true
  
  # 모델 설정
  model.bi_model: ko-simcse  # 기본 모델 고정
  model.use_bge_template: true  # BGE 템플릿 사용
  model.max_len: 128  # 시퀀스 길이
  model.peft.enabled: true  # LoRA 활성화
  model.peft.target_modules: ["query", "value"]  # LoRA target modules 고정
  
  # 데이터 설정
  data.pairs: data/pairs_train.jsonl
  data.passages: data/merged_corpus.jsonl
  
  # 기타 설정
  test_run: false  # 실제 학습 모드
  seed: 42  # 재현성을 위한 시드

# WandB 프로젝트 설정 (선택사항)
project: lexdpr
# entity: your-wandb-entity  # WandB 엔티티 (선택사항)

# 시간 제한 설정 (기본값: 새벽 1시~8시 KST)
# 여러 날짜에 나눠서 실행할 때 사용
time_window: "1-8"  # 1시~8시에만 실행 (KST 기준)
timezone: "Asia/Seoul"
"""

@app.command("init")
def sweep_init(
    output: str = typer.Option(
        "configs/sweep.yaml",
        "--output",
        "-o",
        help="생성할 스윕 설정 파일 경로 (기본값: configs/sweep.yaml)",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test/--no-smoke-test",
        help="SMOKE TEST 모드용 템플릿 생성 (기본값: False)",
    ),
):
    """
    WandB Sweep 설정 파일 템플릿을 생성합니다.
    
    기본 템플릿을 생성합니다.
    
    예시:
      poetry run lex-dpr sweep init
      poetry run lex-dpr sweep init --output configs/my_sweep.yaml --smoke-test
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    template = _get_sweep_template(smoke_test=smoke_test)
    
    if output_path.exists():
        logger.warning(f"파일이 이미 존재합니다: {output_path}")
        response = typer.prompt("덮어쓰시겠습니까? (y/N)", default="N")
        if response.lower() != "y":
            logger.info("취소되었습니다.")
            return
    
    output_path.write_text(template, encoding="utf-8")
    mode_text = "SMOKE TEST 모드용 " if smoke_test else ""
    logger.info(f"✅ {mode_text}스윕 설정 템플릿 생성 완료: {output_path}")
    if smoke_test:
        logger.info("   (test_run=true, epochs=1로 설정됨)")
    logger.info("")
    logger.info("다음 단계:")
    logger.info("  1. 설정 파일을 편집하여 탐색할 파라미터 범위를 설정하세요")
    logger.info(f"  2. poetry run lex-dpr sweep --config {output_path} 로 스윕을 시작하세요")

@app.command("preset")
def sweep_preset(
    output: str = typer.Option(
        "configs/sweep.yaml",
        "--output",
        "-o",
        help="생성할 스윕 설정 파일 경로 (기본값: configs/sweep.yaml)",
    ),
    run: bool = typer.Option(
        True,
        "--run/--no-run",
        help="설정 파일 생성 후 바로 스윕 실행 (기본값: True)",
    ),
    run_agent: bool = typer.Option(
        True,
        "--run-agent/--no-run-agent",
        help="스윕 생성 후 에이전트 자동 실행 (기본값: True)",
    ),
):
    """
    넉넉한 범위의 WandB Sweep 설정 파일을 생성하고 바로 실행합니다.
    
    넉넉한 하이퍼파라미터 범위로 설정되어 있어 다양한 조합을 탐색할 수 있습니다.
    생성된 설정 파일에는 time_window가 1-8시(KST)로 자동 설정됩니다.
    
    예시:
      poetry run lex-dpr sweep preset
      poetry run lex-dpr sweep preset --output configs/my_sweep.yaml
      poetry run lex-dpr sweep preset --no-run  # 생성만 하고 실행하지 않음
    """
    # 로깅 설정 (즉시 출력되도록)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # 기존 설정 덮어쓰기
    )
    # 로그가 즉시 출력되도록 설정
    for handler in logging.root.handlers:
        handler.flush()
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    template = _get_sweep_preset_template()
    
    if output_path.exists():
        logger.warning(f"파일이 이미 존재합니다: {output_path}")
        response = typer.prompt("덮어쓰시겠습니까? (y/N)", default="N")
        if response.lower() != "y":
            logger.info("취소되었습니다.")
            return
    
    output_path.write_text(template, encoding="utf-8")
    logger.info(f"✅ 넉넉한 범위의 스윕 설정 파일 생성 완료: {output_path}")
    logger.info("")
    logger.info("📋 포함된 하이퍼파라미터 범위:")
    logger.info("  - 학습률: 1e-6 ~ 1e-3 (log_uniform)")
    logger.info("  - Temperature: 0.01 ~ 0.3 (uniform)")
    logger.info("  - Weight Decay: 0.0 ~ 0.1 (uniform)")
    logger.info("  - Warmup Ratio: 0.0 ~ 0.3 (uniform)")
    logger.info("  - Gradient Accumulation Steps: [2, 4, 8, 16, 32]")
    logger.info("  - Gradient Clipping: 0.0 ~ 5.0 (uniform)")
    logger.info("  - LoRA rank: [4, 8, 16, 32, 64]")
    logger.info("  - LoRA alpha: [8, 16, 32, 64, 128]")
    logger.info("  - LoRA dropout: 0.0 ~ 0.3 (uniform)")
    logger.info("  - 배치 크기: [8, 16, 32, 64]")
    logger.info("  - 데이터 증폭: [0, 1, 2, 3]")
    logger.info("")
    
    if run:
        logger.info("=" * 80)
        logger.info("스윕을 시작합니다...")
        logger.info("=" * 80)
        logger.info("")
        try:
            _run_sweep_impl(output_path, smoke_test=False, run_agent=run_agent)
        except Exception as e:
            logger.error(f"스윕 시작 실패: {e}")
            raise typer.Exit(1)
    else:
        logger.info("다음 단계:")
        logger.info(f"  poetry run lex-dpr sweep --config {output_path} 로 스윕을 시작하세요")


@app.command("start")
def sweep_start(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="스윕 설정 파일 경로",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="WandB 프로젝트 이름 (설정 파일의 project 우선)",
    ),
    entity: Optional[str] = typer.Option(
        None,
        "--entity",
        "-e",
        help="WandB 엔티티 이름 (설정 파일의 entity 우선)",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="SMOKE TEST 모드로 실행 (test_run=true, epochs=1 자동 추가)",
    ),
):
    """
    WandB에 스윕을 생성하고 시작합니다.
    
    스윕 ID를 반환하며, 이 ID를 사용하여 에이전트를 실행할 수 있습니다.
    
    예시:
      poetry run lex-dpr sweep start --config configs/my_sweep.yaml
      poetry run lex-dpr sweep start --config configs/my_sweep.yaml --smoke-test
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb가 설치되지 않았습니다. 'poetry install --extras wandb'로 설치하세요.")
        raise typer.Exit(1)
    
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise typer.Exit(1)
    
    # 스윕 설정 파일 로드
    sweep_config = OmegaConf.load(config_path)
    
    # SMOKE TEST 모드 처리
    if smoke_test:
        logger.info("🧪 SMOKE TEST 모드로 실행합니다.")
        # fixed 섹션에 test_run과 epochs 추가/수정
        if "fixed" not in sweep_config:
            sweep_config["fixed"] = {}
        sweep_config["fixed"]["test_run"] = True
        sweep_config["fixed"]["trainer.epochs"] = 1
        # eval_steps도 더 짧게 설정 (없으면)
        if "trainer.eval_steps" not in sweep_config.get("fixed", {}):
            sweep_config["fixed"]["trainer.eval_steps"] = 50
    
    # WandB 프로젝트 설정
    wandb_project = project or _convert_to_dict(sweep_config.get("project", "lexdpr"))
    wandb_entity = entity or _convert_to_dict(sweep_config.get("entity", None))
    
    # 문자열로 변환 (OmegaConf 객체일 수 있음)
    if wandb_project and not isinstance(wandb_project, str):
        wandb_project = str(wandb_project)
    if wandb_entity and not isinstance(wandb_entity, str):
        wandb_entity = str(wandb_entity)
    
    # SMOKE TEST 모드일 경우 프로젝트 이름에 접미사 추가
    if smoke_test:
        wandb_project = f"{wandb_project}-smoke-test"
    
    # 프로그램 경로 확인
    program = _convert_to_dict(sweep_config.get("program", "lex_dpr/cli/train.py"))
    if not isinstance(program, str):
        program = str(program)
    if not Path(program).exists():
        logger.warning(f"프로그램 경로를 찾을 수 없습니다: {program}")
        logger.info("상대 경로로 시도합니다...")
    
    # 스윕 설정 딕셔너리 생성 (WandB 형식)
    # OmegaConf 객체를 일반 Python 딕셔너리로 변환
    method = _convert_to_dict(sweep_config.get("method", "random"))
    metric = _convert_to_dict(sweep_config.get("metric", {"name": "eval/ndcg@10", "goal": "maximize"}))
    parameters = _convert_to_dict(sweep_config.get("parameters", {}))
    
    sweep_dict = {
        "method": method,
        "metric": metric,
        "parameters": parameters or {},
    }
    
    # Early termination 설정 추가
    early_terminate = sweep_config.get("early_terminate")
    if early_terminate:
        early_terminate_dict = _convert_to_dict(early_terminate)
        sweep_dict["early_terminate"] = early_terminate_dict
        logger.info(f"Early termination 설정: {early_terminate_dict}")
    
    # fixed 파라미터를 parameters에 추가 (WandB는 fixed를 직접 지원하지 않으므로 value로 추가)
    fixed_params = _convert_to_dict(sweep_config.get("fixed", {}))
    if fixed_params:
        logger.info(f"고정 파라미터 적용: {list(fixed_params.keys())}")
        for key, value in fixed_params.items():
            # 점(.)으로 구분된 키를 중첩 구조로 변환
            if key not in sweep_dict["parameters"]:
                sweep_dict["parameters"][key] = {"value": _convert_to_dict(value)}
            else:
                logger.warning(f"파라미터 {key}가 이미 parameters에 정의되어 있습니다. fixed 값이 무시됩니다.")
    
    # WandB에 스윕 생성
    logger.info(f"WandB 프로젝트: {wandb_project}")
    if wandb_entity:
        logger.info(f"WandB 엔티티: {wandb_entity}")
    logger.info("스윕 생성 중...")
    
    sweep_id = wandb.sweep(
        sweep_dict,
        project=wandb_project,
        entity=wandb_entity,
    )
    
    # 스윕 ID를 설정 파일에 저장
    sweep_config["sweep_id"] = sweep_id
    OmegaConf.save(sweep_config, config_path)
    logger.info(f"스윕 ID가 설정 파일에 저장되었습니다: {config_path}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 스윕 생성 완료!")
    logger.info(f"스윕 ID: {sweep_id}")
    logger.info("")
    logger.info("다음 단계:")
    logger.info(f"  poetry run lex-dpr sweep agent --config {config}")
    logger.info(f"  또는")
    logger.info(f"  poetry run lex-dpr sweep agent {sweep_id}")
    logger.info("")
    logger.info(f"WandB 대시보드: https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}/sweeps/{sweep_id}")
    logger.info("=" * 80)
    
    return sweep_id


def _run_agent_impl(
    sweep_id: str,
    count: Optional[int] = None,
    time_window: Optional[Tuple[int, int]] = None,
    timezone: str = "Asia/Seoul",
):
    """에이전트 실행 내부 구현 함수"""
    try:
        import wandb
    except ImportError:
        logger.error("wandb가 설치되지 않았습니다. 'poetry install --extras wandb'로 설치하세요.")
        raise typer.Exit(1)
    
    logger.info("=" * 80)
    logger.info("🔍 WandB Sweep 에이전트 시작")
    logger.info(f"스윕 ID: {sweep_id}")
    if count:
        logger.info(f"실행 횟수: {count}")
    logger.info("=" * 80)
    logger.info("")
    
    # 설정 파일 로드
    base_path = _get_config_path("base.yaml")
    data_path = _get_config_path("data.yaml")
    model_path = _get_config_path("model.yaml")
    
    logger.info("설정 파일 로드 중...")
    base = OmegaConf.load(base_path)
    
    if data_path.exists():
        data = OmegaConf.load(data_path)
        base = OmegaConf.merge(base, {"data": data})
    
    if model_path.exists():
        model = OmegaConf.load(model_path)
        base = OmegaConf.merge(base, {"model": model})
    
    cfg = base
    
    # WandB 에이전트 실행 함수 정의
    def train_fn():
        """WandB 에이전트가 호출하는 학습 함수"""
        # wandb.config는 이미 설정되어 있음
        # train.py의 main()을 호출하여 WandB Sweep 모드로 실행
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["train"]
            from lex_dpr.cli import train as train_module
            train_module.main()
        finally:
            sys.argv = original_argv
    
    # WandB 에이전트 실행
    try:
        while True:
            if time_window:
                in_window, next_start_time = _check_time_window(time_window, timezone)
                if not in_window:
                    if next_start_time:
                        import pytz
                        tz = pytz.timezone(timezone)
                        wait_seconds = (next_start_time - datetime.now(tz)).total_seconds()
                        logger.info(f"현재 시간은 스윕 실행 시간 범위({time_window[0]}-{time_window[1]}시) 밖입니다.")
                        logger.info(f"다음 시작 시간까지 대기합니다: {next_start_time.strftime('%Y-%m-%d %H:%M:%S')} ({int(wait_seconds // 60)}분 후)")
                        time.sleep(wait_seconds + 5)  # 5초 여유
                        continue  # 다시 시간 체크
                    else:
                        logger.info("시간 범위 설정 오류 또는 pytz 미설치로 시간 제한 없이 에이전트를 실행합니다.")
            
            wandb.agent(sweep_id, function=train_fn, count=count)
            if count is not None:  # count가 지정된 경우, 한 번 실행 후 종료
                break
            # count가 None인 경우, 시간 제한이 있다면 다음 루프에서 다시 체크
            if time_window:
                time.sleep(60)  # 1분 대기 후 다시 시간 체크
            else:  # 시간 제한이 없으면 무한 루프 방지
                break  # count가 없으면 한 번만 실행하고 종료 (기존 동작 유지)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("에이전트가 중단되었습니다.")
    except Exception as e:
        logger.error(f"에이전트 실행 실패: {e}")
        raise


@app.command("agent")
def sweep_agent(
    sweep_id: Optional[str] = typer.Argument(None, help="WandB 스윕 ID (없으면 설정 파일에서 읽음)"),
    config: Optional[str] = typer.Option(
        "configs/sweep.yaml",
        "--config",
        "-c",
        help="스윕 설정 파일 경로 (sweep_id가 없을 때 사용, 기본값: configs/sweep.yaml)",
    ),
    count: Optional[int] = typer.Option(
        None,
        "--count",
        help="실행할 에이전트 실행 횟수 (None이면 무제한)",
    ),
    time_window: Optional[str] = typer.Option(
        None,
        "--time-window",
        help="실행 시간 범위 (예: '1-8' = 1시~8시, KST 기준)",
    ),
    timezone: str = typer.Option(
        "Asia/Seoul",
        "--timezone",
        help="타임존 (기본값: Asia/Seoul)",
    ),
):
    """
    WandB Sweep 에이전트를 실행합니다.
    
    스윕에서 제공하는 파라미터로 학습을 실행합니다.
    여러 머신에서 동시에 실행하여 병렬 탐색이 가능합니다.
    여러 날짜에 나눠서 실행할 수도 있습니다.
    
    스윕 ID를 직접 지정하거나, 설정 파일에서 자동으로 읽을 수 있습니다.
    
    여러 날짜에 나눠서 실행하는 방법:
      1. 첫 날: poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10
      2. 둘째 날: poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10
      3. 셋째 날: poetry run lex-dpr sweep agent --config configs/my_sweep.yaml --count 10
      (같은 스윕에 계속 참여하여 탐색 진행)
    
    스윕 종료 조건:
      - 기본적으로 무한정 실행 (모든 파라미터 조합 탐색)
      - --count 옵션으로 해당 에이전트의 실행 횟수만 제한
      - WandB 대시보드에서 수동으로 스윕 중단 가능
    
    예시:
      poetry run lex-dpr sweep agent --config configs/smoke_sweep.yaml
      poetry run lex-dpr sweep agent <sweep-id>
      poetry run lex-dpr sweep agent <sweep-id> --count 5  # 5개만 실행하고 종료
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb가 설치되지 않았습니다. 'poetry install --extras wandb'로 설치하세요.")
        raise typer.Exit(1)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # sweep_id가 없으면 설정 파일에서 읽기
    if sweep_id is None:
        if config is None:
            # 기본값으로 configs/sweep.yaml 사용
            config = "configs/sweep.yaml"
        
        config_path = Path(config)
        if not config_path.exists():
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            logger.error("먼저 'poetry run lex-dpr sweep preset'으로 스윕 설정 파일을 생성하세요.")
            raise typer.Exit(1)
        
        sweep_config = OmegaConf.load(config_path)
        sweep_id = sweep_config.get("sweep_id")
        if sweep_id is None:
            logger.error(f"설정 파일에 sweep_id가 없습니다: {config_path}")
            logger.error("먼저 'poetry run lex-dpr sweep --config {config_path}' 또는 'poetry run lex-dpr sweep preset'으로 스윕을 생성하세요.")
            raise typer.Exit(1)
        
        logger.info(f"설정 파일에서 sweep_id를 읽었습니다: {sweep_id}")
        
        # 설정 파일에서 time_window와 timezone 읽기 (설정 파일 우선)
        time_window_config = sweep_config.get("time_window")
        if time_window_config:
            if isinstance(time_window_config, str):
                time_window = time_window_config  # CLI 옵션보다 설정 파일 우선
            elif isinstance(time_window_config, (list, tuple)) and len(time_window_config) == 2:
                time_window = f"{time_window_config[0]}-{time_window_config[1]}"
        
        timezone_config = sweep_config.get("timezone")
        if timezone_config:
            timezone = timezone_config  # CLI 옵션보다 설정 파일 우선
    
    logger.info("=" * 80)
    logger.info("🔍 WandB Sweep 에이전트 시작")
    logger.info(f"스윕 ID: {sweep_id}")
    if count:
        logger.info(f"실행 횟수: {count}")
    if time_window:
        logger.info(f"실행 시간 범위: {time_window} ({timezone})")
    logger.info("=" * 80)
    logger.info("")
    
    # time_window 파싱
    time_window_tuple = None
    if time_window:
        if isinstance(time_window, str):
            parts = time_window.split("-")
            if len(parts) == 2:
                try:
                    start_hour = int(parts[0].strip())
                    end_hour = int(parts[1].strip())
                    if 0 <= start_hour < 24 and 0 <= end_hour <= 24:
                        time_window_tuple = (start_hour, end_hour)
                    else:
                        logger.warning(f"잘못된 시간 범위: {time_window}. 시간 제한 없이 실행합니다.")
                except ValueError:
                    logger.warning(f"잘못된 시간 범위 형식: {time_window}. 시간 제한 없이 실행합니다.")
        elif isinstance(time_window, (list, tuple)) and len(time_window) == 2:
            time_window_tuple = tuple(time_window)
    
    # _run_agent_impl 호출
    _run_agent_impl(sweep_id=sweep_id, count=count, time_window=time_window_tuple, timezone=timezone)


@app.command("run")
def sweep_run(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="스윕 설정 파일 경로",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="WandB 프로젝트 이름",
    ),
    count: Optional[int] = typer.Option(
        None,
        "--count",
        help="실행할 에이전트 실행 횟수",
    ),
):
    """
    스윕 설정 파일로부터 직접 스윕을 시작하고 실행합니다.
    
    내부적으로 'start'와 'agent'를 순차적으로 실행합니다.
    
    예시:
      poetry run lex-dpr sweep run --config configs/my_sweep.yaml
    """
    # 스윕 시작
    logger.info("스윕 시작 중...")
    sweep_start(config=config, project=project, entity=None)
    
    # 스윕 ID를 어떻게 전달할지? 임시로 사용자에게 안내
    logger.info("")
    logger.info("스윕이 생성되었습니다. 위에 표시된 스윕 ID를 사용하여 에이전트를 실행하세요:")
    logger.info("  poetry run lex-dpr sweep agent <sweep-id>")


if __name__ == "__main__":
    app()

