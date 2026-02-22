"""
WandB Sweep CLI 명령어

사용 예시:
  poetry run lex-dpr sweep init --output configs/my_sweep.yaml
  poetry run lex-dpr sweep start --config configs/my_sweep.yaml
  poetry run lex-dpr sweep agent <sweep-id>
  poetry run lex-dpr sweep run --config configs/my_sweep.yaml
"""

import logging
import os
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

# PyTorch import (OOM 처리용)
try:
    import torch
except ImportError:
    torch = None

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


def _sample_params_local(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    로컬에서 파라미터 샘플링 (random 방법만 지원)
    
    Args:
        parameters: sweep 설정의 parameters 딕셔너리
        
    Returns:
        샘플링된 파라미터 딕셔너리
    """
    import random
    import math
    
    sampled = {}
    
    for key, param_config in parameters.items():
        if isinstance(param_config, dict):
            # value로 고정된 값
            if "value" in param_config:
                sampled[key] = param_config["value"]
            # values로 categorical 선택
            elif "values" in param_config:
                sampled[key] = random.choice(param_config["values"])
            # distribution으로 연속값 샘플링
            elif "distribution" in param_config:
                dist = param_config["distribution"]
                if dist == "uniform":
                    min_val = param_config.get("min", 0.0)
                    max_val = param_config.get("max", 1.0)
                    sampled[key] = random.uniform(min_val, max_val)
                elif dist == "log_uniform_values":
                    min_val = param_config.get("min", 1e-6)
                    max_val = param_config.get("max", 1.0)
                    # log space에서 uniform 샘플링
                    log_min = math.log10(min_val)
                    log_max = math.log10(max_val)
                    log_val = random.uniform(log_min, log_max)
                    sampled[key] = 10 ** log_val
                else:
                    # 지원하지 않는 distribution은 기본값 사용
                    if "min" in param_config and "max" in param_config:
                        sampled[key] = random.uniform(param_config["min"], param_config["max"])
                    else:
                        sampled[key] = param_config.get("value", 0.0)
            else:
                # 알 수 없는 형식은 value가 있으면 사용
                sampled[key] = param_config.get("value", 0.0)
        else:
            # 단순 값
            sampled[key] = param_config
    
    return sampled


def _build_param_space(parameters: Dict[str, Any]) -> list:
    """탐색 가능한 파라미터만 (key, type, bounds_or_values) 리스트로 반환. 'value' 고정은 제외."""
    space = []
    for key, param_config in parameters.items():
        if not isinstance(param_config, dict):
            continue
        if "value" in param_config:
            continue
        if "values" in param_config:
            space.append((key, "categorical", list(param_config["values"])))
        elif param_config.get("distribution") == "log_uniform_values":
            mn = float(param_config.get("min", 1e-6))
            mx = float(param_config.get("max", 1.0))
            import math
            space.append((key, "log_continuous", (math.log10(mn), math.log10(mx))))
        elif param_config.get("distribution") == "uniform" or "min" in param_config:
            mn = float(param_config.get("min", 0.0))
            mx = float(param_config.get("max", 1.0))
            space.append((key, "continuous", (mn, mx)))
    return space


def _params_to_vector(params_dict: Dict[str, Any], param_space: list) -> Any:
    """파라미터 딕셔너리를 고정 길이 실수 벡터로 인코딩 (GP 입력용)."""
    import numpy as np
    vec = []
    for key, typ, bounds_or_values in param_space:
        v = params_dict.get(key)
        if typ == "categorical":
            try:
                idx = list(bounds_or_values).index(v)
            except (ValueError, TypeError):
                idx = 0
            vec.append(float(idx))
        elif typ == "log_continuous":
            import math
            x = float(v) if v is not None else (bounds_or_values[0] + bounds_or_values[1]) / 2
            vec.append(math.log10(max(x, 1e-10)))
        else:
            vec.append(float(v) if v is not None else (bounds_or_values[0] + bounds_or_values[1]) / 2)
    return np.array(vec, dtype=np.float64).reshape(1, -1)


def _vector_to_params(vec: Any, param_space: list) -> Dict[str, Any]:
    """벡터를 파라미터 딕셔너리로 디코딩."""
    params = {}
    for i, (key, typ, bounds_or_values) in enumerate(param_space):
        x = float(vec.flat[i])
        if typ == "categorical":
            vals = list(bounds_or_values)
            idx = max(0, min(int(round(x)), len(vals) - 1))
            params[key] = vals[idx]
        elif typ == "log_continuous":
            params[key] = 10 ** x
        else:
            lo, hi = bounds_or_values
            params[key] = float(max(lo, min(x, hi)))
    return params


def _suggest_next_bayes(
    parameters: Dict[str, Any],
    param_space: list,
    history: list,
    goal: str,
    n_initial: int = 5,
    n_candidates: int = 200,
) -> Dict[str, Any]:
    """
    이력 기반으로 다음 설정 추천 (GP + Expected Improvement). 추가 의존성 없이 sklearn+numpy 사용.
    history: [(params_dict, metric), ...]
    goal: "maximize" or "minimize"
    """
    import random
    import numpy as np
    from scipy.stats import norm

    if len(history) < n_initial:
        return _sample_params_local(parameters)

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError:
        return _sample_params_local(parameters)

    X_list = []
    y_list = []
    for params_dict, metric in history:
        if metric is None:
            continue
        try:
            x = _params_to_vector(params_dict, param_space)
            X_list.append(x.ravel())
            y_list.append(float(metric))
        except Exception:
            continue

    if len(y_list) < n_initial:
        return _sample_params_local(parameters)

    X = np.array(X_list)
    y = np.array(y_list)
    if goal == "minimize":
        y = -y

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4, alpha=1e-6)
    gp.fit(X, y)

    y_best = float(np.max(y))
    candidates = [_sample_params_local(parameters) for _ in range(n_candidates)]
    X_cand = np.vstack([_params_to_vector(p, param_space) for p in candidates])
    mu, std = gp.predict(X_cand, return_std=True)
    std = np.maximum(std, 1e-6)
    z = (mu - y_best) / std
    ei = (mu - y_best) * norm.cdf(z) + std * norm.pdf(z)
    best_idx = int(np.argmax(ei))
    return candidates[best_idx]


def _run_sweep_impl(
    config_path: Path,
    smoke_test: bool,
    run_agent: bool,
    force_local: bool = False,
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
    metric = _convert_to_dict(sweep_config.get("metric", {"name": "eval/ndcg_at_10", "goal": "maximize"}))
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
    
    # 로컬 모드 플래그
    is_local_mode = force_local
    
    if force_local:
        logger.info("💾 로컬 모드 강제 활성화 (네트워크 연결 없이 실행)")
        logger.info("")
        sweep_id = f"local-{int(time.time())}"
    else:
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
            # 네트워크 연결 실패 시 로컬 모드로 전환
            error_msg = str(e).lower()
            is_network_error = (
                "connection" in error_msg or 
                "timeout" in error_msg or 
                "network" in error_msg or
                "unreachable" in error_msg or
                "refused" in error_msg
            )
            
            if is_network_error:
                logger.warning("")
                logger.warning("⚠️  WandB 서버 연결 실패 - 로컬 모드로 전환합니다")
                logger.warning(f"   에러: {str(e)[:100]}")
                logger.warning("")
                is_local_mode = True
                # 로컬 모드용 가짜 sweep_id 생성
                sweep_id = f"local-{int(time.time())}"
            else:
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
    sweep_config["_local_mode"] = is_local_mode  # 로컬 모드 플래그 저장
    OmegaConf.save(sweep_config, config_path)
    logger.info(f"스윕 ID가 설정 파일에 저장되었습니다: {config_path}")
    if is_local_mode:
        logger.info("💾 로컬 모드: 네트워크 없이 로컬에서만 실행됩니다")
    
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
        if not is_local_mode:
            # 스윕 생성 직후 WandB 서버에 완전히 반영되기까지 잠시 대기
            # (특히 nohup 등 비동기 환경에서 타이밍 이슈 방지)
            logger.info("")
            logger.info("⏳ WandB 서버에 스윕이 완전히 반영되기를 기다리는 중... (2초)")
            time.sleep(2)  # 2초 대기
            logger.info("✅ 대기 완료")
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
        _run_agent_impl(
            sweep_id=sweep_id, 
            count=None, 
            time_window=time_window_tuple, 
            timezone=timezone_config,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            config_path=config_path,
        )
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
        _run_sweep_impl(config_path, smoke_test=True, run_agent=run_agent, force_local=False)
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
        _run_sweep_impl(config_path, smoke_test=False, run_agent=run_agent, force_local=False)
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
  data.passages: data/processed/merged_corpus.jsonl"""
    else:
        fixed_section = """# 고정 파라미터 (스윕 설정 파일에 직접 정의)
# 이 값들은 모든 스윕 실행에서 동일하게 사용됩니다.
fixed:
  trainer.epochs: 3
  trainer.eval_steps: 300
  data.pairs: data/pairs_train.jsonl
  data.passages: data/processed/merged_corpus.jsonl"""
    
    return f"""# WandB Sweep 설정 파일
# 이 파일은 WandB Sweep의 하이퍼파라미터 탐색 범위를 정의합니다.
{f"# SMOKE TEST 모드: 빠른 테스트를 위한 설정 (test_run=true, epochs=1)" if smoke_test else ""}

# 프로그램 경로 (학습 스크립트)
program: lex_dpr/cli/train.py

# 탐색 방법: grid, random, bayes
method: bayes

# 최적화할 메트릭
metric:
  name: eval/ndcg_at_10  # WandB에 로깅되는 메트릭 이름 (@는 _at_로 변환됨)
  goal: maximize       # maximize 또는 minimize

# 탐색할 하이퍼파라미터
parameters:
  trainer.lr:
    distribution: log_uniform_values  # log_uniform_values, uniform, categorical
    min: 0.000001  # 1e-6
    max: 0.0001     # 1e-4
  
  trainer.temperature:
    distribution: uniform
    min: 0.01
    max: 0.2
  
  trainer.gradient_accumulation_steps:
    values: [4, 8, 16, 32]  # categorical (고정 값들 중 선택)
  
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
  name: eval/recall_at_10  # WandB에 로깅되는 메트릭 이름 (@는 _at_로 변환됨)
  goal: maximize       # maximize 또는 minimize

# Early termination 설정 (Bayesian search에서 수렴 시 자동 종료)
# Hyperband 알고리즘: 여러 run 중 성능이 낮은 run을 조기에 종료하여 리소스 절약
# epochs=50, eval_steps=300이면 대략 50번의 평가가 가능하므로 max_iter를 충분히 크게 설정
early_terminate:
  type: hyperband
  min_iter: 10  # 최소 30번 평가 후 종료 판단 (너무 일찍 종료 방지)
  max_iter: 5000  # 최대 평가 수 크게 설정
  s: 2  # Successive halving factor

# 탐색할 하이퍼파라미터 (넉넉한 범위)
parameters:
  # 학습률 (보수적인 범위로 조정 - overfitting 방지)
  # log_uniform 분포: 작은 학습률에 더 많은 샘플링 (일반적으로 작은 학습률이 더 안정적)
  trainer.lr:
    distribution: log_uniform_values
    min: 0.0000005  # 5e-7 (더 낮게 조정)
    max: 0.0001     # 1e-4 (더 낮게 조정)
  
  # Loss temperature (넉넉한 범위)
  trainer.temperature:
    distribution: uniform
    min: 0.01
    max: 0.3
  
  # Optimizer weight decay (정규화 강도 조절)
  # 최소값을 0.01로 상향하여 기본적인 정규화 보장 (overfitting 방지)
  trainer.weight_decay:
    distribution: uniform
    min: 0.01      # 0.0 제거, 최소 정규화 보장
    max: 0.5
  
  # Warmup ratio (넉넉한 범위, continuous)
  trainer.warmup_ratio:
    distribution: uniform
    min: 0.0
    max: 0.2
  
  # Gradient accumulation steps (넉넉한 범위)
  # 효과적인 배치 크기 = 배치 크기 × gradient_accumulation_steps
  # 메모리가 부족할 때 작은 배치 크기와 큰 accumulation steps 조합 사용
  trainer.gradient_accumulation_steps:
    values: [16, 32, 64]
  
  # Gradient clipping (넉넉한 범위, continuous)
  trainer.gradient_clip_norm:
    distribution: uniform
    min: 1.0
    max: 20.0
  
  # LoRA rank (integer, categorical 유지)
  model.peft.r:
    values: [4, 8, 16]
  
  # LoRA alpha (integer, categorical 유지)
  model.peft.alpha:
    values: [8, 16, 32]
  
  # LoRA dropout (정규화 강도 조절)
  # 최소값을 0.05로 상향하여 기본적인 dropout 보장 (overfitting 방지)
  model.peft.dropout:
    distribution: uniform
    min: 0.05      # 0.0 제거, 최소 dropout 보장
    max: 0.1
  
  # 배치 크기 (메모리 효율적인 범위)
  # 배치 내 negative sampling 덕분에 작은 배치로도 학습이 가능합니다.
  # 메모리가 충분한 경우 큰 배치 크기로 학습 속도 향상 가능
  data.batches.bi:
    values: [32, 64, 128, 256]  # 메모리 허용 시 큰 배치 크기 사용 가능
  
  # 데이터 증폭 (integer, categorical 유지)
  # 최대값을 2로 제한하여 과도한 증폭 방지 (overfitting 방지)
  data.multiply:
    values: [0, 1, 2]  # 3 제거 (과도한 증폭 방지)
  
  # Hard negative 사용 여부 및 비율
  # Hard negative와 in-batch negative를 섞어서 사용
  data.use_hard_negatives:
    values: [true]  # false = in-batch만, true = hard negative 포함
  
  # Hard negative 비율 (use_hard_negatives=true일 때만 적용)
  # 0.0 = in-batch negative만 사용, 1.0 = hard negative만 사용
  # 중간 값은 두 방식을 혼합하여 사용
  data.hard_negative_ratio:
    distribution: uniform
    min: 0.2
    max: 0.6
  
  # Validation loss 계산 시 전체 corpus에서 negative 샘플링
  trainer.use_full_corpus_negatives:
    values: [true]  # 항상 활성화 (실전 모방)
  
  # Validation loss 계산 시 각 query당 샘플링할 negative 개수
  # 작은 값(64)은 빠른 평가, 큰 값(1024)은 더 정확한 평가
  trainer.num_negatives_per_query:
    values: [16, 32, 64]  # 전체 corpus에서 샘플링할 negative 개수 (메모리와 정확도 트레이드오프)
  
  # 기본 모델 (categorical)
  # bge-m3-ko는 메모리 사용량이 크므로 주의 필요 (배치 크기/시퀀스 길이 조절 권장)
  model.bi_model:
    values: [
      ko-simcse,  # ~110M, 한국어 전용, 작은 메모리 (권장)
      bge-m3-ko,  # ~560M, 한국어 최적화, 큰 메모리 (OOM 주의)
      multilingual-minilm,  # ~117M, 다국어, 경량
      multilingual-e5-small  # ~118M, 다국어, E5 시리즈
    ]
  
  # 시퀀스 길이 (메모리 효율적인 범위로 제한)
  # 768은 메모리 사용량이 매우 크므로 제거
  model.max_len:
    values: [256, 384, 512]  # 768 제거 (OOM 방지)


# 고정 파라미터 (모든 스윕 실행에서 동일하게 사용)
fixed:
  # 학습 설정
  trainer.epochs: 200  # 넉넉한 에포크 수 (실제 학습에서는 충분한 에포크 필요)
  trainer.eval_steps: 300  # 평가 주기
  trainer.k: 20  # 평가 시 top-k
  trainer.k_values: [1, 3, 5, 10, 20]  # 평가 메트릭 k 값들
  
  # Early Stopping 설정 (학습 효율성)
  trainer.early_stopping.enabled: true
  trainer.early_stopping.metric: "cosine_recall@10"
  trainer.early_stopping.patience: 8
  trainer.early_stopping.min_delta: 0.001
  trainer.early_stopping.mode: "max"
  trainer.early_stopping.restore_best_weights: true
  
  # 메모리 최적화 설정
  trainer.gradient_checkpointing: true  # 메모리 절약 (약 10GB 절감)
  trainer.use_amp: true  # Mixed precision training (메모리 절약)
  
  # 모델 설정
  model.use_bge_template: true  # BGE 템플릿 사용
  model.peft.enabled: true  # LoRA 활성화
  model.peft.target_modules: ["query", "value"]  # LoRA target modules 고정
  
  # 데이터 설정
  data.pairs: data/pairs_train.jsonl
  data.passages: data/processed/merged_corpus.jsonl
  
  # 기타 설정
  test_run: false  # 실제 학습 모드
  seed: 42  # 재현성을 위한 시드

# WandB 프로젝트 설정 (선택사항)
project: lexdpr
entity: zae-park  # WandB 엔티티 (선택사항, 없으면 현재 로그인한 사용자 사용)

# 시간 제한 설정 (기본값: 오후 5시~새벽 7시 KST)
# 여러 날짜에 나눠서 실행할 때 사용
# 17-7은 17시(오후 5시)부터 다음날 7시(새벽 7시)까지 실행
time_window: "0-24"  # 17시~7시에만 실행 (KST 기준)
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
    local: bool = typer.Option(
        False,
        "--local/--no-local",
        help="로컬 모드 강제 활성화 (네트워크 연결 없이 실행, 기본값: False)",
    ),
):
    """
    넉넉한 범위의 WandB Sweep 설정 파일을 생성하고 바로 실행합니다.
    
    넉넉한 하이퍼파라미터 범위로 설정되어 있어 다양한 조합을 탐색할 수 있습니다.
    생성된 설정 파일에는 time_window가 17-7시(KST)로 자동 설정됩니다.
    
    주요 하이퍼파라미터 범위:
    - 배치 크기: 32~256 (메모리 허용 시 큰 배치 사용 가능)
    - Weight Decay: 0.0~0.5 (넉넉한 정규화 범위)
    - Gradient Accumulation: 2~64 (효과적인 배치 크기 조절)
    - Validation Negative 샘플링: 64~1024 (메모리와 정확도 트레이드오프)
    
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
    logger.info("  - 학습률: 5e-7 ~ 1e-4 (log_uniform, 보수적 범위)")
    logger.info("  - Temperature: 0.01 ~ 0.3 (uniform)")
    logger.info("  - Weight Decay: 0.01 ~ 0.5 (uniform, 최소 정규화 보장)")
    logger.info("  - Warmup Ratio: 0.0 ~ 0.2 (uniform)")
    logger.info("  - Gradient Accumulation Steps: [16, 32, 64]")
    logger.info("  - Gradient Clipping: 1.0 ~ 20.0 (uniform)")
    logger.info("  - LoRA rank: [4, 8, 16]")
    logger.info("  - LoRA alpha: [8, 16, 32]")
    logger.info("  - LoRA dropout: 0.05 ~ 0.3 (uniform, 최소 dropout 보장)")
    logger.info("  - 배치 크기: [32, 64, 128, 256]")
    logger.info("  - 데이터 증폭: [0, 1, 2] (과도한 증폭 방지)")
    logger.info("  - Hard Negative 비율: 0.0 ~ 1.0 (uniform)")
    logger.info("  - Validation Negative 샘플링: [16, 32, 64]")
    logger.info("  - 기본 모델: [ko-simcse, bge-m3-ko, multilingual-minilm, multilingual-e5-small]")
    logger.info("    * ko-simcse: ~110M, 한국어 전용, 작은 메모리 (권장)")
    logger.info("    * bge-m3-ko: ~560M, 한국어 최적화, 큰 메모리 (OOM 주의, 배치/시퀀스 길이 조절 권장)")
    logger.info("    * multilingual-minilm: ~117M, 다국어, 경량")
    logger.info("    * multilingual-e5-small: ~118M, 다국어, E5 시리즈")
    logger.info("  - 시퀀스 길이: [256, 384, 512]")
    logger.info("  - Early Stopping Patience: [5, 8, 10, 15, 20] (overfitting 방지)")
    logger.info("")
    
    if run:
        logger.info("=" * 80)
        logger.info("스윕을 시작합니다...")
        logger.info("=" * 80)
        logger.info("")
        try:
            _run_sweep_impl(output_path, smoke_test=False, run_agent=run_agent, force_local=local)
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
    metric = _convert_to_dict(sweep_config.get("metric", {"name": "eval/ndcg_at_10", "goal": "maximize"}))
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
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    config_path: Optional[Path] = None,
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
    
    # WandB 프로젝트 및 엔티티 정보 설정 (함수 파라미터로 받음, 없으면 기본값 사용)
    if wandb_project is None:
        wandb_project = "lexdpr"
    
    logger.info(f"WandB 프로젝트: {wandb_project}")
    if wandb_entity:
        logger.info(f"WandB 엔티티: {wandb_entity}")
    else:
        logger.info("WandB 엔티티: (자동 - 현재 로그인한 사용자)")
    logger.info("")
    
    # WandB 에이전트 실행 함수 정의
    def train_fn():
        """WandB 에이전트가 호출하는 학습 함수"""
        # 각 run 시작 전에 GPU 메모리 정리 (이전 run의 잔여 메모리 제거)
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                # 강력한 메모리 정리
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # 모든 GPU 디바이스에서 메모리 정리
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()  # IPC 메모리 정리
                # Python GC로 남은 객체 정리
                gc.collect()
                gc.collect()  # 추가 GC (순환 참조 정리)
                logger.debug("Run 시작 전 GPU 메모리 정리 완료")
        except Exception as e:
            logger.debug(f"메모리 정리 중 오류 (무시됨): {e}")
        
        # wandb.agent()가 자동으로 wandb.init()을 호출하고 wandb.config를 설정함
        # 하지만 명시적으로 확인하고 로깅
        try:
            import wandb
            logger.info("=" * 80)
            logger.info("🚀 WandB Sweep Run 시작")
            logger.info(f"   wandb.run 존재: {wandb.run is not None}")
            
            # wandb.init()이 호출되지 않았으면 명시적으로 호출
            if wandb.run is None:
                logger.warning("wandb.run이 None입니다. wandb.init()을 호출합니다...")
                try:
                    # nohup 등 비-TTY 환경에서도 정상 동작하도록 설정
                    init_kwargs = {}
                    # WandB 모드는 환경 변수에서 읽거나 기본값 사용
                    # API_KEY가 없으면 자동으로 offline 모드
                    wandb_api_key = os.getenv("WANDB_API_KEY")
                    wandb_mode = os.environ.get("WANDB_MODE", "offline" if not wandb_api_key else "offline")
                    init_kwargs["mode"] = wandb_mode
                    
                    # TTY가 없을 때도 정상 동작하도록 설정
                    if not sys.stdout.isatty():
                        # 로컬 저장 위치 명시 (nohup으로 실행할 때 경로 문제 방지)
                        if "WANDB_DIR" not in os.environ:
                            init_kwargs["dir"] = os.getcwd()
                    
                    wandb.init(**init_kwargs)
                    # WandB 로컬 저장 위치 확인 및 로깅
                    if wandb.run:
                        wandb_dir = wandb.run.dir if hasattr(wandb.run, 'dir') else None
                        if wandb_dir:
                            logger.info(f"📁 WandB 로컬 저장 위치: {wandb_dir}")
                            logger.info(f"   나중에 wandb sync로 업로드: wandb sync {wandb_dir}")
                        else:
                            # 기본 위치는 현재 작업 디렉토리의 wandb/ 폴더
                            default_wandb_dir = os.path.join(os.getcwd(), "wandb")
                            logger.info(f"📁 WandB 로컬 저장 위치 (기본): {default_wandb_dir}")
                            logger.info(f"   나중에 wandb sync로 업로드: wandb sync {default_wandb_dir}")
                except Exception as init_error:
                    logger.warning(f"⚠️  wandb.init() 실패: {init_error}")
                    logger.warning("   WandB 없이 학습을 계속 진행합니다. (로컬에만 저장됨)")
                    logger.warning("   나중에 wandb sync로 업로드할 수 있습니다.")
            
            if wandb.run:
                logger.info(f"   sweep_id: {getattr(wandb.run, 'sweep_id', None)}")
                logger.info(f"   run_id: {getattr(wandb.run, 'id', None)}")
                
                # wandb.config 안전하게 접근
                try:
                    config_dict = dict(wandb.config) if wandb.config else {}
                    config_count = len(config_dict)
                    logger.info(f"   wandb.config 파라미터 수: {config_count}")
                    if config_count > 0:
                        logger.info(f"   wandb.config 샘플 (처음 10개):")
                        for i, (key, value) in enumerate(list(config_dict.items())[:10]):
                            logger.info(f"     {key} = {value}")
                    else:
                        logger.warning("⚠️  wandb.config가 비어있습니다! Sweep 파라미터가 전달되지 않았을 수 있습니다.")
                except Exception as e:
                    error_msg = str(e).lower()
                    if "404" in error_msg or "not found" in error_msg:
                        logger.warning(f"⚠️  WandB API 에러 (404): {e}")
                        logger.warning("   'agent not found' 에러는 WandB sweep이 존재하지 않거나 접근할 수 없을 때 발생합니다.")
                        logger.warning("   학습은 계속 진행되며, 로컬에만 저장됩니다.")
                        logger.warning("   나중에 wandb sync로 업로드하거나, 새로운 sweep을 생성하세요.")
                    else:
                        logger.warning(f"⚠️  wandb.config 접근 실패: {e}")
                        logger.warning("   하지만 학습은 계속 진행됩니다.")
            logger.info("=" * 80)
            logger.info("")
        except Exception as e:
            logger.error(f"WandB 상태 확인 실패: {e}", exc_info=True)
        
        # train.py의 main()을 호출하여 WandB Sweep 모드로 실행
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["train"]
            from lex_dpr.cli import train as train_module
            train_module.main()
        except RuntimeError as e:
            # CUDA OOM 에러 처리 (RuntimeError와 torch.cuda.OutOfMemoryError 모두 처리)
            import torch  # except 절 내에서 import
            error_msg = str(e).lower()
            # torch.cuda.OutOfMemoryError는 RuntimeError의 서브클래스이므로 isinstance로 확인
            is_oom = (
                "out of memory" in error_msg or 
                "cuda" in error_msg or 
                (torch is not None and hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(e, torch.cuda.OutOfMemoryError))
            )
            
            if is_oom:
                import wandb
                # torch는 이미 위에서 import됨
                
                logger.error("=" * 80)
                logger.error("❌ CUDA Out of Memory (OOM) 발생!")
                logger.error(f"   에러 메시지: {e}")
                logger.error("=" * 80)
                logger.error("")
                
                # CUDA 메모리 정리 시도
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("CUDA 캐시 정리 완료")
                except Exception:
                    pass
                
                logger.error("💡 해결 방법:")
                logger.error("   1. 배치 크기를 줄이세요 (data.batches.bi)")
                logger.error("   2. 시퀀스 길이를 줄이세요 (model.max_len)")
                logger.error("   3. Gradient accumulation steps를 늘리세요")
                logger.error("   4. 더 작은 모델을 사용하세요")
                logger.error("")
                
                # WandB에 실패 원인 로깅 및 태그 추가
                if wandb.run:
                    try:
                        # 태그 추가 (WandB 대시보드에서 필터링 가능)
                        # wandb.run.tags는 tuple일 수 있으므로 리스트로 변환
                        if not hasattr(wandb.run, 'tags') or wandb.run.tags is None:
                            wandb.run.tags = []
                        elif isinstance(wandb.run.tags, tuple):
                            wandb.run.tags = list(wandb.run.tags)
                        
                        if "OOM" not in wandb.run.tags:
                            wandb.run.tags.append("OOM")
                        if "failed" not in wandb.run.tags:
                            wandb.run.tags.append("failed")
                        
                        # Summary에 실패 정보 추가
                        wandb.run.summary["status"] = "failed"
                        wandb.run.summary["failure_reason"] = "OOM"
                        wandb.run.summary["failure_type"] = "memory"  # 메모리 관련 실패
                        wandb.run.summary["error_message"] = str(e)[:500]  # 메시지 길이 제한
                        
                        # 현재 설정 정보도 기록 (OOM 원인 분석용)
                        try:
                            config_dict = dict(wandb.config) if wandb.config else {}
                            if "data.batches.bi" in config_dict:
                                wandb.run.summary["batch_size"] = config_dict["data.batches.bi"]
                            if "model.max_len" in config_dict:
                                wandb.run.summary["max_len"] = config_dict["model.max_len"]
                            if "model.bi_model" in config_dict:
                                wandb.run.summary["model"] = config_dict["model.bi_model"]
                        except Exception:
                            pass
                        
                        wandb.finish(exit_code=1)
                    except Exception as e_inner:
                        logger.warning(f"WandB 로깅 실패: {e_inner}")
                
                # 예외를 다시 발생시켜 WandB가 실패로 기록하도록 함
                raise
            else:
                # 다른 RuntimeError는 그대로 전파
                raise
        finally:
            sys.argv = original_argv
            # 각 run 사이에 메모리 정리 (time_window 내에서만)
            # time_window 밖에서는 다른 프로세스(VLLM 등)가 GPU를 사용 중일 수 있으므로 정리하지 않음
            try:
                # time_window 체크
                should_cleanup = True
                if time_window:
                    in_window, _ = _check_time_window(time_window, timezone)
                    if not in_window:
                        logger.debug("Time window 밖이므로 GPU 메모리 정리 건너뜀")
                        should_cleanup = False
                
                # time_window 내에서만 메모리 정리
                if should_cleanup:
                    try:
                        import torch
                        import gc
                        if torch is not None and torch.cuda.is_available():
                            # 모든 GPU에서 강력한 메모리 정리
                            # 먼저 Python GC로 객체 정리 (모델 참조 제거)
                            gc.collect()
                            gc.collect()  # 추가 GC (순환 참조 정리)
                            
                            # 그 다음 PyTorch 캐시 정리
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            # 추가 정리: 모든 GPU 디바이스 확인
                            for i in range(torch.cuda.device_count()):
                                with torch.cuda.device(i):
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()  # IPC 메모리 정리
                            
                            # 최종 GC (정리 후 남은 객체)
                            gc.collect()
                            logger.debug("Run 종료 후 GPU 메모리 정리 완료 (time_window 내)")
                    except ImportError:
                        pass  # torch가 설치되지 않은 경우 무시
            except Exception as e:
                logger.debug(f"메모리 정리 중 오류 (무시됨): {e}")
                pass
    
    # WandB 에이전트 실행
    # count가 지정된 경우 전체 실행 횟수를 추적
    run_count = 0
    try:
        while True:
            if time_window:
                in_window, next_start_time = _check_time_window(time_window, timezone)
                if not in_window:
                    # time_window 밖에서는 GPU 메모리 정리하지 않음 (다른 프로세스가 사용 중일 수 있음)
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
            
            # wandb.agent()에 프로젝트와 엔티티 정보 전달
            # WandB는 sweep_id만으로도 작동하지만, project와 entity를 명시하면 더 정확함
            # sweep_id가 entity/project/sweep_id 형식인지 확인
            agent_kwargs = {}
            
            # sweep_id 형식 확인 및 파싱
            if "/" in sweep_id:
                # entity/project/sweep_id 형식
                parts = sweep_id.split("/")
                if len(parts) == 3:
                    # entity/project/sweep_id 형식이면 그대로 사용
                    actual_sweep_id = sweep_id
                    logger.info(f"📋 Sweep ID 형식: entity/project/sweep_id")
                elif len(parts) == 2:
                    # project/sweep_id 형식
                    actual_sweep_id = sweep_id
                    logger.info(f"📋 Sweep ID 형식: project/sweep_id")
                else:
                    actual_sweep_id = sweep_id
                    logger.warning(f"⚠️  예상치 못한 sweep_id 형식: {sweep_id}")
            else:
                # 단순 sweep_id 형식 - project와 entity를 명시적으로 전달
                actual_sweep_id = sweep_id
                if wandb_project:
                    agent_kwargs["project"] = wandb_project
                # entity는 명시적으로 설정된 경우에만 전달
                # None이면 전달하지 않아 WandB가 자동으로 현재 사용자 엔티티를 사용하도록 함
                if wandb_entity:
                    agent_kwargs["entity"] = wandb_entity
                logger.info(f"📋 Sweep ID 형식: 단순 ID (project/entity 별도 전달)")
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("🚀 WandB Agent 실행 준비")
            logger.info("=" * 80)
            logger.info(f"  sweep_id: {actual_sweep_id}")
            logger.info(f"  project: {wandb_project or '(sweep_id에 포함됨)'}")
            logger.info(f"  entity: {wandb_entity or '(sweep_id에 포함됨 또는 자동 감지)'}")
            if agent_kwargs:
                logger.info(f"  추가 인자: {agent_kwargs}")
            logger.info("=" * 80)
            logger.info("")
            
            # CUDA 메모리 할당 최적화 환경 변수 설정
            try:
                import torch
                if torch.cuda.is_available():
                    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                        logger.info("CUDA 메모리 할당 최적화 활성화: expandable_segments:True")
            except ImportError:
                pass  # torch가 설치되지 않은 경우 무시
            
            # wandb.agent() 호출 전 환경 변수 설정 (nohup 등 비-TTY 환경 대응)
            # nohup으로 실행할 때 TTY가 없어서 wandb가 동작을 변경할 수 있음
            # Python 출력 버퍼링 비활성화 (로그가 즉시 출력되도록)
            if "PYTHONUNBUFFERED" not in os.environ:
                os.environ["PYTHONUNBUFFERED"] = "1"
            # WandB가 TTY 없이도 정상 동작하도록 설정
            if "WANDB_CONSOLE" not in os.environ:
                os.environ["WANDB_CONSOLE"] = "wrap"  # TTY 없을 때도 로그 출력
            # WandB 모드 설정
            # API_KEY가 없으면 자동으로 offline 모드로 설정
            # 환경 변수가 이미 설정되어 있으면 그대로 사용
            if "WANDB_MODE" not in os.environ:
                # WANDB_API_KEY 확인
                wandb_api_key = os.getenv("WANDB_API_KEY")
                if not wandb_api_key:
                    # API_KEY가 없으면 자동으로 offline 모드
                    os.environ["WANDB_MODE"] = "offline"
                    logger.info("💾 WandB API_KEY가 없습니다. offline 모드로 실행합니다 (로컬에만 저장)")
                    logger.info("   나중에 업로드하려면: wandb sync <run-directory>")
                else:
                    # API_KEY가 있으면 기본값으로 offline 모드 (사용자가 명시적으로 online 설정 가능)
                    os.environ["WANDB_MODE"] = "offline"
                    logger.info("💾 WandB 모드: offline (로컬에만 저장, 기본값)")
                    logger.info("   온라인 모드로 전환하려면: set WANDB_MODE=online")
                    logger.info("   나중에 업로드하려면: wandb sync <run-directory>")
            else:
                wandb_mode = os.environ.get("WANDB_MODE", "offline")
                logger.info(f"💾 WandB 모드: {wandb_mode} (환경 변수에서 읽음)")
            
            # wandb.agent() 호출 전 sweep 존재 여부 확인 (404 에러 진단)
            # 스윕 생성 직후에는 WandB 서버에 완전히 반영되기까지 시간이 걸릴 수 있으므로
            # 재시도 로직 포함
            sweep_verified = False
            max_retries = 3
            retry_delay = 2  # 초
            
            for retry in range(max_retries):
                try:
                    api = wandb.Api()
                    # sweep_id 형식에 따라 다르게 처리
                    if "/" in sweep_id:
                        # entity/project/sweep_id 형식
                        sweep_path = sweep_id
                    else:
                        # project/sweep_id 형식으로 변환
                        if wandb_entity:
                            sweep_path = f"{wandb_entity}/{wandb_project}/{sweep_id}"
                        else:
                            sweep_path = f"{wandb_project}/{sweep_id}"
                    
                    if retry > 0:
                        logger.info(f"🔍 Sweep 확인 재시도 중 ({retry}/{max_retries-1}): {sweep_path}")
                    else:
                        logger.info(f"🔍 Sweep 확인 중: {sweep_path}")
                    
                    try:
                        sweep_obj = api.sweep(sweep_path)
                        logger.info(f"✅ Sweep 확인 완료: {sweep_obj.name or sweep_id}")
                        logger.info(f"   상태: {getattr(sweep_obj, 'state', 'unknown')}")
                        sweep_verified = True
                        break  # 성공하면 루프 종료
                    except Exception as check_error:
                        error_msg = str(check_error).lower()
                        if retry < max_retries - 1:
                            # 마지막 재시도가 아니면 잠시 대기 후 재시도
                            if "404" in error_msg or "not found" in error_msg:
                                logger.debug(f"   Sweep 아직 준비 중... ({retry_delay}초 후 재시도)")
                                time.sleep(retry_delay)
                                continue
                        # 마지막 재시도이거나 404가 아닌 에러면 에러 처리
                        error_str = str(check_error)
                        
                        logger.warning("=" * 80)
                        logger.warning("⚠️  Sweep 확인 실패")
                        logger.warning("=" * 80)
                        
                        if "404" in error_msg or "not found" in error_msg:
                            logger.warning(f"   에러 타입: 404 (리소스를 찾을 수 없음)")
                            logger.warning(f"   Sweep 경로: {sweep_path}")
                            logger.warning("")
                            logger.warning("   가능한 원인:")
                            logger.warning("   1. Sweep이 삭제되었거나 존재하지 않음")
                            logger.warning("   2. Sweep ID가 잘못됨")
                            logger.warning("   3. Project 또는 Entity가 일치하지 않음")
                            logger.warning("   4. WandB 접근 권한 문제 (다른 사용자의 private sweep)")
                            logger.warning("")
                            logger.warning("   확인 방법:")
                            logger.warning(f"   1. WandB 대시보드 확인: https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}/sweeps")
                            logger.warning("   2. wandb login 상태 확인: wandb login")
                            logger.warning("   3. 설정 파일의 sweep_id, project, entity 확인")
                        elif "401" in error_msg or "unauthorized" in error_msg or "permission" in error_msg:
                            logger.warning(f"   에러 타입: 401 (인증/권한 문제)")
                            logger.warning("   가능한 원인: WandB 로그인 만료 또는 접근 권한 없음")
                            logger.warning("   해결 방법: wandb login 재실행")
                        elif "500" in error_msg or "503" in error_msg or "server" in error_msg:
                            logger.warning(f"   에러 타입: 서버 에러 ({error_str[:100]})")
                            logger.warning("   가능한 원인: WandB 서버 일시적 문제")
                            logger.warning("   해결 방법: 잠시 후 재시도 또는 WandB 상태 확인: https://status.wandb.ai")
                        else:
                            logger.warning(f"   에러 타입: 기타 ({error_str[:100]})")
                        
                        logger.warning("")
                        logger.warning("💡 참고: 학습은 계속 진행되며, 로컬에만 저장됩니다.")
                        logger.warning("   나중에 wandb sync로 업로드할 수 있습니다.")
                        logger.warning("=" * 80)
                        logger.warning("")
                except Exception as api_error:
                    logger.debug(f"WandB API 확인 중 에러 (무시하고 계속 진행): {api_error}")
            
            # 로컬 모드 확인 (sweep_id가 local-으로 시작하면 로컬 모드)
            is_local_mode = actual_sweep_id.startswith("local-")
            
            if is_local_mode:
                # 로컬 모드: 설정 파일에서 파라미터 읽어서 로컬 루프로 실행
                logger.info("💾 로컬 모드로 실행합니다 (네트워크 연결 없음)")
                logger.info("")
                
                # 설정 파일 경로 확인
                if not config_path:
                    config_path = Path("configs/sweep.yaml")
                
                if config_path and config_path.exists():
                    sweep_config = OmegaConf.load(config_path)
                    parameters = _convert_to_dict(sweep_config.get("parameters", {}))
                    fixed_params = _convert_to_dict(sweep_config.get("fixed", {}))
                    
                    # fixed 파라미터를 parameters에 추가
                    for key, value in fixed_params.items():
                        if key not in parameters:
                            parameters[key] = {"value": _convert_to_dict(value)}
                    
                    metric_cfg = _convert_to_dict(sweep_config.get("metric", {}))
                    metric_name = metric_cfg.get("name", "eval/recall_at_10")
                    goal = str(metric_cfg.get("goal", "maximize")).lower()
                    param_space = _build_param_space(parameters)
                    fixed_params = {
                        k: _convert_to_dict(v.get("value"))
                        for k, v in parameters.items()
                        if isinstance(v, dict) and "value" in v
                    }
                    n_initial = 5
                    history = []

                    logger.info(f"로컬 파라미터 샘플링 시작 (파라미터 수: {len(parameters)}, Bayes n_initial={n_initial})")
                    
                    # 로컬 루프로 실행
                    agent_count = count if count else 999999  # count가 없으면 무제한
                    run_count = 0
                    
                    def run_one_and_return_metric(full_params: dict, metric_key: str):
                        # 로컬 모드에서는 WANDB_API_KEY를 무시하고 항상 offline
                        init_kwargs = {"mode": "offline", "config": full_params, "project": wandb_project}
                        if wandb_entity:
                            init_kwargs["entity"] = wandb_entity
                        wandb.init(**init_kwargs)
                        train_fn()
                        metric_val = None
                        if wandb.run and getattr(wandb.run, "summary", None):
                            try:
                                s = dict(wandb.run.summary)
                                metric_val = s.get(metric_key) or s.get(metric_key.replace("_at_", "@"))
                            except Exception:
                                pass
                        wandb.finish()
                        return metric_val
                    
                    while run_count < agent_count:
                        # time_window 체크
                        if time_window:
                            in_window, next_start_time = _check_time_window(time_window, timezone)
                            if not in_window:
                                logger.info(f"⏸️  시간 제한 밖입니다. {next_start_time}까지 대기합니다...")
                                if next_start_time:
                                    wait_seconds = (next_start_time - datetime.now()).total_seconds()
                                    if wait_seconds > 0:
                                        time.sleep(min(wait_seconds, 3600))  # 최대 1시간 대기
                                continue
                        
                        # 파라미터 샘플링: 초반 n_initial은 random, 이후 Bayes(EI) 추천
                        suggested = _suggest_next_bayes(
                            parameters, param_space, history, goal,
                            n_initial=n_initial, n_candidates=200,
                        )
                        full_params = {**fixed_params, **suggested}
                        
                        try:
                            metric_val = run_one_and_return_metric(full_params, metric_name)
                            if metric_val is not None:
                                history.append((full_params, float(metric_val)))
                            else:
                                bad = float("-inf") if goal == "maximize" else float("inf")
                                history.append((full_params, bad))
                            run_count += 1
                        except KeyboardInterrupt:
                            logger.info("사용자에 의해 중단되었습니다.")
                            break
                        except Exception as e:
                            logger.error(f"Run 실패: {e}")
                            continue
                    
                    logger.info(f"로컬 모드 실행 완료 (총 {run_count} runs)")
                else:
                    logger.error(f"로컬 모드: 설정 파일을 찾을 수 없습니다: {config_path}")
                    raise typer.Exit(1)
            else:
                # 일반 모드: wandb.agent() 호출
                # time_window가 설정된 경우, 각 run 시작 전에 시간을 체크하기 위해 count=1로 설정
                # 이렇게 하면 각 run이 끝난 후 time_window를 체크할 수 있음
                agent_count = 1 if time_window else count  # time_window가 있으면 한 번에 하나씩만 실행
                
                try:
                    logger.info(f"🔄 wandb.agent() 호출 중...")
                    logger.info(f"   sweep_id: {actual_sweep_id}")
                    logger.info(f"   count: {agent_count}")
                    logger.info(f"   kwargs: {agent_kwargs}")
                    wandb.agent(actual_sweep_id, function=train_fn, count=agent_count, **agent_kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                is_404 = "404" in error_msg or "not found" in error_msg or "agent not found" in error_msg
                
                logger.error("=" * 80)
                logger.error(f"❌ WandB Agent 실행 실패: {e}")
                logger.error("=" * 80)
                logger.error("")
                
                if is_404:
                    logger.error("⚠️  'agent not found (404)' 에러가 발생했습니다.")
                    logger.error("")
                    logger.error("가능한 원인:")
                    logger.error("  1. Sweep이 삭제되었거나 존재하지 않음")
                    logger.error("  2. Sweep ID가 잘못됨")
                    logger.error("  3. Entity 또는 Project가 일치하지 않음")
                    logger.error("  4. WandB API 접근 권한 문제")
                    logger.error("")
                    logger.error("현재 설정:")
                    logger.error(f"  - Sweep ID: {sweep_id}")
                    logger.error(f"  - Project: {wandb_project}")
                    logger.error(f"  - Entity: {wandb_entity or '(자동 감지)'}")
                    logger.error("")
                    logger.error("해결 방법:")
                    logger.error("  1. WandB 대시보드에서 sweep이 존재하는지 확인:")
                    logger.error(f"     https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}/sweeps")
                    logger.error("  2. 설정 파일의 sweep_id가 올바른지 확인")
                    logger.error("  3. WandB 로그인 상태 확인: wandb login")
                    logger.error("  4. Entity가 올바른지 확인 (대시보드 URL에서 확인)")
                    logger.error("")
                    logger.error("💡 참고: 학습 중 WandB API 에러가 발생해도 학습은 계속 진행됩니다.")
                    logger.error("   로컬에 결과가 저장되며, 나중에 wandb sync로 업로드할 수 있습니다.")
                else:
                    logger.error("가능한 원인:")
                    logger.error("  1. 네트워크 연결 문제")
                    logger.error("  2. WandB API 서버 문제")
                    logger.error("  3. 인증 문제 (wandb login 확인)")
                    logger.error("")
                    logger.error("해결 방법:")
                    logger.error("  1. 네트워크 연결 확인")
                    logger.error("  2. WandB 상태 확인: https://status.wandb.ai")
                    logger.error("  3. wandb login 재실행")
                
                logger.error("")
                logger.error("=" * 80)
                # 404 에러는 학습을 계속 진행할 수 있도록 예외를 다시 발생시키지 않음
                # (학습 자체는 wandb 없이도 가능하므로)
                if not is_404:
                    raise  # 404가 아닌 다른 에러는 예외를 다시 발생
            
            # wandb.agent()가 완료된 후 시간 체크
            # time_window가 설정된 경우, 각 run이 끝난 후 time_window를 체크하여 다음 run 시작 여부 결정
            if time_window:
                in_window, next_start_time = _check_time_window(time_window, timezone)
                if not in_window:
                    # run이 끝난 후 time_window 밖이면 GPU 메모리 정리 후 대기
                    try:
                        import torch
                        import gc
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            # 모든 GPU 디바이스에서 메모리 정리
                            for i in range(torch.cuda.device_count()):
                                with torch.cuda.device(i):
                                    torch.cuda.empty_cache()
                            logger.debug("Run 종료 후 time window 밖에서 GPU 메모리 정리 완료")
                        gc.collect()
                    except Exception:
                        pass
                    
                    if next_start_time:
                        import pytz
                        tz = pytz.timezone(timezone)
                        wait_seconds = (next_start_time - datetime.now(tz)).total_seconds()
                        logger.info(f"현재 시간은 스윕 실행 시간 범위({time_window[0]}-{time_window[1]}시) 밖입니다.")
                        logger.info(f"다음 시작 시간까지 대기합니다: {next_start_time.strftime('%Y-%m-%d %H:%M:%S')} ({int(wait_seconds // 60)}분 후)")
                        time.sleep(wait_seconds + 5)  # 5초 여유
                        continue  # 다시 시간 체크 후 계속
                    else:
                        # next_start_time이 None이면 시간 제한 없이 계속 실행
                        logger.warning("다음 시작 시간을 계산할 수 없습니다. 계속 실행합니다.")
            
            # count가 지정된 경우 처리
            if count is not None:
                run_count += 1
                if run_count >= count:
                    logger.info(f"지정된 실행 횟수({count})에 도달했습니다. 종료합니다.")
                    break
            
            # count가 None이고 time_window도 없는 경우, 한 번만 실행하고 종료
            if count is None and not time_window:
                break  # count가 없고 시간 제한도 없으면 한 번만 실행하고 종료 (기존 동작 유지)
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
    
    # 설정 파일 경로 설정
    if config is None:
        config = "configs/sweep.yaml"
    config_path = Path(config)
    
    # 설정 파일 로드 (project, entity 등을 읽기 위해 항상 로드)
    sweep_config = None
    if config_path.exists():
        sweep_config = OmegaConf.load(config_path)
    
    # sweep_id가 없으면 설정 파일에서 읽기
    if sweep_id is None:
        if not config_path.exists():
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            logger.error("먼저 'poetry run lex-dpr sweep preset'으로 스윕 설정 파일을 생성하세요.")
            raise typer.Exit(1)
        
        sweep_id = sweep_config.get("sweep_id")
        if sweep_id is None:
            logger.error(f"설정 파일에 sweep_id가 없습니다: {config_path}")
            logger.error("먼저 'poetry run lex-dpr sweep --config {config_path}' 또는 'poetry run lex-dpr sweep preset'으로 스윕을 생성하세요.")
            raise typer.Exit(1)
        
        logger.info(f"설정 파일에서 sweep_id를 읽었습니다: {sweep_id}")
    
    # 설정 파일에서 time_window와 timezone 읽기 (설정 파일 우선)
    if sweep_config:
        time_window_config = sweep_config.get("time_window")
        if time_window_config and time_window is None:
            if isinstance(time_window_config, str):
                time_window = time_window_config
            elif isinstance(time_window_config, (list, tuple)) and len(time_window_config) == 2:
                time_window = f"{time_window_config[0]}-{time_window_config[1]}"
        
        timezone_config = sweep_config.get("timezone")
        if timezone_config:
            timezone = timezone_config
    
    logger.info("=" * 80)
    logger.info("🔍 WandB Sweep 에이전트 시작")
    logger.info(f"스윕 ID: {sweep_id}")
    if count:
        logger.info(f"실행 횟수: {count}")
    if time_window:
        logger.info(f"실행 시간 범위: {time_window} ({timezone})")
    logger.info("=" * 80)
    logger.info("")
    
    # 네트워크 연결 및 WandB API 접근 가능 여부 확인
    logger.info("🌐 WandB 네트워크 연결 확인 중...")
    network_ok = False
    try:
        import wandb
        import requests
        import socket
        
        # 1. 기본 인터넷 연결 확인
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            logger.info("   ✅ 기본 인터넷 연결: 정상")
        except (socket.error, OSError):
            logger.warning("   ⚠️  기본 인터넷 연결: 실패 (DNS 서버에 접근 불가)")
        
        # 2. WandB API 서버 연결 확인
        try:
            response = requests.get("https://api.wandb.ai", timeout=5)
            if response.status_code == 200:
                logger.info("   ✅ WandB API 서버 연결: 정상")
                network_ok = True
            else:
                logger.warning(f"   ⚠️  WandB API 서버 응답: {response.status_code}")
        except requests.exceptions.Timeout:
            logger.warning("   ⚠️  WandB API 서버 연결: 타임아웃 (네트워크 지연 또는 방화벽)")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"   ⚠️  WandB API 서버 연결: 실패 ({e})")
            logger.warning("      가능한 원인: 방화벽, 프록시 설정, 네트워크 문제")
        except Exception as e:
            logger.warning(f"   ⚠️  WandB API 서버 연결 확인 실패: {e}")
        
        # 3. DNS 확인
        try:
            import socket
            socket.gethostbyname("api.wandb.ai")
            logger.info("   ✅ DNS 해석: 정상")
        except socket.gaierror:
            logger.warning("   ⚠️  DNS 해석: 실패 (api.wandb.ai를 찾을 수 없음)")
        
        logger.info("")
    except ImportError:
        logger.warning("   ⚠️  네트워크 확인을 위해 requests 라이브러리가 필요합니다.")
        logger.info("")
    
    # Sweep 존재 여부 확인 (선택사항, 에러 발생 시 무시)
    try:
        import wandb
        api = wandb.Api()
        try:
            sweep_obj = api.sweep(f"{wandb_entity or ''}/{wandb_project}/{sweep_id}".lstrip('/'))
            logger.info(f"✅ Sweep 확인 완료: {sweep_obj.name or sweep_id}")
            logger.info(f"   상태: {getattr(sweep_obj, 'state', 'unknown')}")
            logger.info("")
        except Exception as check_error:
            error_msg = str(check_error).lower()
            if "404" in error_msg or "not found" in error_msg:
                logger.warning("⚠️  Sweep을 찾을 수 없습니다 (404).")
                logger.warning(f"   Sweep ID: {sweep_id}")
                logger.warning(f"   Project: {wandb_project}")
                logger.warning(f"   Entity: {wandb_entity or '(자동 감지)'}")
                logger.warning("")
                logger.warning("가능한 원인:")
                logger.warning("  1. Sweep이 삭제되었거나 존재하지 않음")
                logger.warning("  2. Entity 또는 Project가 일치하지 않음")
                logger.warning("  3. WandB 접근 권한 문제")
                logger.warning("")
                logger.warning("해결 방법:")
                logger.warning(f"  1. WandB 대시보드 확인: https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}/sweeps")
                logger.warning("  2. 새로운 sweep 생성: poetry run lex-dpr sweep preset")
                logger.warning("  3. 설정 파일의 sweep_id 확인")
                logger.warning("")
                logger.warning("⚠️  계속 진행하면 'agent not found' 에러가 발생할 수 있습니다.")
                logger.warning("   하지만 학습은 계속 진행되며, 로컬에만 저장됩니다.")
                logger.warning("")
                logger.warning("💡 WandB 로컬 저장 위치:")
                import os
                default_wandb_dir = os.path.join(os.getcwd(), "wandb")
                logger.warning(f"   {default_wandb_dir}")
                logger.warning("   각 run은 wandb/run-{timestamp}-{run_id}/ 형태로 저장됩니다.")
                logger.warning("   나중에 wandb sync로 업로드: wandb sync <run-directory>")
                logger.warning("")
            else:
                # 네트워크 관련 에러인지 확인
                error_msg = str(check_error).lower()
                if "timeout" in error_msg or "connection" in error_msg or "network" in error_msg:
                    logger.warning(f"⚠️  네트워크 연결 문제로 보입니다: {check_error}")
                    logger.warning("   네트워크 상태 확인 방법:")
                    logger.warning("   1. ping api.wandb.ai")
                    logger.warning("   2. curl -I https://api.wandb.ai")
                    logger.warning("   3. wandb status (WandB 인증 상태 확인)")
                    logger.warning("")
                else:
                    logger.debug(f"Sweep 확인 중 에러 (무시됨): {check_error}")
    except Exception as e:
        logger.debug(f"Sweep 확인을 건너뜁니다: {e}")
    
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
    
    # 설정 파일에서 프로젝트 및 엔티티 정보 읽기 (이미 로드된 sweep_config 사용)
    wandb_project = "lexdpr"
    wandb_entity = None
    if sweep_config:
        wandb_project = sweep_config.get("project", "lexdpr")
        wandb_entity = sweep_config.get("entity", None)
    
    # _run_agent_impl 호출
    _run_agent_impl(
        sweep_id=sweep_id,
        count=count,
        time_window=time_window_tuple,
        timezone=timezone,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        config_path=config_path,
    )


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

