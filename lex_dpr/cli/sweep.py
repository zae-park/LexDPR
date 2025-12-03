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
import warnings
from pathlib import Path
from typing import Optional

# train.py의 함수들을 import하여 재사용

# FutureWarning 억제
warnings.filterwarnings("ignore", category=FutureWarning)

import typer
from omegaconf import OmegaConf

from lex_dpr.cli.train import _get_config_path
from lex_dpr.trainer.sweep_trainer import SweepTrainer

logger = logging.getLogger("lex_dpr.cli.sweep")

app = typer.Typer(
    name="sweep",
    help="WandB Sweep을 통한 하이퍼파라미터 튜닝",
    add_completion=False,
)


def _get_sweep_template() -> str:
    """스윕 설정 템플릿 반환"""
    return """# WandB Sweep 설정 파일
# 이 파일은 WandB Sweep의 하이퍼파라미터 탐색 범위를 정의합니다.

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
    distribution: log_uniform  # log_uniform, uniform, categorical
    min: 1e-6
    max: 1e-3
  
  trainer.temperature:
    distribution: uniform
    min: 0.01
    max: 0.2
  
  trainer.gradient_accumulation_steps:
    values: [4, 8, 16]  # categorical (고정 값들 중 선택)
  
  # trainer.epochs:
  #   value: 3  # 고정 값 (스윕에서 변경하지 않음)

# 고정 파라미터 (스윕 설정 파일에 직접 정의)
# 이 값들은 모든 스윕 실행에서 동일하게 사용됩니다.
fixed:
  trainer.epochs: 3
  trainer.eval_steps: 300
  data.pairs: data/pairs_train.jsonl
  data.passages: data/merged_corpus.jsonl

# WandB 프로젝트 설정 (선택사항)
# project: lexdpr
# entity: your-wandb-entity
"""


@app.command("init")
def sweep_init(
    output: str = typer.Option(
        "configs/sweep.yaml",
        "--output",
        "-o",
        help="생성할 스윕 설정 파일 경로",
    ),
):
    """
    WandB Sweep 설정 파일 템플릿을 생성합니다.
    
    예시:
      poetry run lex-dpr sweep init --output configs/my_sweep.yaml
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    template = _get_sweep_template()
    
    if output_path.exists():
        logger.warning(f"파일이 이미 존재합니다: {output_path}")
        response = typer.prompt("덮어쓰시겠습니까? (y/N)", default="N")
        if response.lower() != "y":
            logger.info("취소되었습니다.")
            return
    
    output_path.write_text(template, encoding="utf-8")
    logger.info(f"✅ 스윕 설정 템플릿 생성 완료: {output_path}")
    logger.info("")
    logger.info("다음 단계:")
    logger.info("  1. 설정 파일을 편집하여 탐색할 파라미터 범위를 설정하세요")
    logger.info("  2. poetry run lex-dpr sweep start --config <파일경로> 로 스윕을 시작하세요")


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
):
    """
    WandB에 스윕을 생성하고 시작합니다.
    
    스윕 ID를 반환하며, 이 ID를 사용하여 에이전트를 실행할 수 있습니다.
    
    예시:
      poetry run lex-dpr sweep start --config configs/my_sweep.yaml
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
    
    # WandB 프로젝트 설정
    wandb_project = project or sweep_config.get("project", "lexdpr")
    wandb_entity = entity or sweep_config.get("entity", None)
    
    # 프로그램 경로 확인
    program = sweep_config.get("program", "lex_dpr/cli/train.py")
    if not Path(program).exists():
        logger.warning(f"프로그램 경로를 찾을 수 없습니다: {program}")
        logger.info("상대 경로로 시도합니다...")
    
    # 스윕 설정 딕셔너리 생성 (WandB 형식)
    sweep_dict = {
        "method": sweep_config.get("method", "random"),
        "metric": sweep_config.get("metric", {"name": "eval/ndcg@10", "goal": "maximize"}),
        "parameters": sweep_config.get("parameters", {}),
    }
    
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
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 스윕 생성 완료!")
    logger.info(f"스윕 ID: {sweep_id}")
    logger.info("")
    logger.info("다음 단계:")
    logger.info(f"  poetry run lex-dpr sweep agent {sweep_id}")
    logger.info("또는")
    logger.info(f"  wandb agent {wandb_project}/{sweep_id}")
    logger.info("")
    logger.info(f"WandB 대시보드: https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}/sweeps/{sweep_id}")
    logger.info("=" * 80)


@app.command("agent")
def sweep_agent(
    sweep_id: str = typer.Argument(..., help="WandB 스윕 ID"),
    count: Optional[int] = typer.Option(
        None,
        "--count",
        "-c",
        help="실행할 에이전트 실행 횟수 (None이면 무제한)",
    ),
):
    """
    WandB Sweep 에이전트를 실행합니다.
    
    스윕에서 제공하는 파라미터로 학습을 실행합니다.
    여러 머신에서 동시에 실행하여 병렬 탐색이 가능합니다.
    
    예시:
      poetry run lex-dpr sweep agent <sweep-id>
      poetry run lex-dpr sweep agent <sweep-id> --count 5
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
        wandb.agent(sweep_id, function=train_fn, count=count)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("에이전트가 중단되었습니다.")
    except Exception as e:
        logger.error(f"에이전트 실행 실패: {e}")
        raise


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

