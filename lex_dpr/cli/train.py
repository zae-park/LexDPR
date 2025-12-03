"""
학습 엔트리포인트

사용 예시:
  poetry run lex-dpr train
  poetry run lex-dpr train trainer.epochs=5 trainer.lr=3e-5
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

# FutureWarning 억제 (선택사항)
warnings.filterwarnings("ignore", category=FutureWarning)

from omegaconf import OmegaConf

from lex_dpr.trainer.base_trainer import BiEncoderTrainer

# 로거 설정
logger = logging.getLogger("lex_dpr.train")


def _is_wandb_sweep_mode() -> bool:
    """WandB Sweep 모드인지 확인"""
    try:
        import wandb
        return wandb.run is not None and hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id is not None
    except (ImportError, AttributeError):
        return False


def _get_config_path(filename: str) -> Path:
    """설정 파일 경로 반환 (사용자 configs 우선, 없으면 패키지 기본값)"""
    user_configs_dir = Path.cwd() / "configs"
    user_path = user_configs_dir / filename
    
    if user_path.exists():
        return user_path
    
    # 패키지 내부 기본값 사용
    import lex_dpr.configs
    package_configs_dir = Path(lex_dpr.configs.__file__).parent
    return package_configs_dir / filename


def _log_config_summary(cfg):
    """주요 설정만 요약해서 로깅"""
    logger.info("=" * 80)
    logger.info("🚀 LexDPR 학습 시작")
    logger.info("=" * 80)
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("📋 주요 설정:")
    logger.info(f"  모드: {cfg.mode}")
    logger.info(f"  출력 디렉토리: {cfg.out_dir}")
    logger.info(f"  시드: {cfg.seed}")
    logger.info("")
    logger.info("🎓 학습 설정:")
    test_run = getattr(cfg, "test_run", False)
    effective_epochs = 1 if test_run else cfg.trainer.epochs
    logger.info(f"  에포크: {effective_epochs}" + (" (테스트 실행 모드)" if test_run else ""))
    logger.info(f"  학습률: {cfg.trainer.lr}")
    logger.info(f"  배치 크기: {cfg.data.batches.bi}")
    logger.info(f"  Gradient Accumulation Steps: {cfg.trainer.gradient_accumulation_steps}")
    logger.info(f"  AMP 사용: {cfg.trainer.use_amp}")
    logger.info(f"  평가 스텝: {cfg.trainer.eval_steps if cfg.trainer.eval_steps > 0 else '비활성화'}")
    
    # Gradient Clipping 상태
    gradient_clip_norm = float(getattr(cfg.trainer, "gradient_clip_norm", 0.0))
    if gradient_clip_norm > 0:
        logger.info(f"  Gradient Clipping: 활성화 (max_norm={gradient_clip_norm})")
    else:
        logger.info(f"  Gradient Clipping: 비활성화")
    
    # Early Stopping 상태
    early_stopping_config = getattr(cfg.trainer, "early_stopping", None)
    if early_stopping_config and getattr(early_stopping_config, "enabled", False):
        metric = getattr(early_stopping_config, "metric", "cosine_ndcg@10")
        patience = getattr(early_stopping_config, "patience", 3)
        logger.info(f"  Early Stopping: 활성화 (metric={metric}, patience={patience})")
    else:
        logger.info(f"  Early Stopping: 비활성화")
    
    if test_run:
        logger.info(f"  🧪 테스트 실행 모드: 활성화 (최대 100 iteration 또는 1 epoch)")
    logger.info("")
    logger.info("📊 데이터:")
    logger.info(f"  Passages: {cfg.data.passages}")
    logger.info(f"  Training Pairs: {cfg.data.pairs}")
    if hasattr(cfg.trainer, 'eval_pairs') and cfg.trainer.eval_pairs:
        logger.info(f"  Evaluation Pairs: {cfg.trainer.eval_pairs}")
    logger.info("")
    logger.info("🤖 모델:")
    logger.info(f"  Base Model: {cfg.model.bi_model}")
    logger.info(f"  BGE Template: {cfg.model.use_bge_template}")
    logger.info(f"  Max Length: {cfg.model.max_len}")
    if hasattr(cfg.model, 'peft') and cfg.model.peft.enabled:
        logger.info(f"  PEFT (LoRA): 활성화 (r={cfg.model.peft.r}, alpha={cfg.model.peft.alpha})")
    else:
        logger.info(f"  PEFT (LoRA): 비활성화")
    logger.info("")
    logger.info("💡 전체 설정을 보려면: poetry run lex-dpr config show")
    logger.info("=" * 80)
    logger.info("")


def main():
    """학습 메인 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    start_time = datetime.now()
    
    # 설정 파일 로드 (사용자 configs 우선, 없으면 패키지 기본값)
    base_path = _get_config_path("base.yaml")
    data_path = _get_config_path("data.yaml")
    model_path = _get_config_path("model.yaml")
    
    logger.info("설정 파일 로드 중...")
    logger.info(f"  - base.yaml: {base_path}")
    if data_path.exists():
        logger.info(f"  - data.yaml: {data_path}")
    if model_path.exists():
        logger.info(f"  - model.yaml: {model_path}")
    logger.info("")
    
    base = OmegaConf.load(base_path)
    
    # data.yaml 로드 및 병합 (원래 로직과 동일하게)
    if data_path.exists():
        data = OmegaConf.load(data_path)
        base = OmegaConf.merge(base, {"data": data})
    
    # model.yaml 로드 및 병합 (원래 로직과 동일하게)
    if model_path.exists():
        model = OmegaConf.load(model_path)
        base = OmegaConf.merge(base, {"model": model})
    
    cfg = base

    # 커맨드라인 인자로 오버라이드 (예: trainer.epochs=5)
    overrides = OmegaConf.from_dotlist(sys.argv[1:])
    if overrides:
        logger.info(f"커맨드라인 오버라이드 적용: {list(overrides.keys())}")
        logger.info("")
    cfg = OmegaConf.merge(cfg, overrides)

    # WandB Sweep 모드 확인
    is_sweep_mode = _is_wandb_sweep_mode()
    
    if is_sweep_mode:
        # SweepTrainer 사용 (wandb.config를 읽어서 cfg에 병합)
        from lex_dpr.trainer.sweep_trainer import SweepTrainer
        logger.info("🔍 WandB Sweep 모드로 실행합니다.")
        logger.info("")
        trainer_wrapper = SweepTrainer(cfg)
        trainer = trainer_wrapper.trainer
        cfg = trainer_wrapper.cfg  # SweepTrainer가 병합한 최종 설정 사용
    else:
        # 일반 BiEncoderTrainer 사용
        trainer = BiEncoderTrainer(cfg)

    # 설정 요약 로깅 (전체 출력 대신)
    _log_config_summary(cfg)
    
    # Trainer 초기화 완료
    logger.info("Trainer 초기화 완료")
    logger.info("")
    
    logger.info("학습 시작")
    logger.info("-" * 80)
    trainer.train()
    logger.info("-" * 80)
    
    # 학습 완료 로깅
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ 학습 완료!")
    logger.info(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"소요 시간: {duration}")
    logger.info(f"모델 저장 위치: {cfg.out_dir}/bi_encoder")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

