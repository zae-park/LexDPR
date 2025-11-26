"""
웹 로깅 서비스 통합 (Neptune, WandB, MLflow)

사용 예시:
  # config에서 설정
  web_logging:
    service: wandb  # neptune, wandb, mlflow
    token: ${NEPTUNE_API_TOKEN}  # 환경 변수 또는 직접 입력
    project: lexdpr
    name: experiment-001
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("lex_dpr.web_logging")


class WebLogger:
    """웹 로깅 서비스 통합 래퍼"""
    
    def __init__(self, service: str, token: Optional[str] = None, **kwargs):
        """
        Args:
            service: 'neptune', 'wandb', 'mlflow'
            token: API 토큰 (환경 변수에서 읽거나 직접 입력)
            **kwargs: 각 서비스별 추가 설정
        """
        self.service = service.lower()
        self.token = token
        self.logger_impl = None
        self.is_active = False
        
        if not token:
            logger.warning(f"{service} 토큰이 제공되지 않았습니다. 웹 로깅을 건너뜁니다.")
            return
        
        try:
            if self.service == "neptune":
                self._init_neptune(token, **kwargs)
            elif self.service == "wandb":
                self._init_wandb(token, **kwargs)
            elif self.service == "mlflow":
                self._init_mlflow(token, **kwargs)
            else:
                logger.warning(f"지원하지 않는 웹 로깅 서비스: {service}")
                return
            
            self.is_active = True
            logger.info(f"✅ {service.upper()} 웹 로깅 초기화 완료")
        except Exception as e:
            logger.warning(f"{service} 초기화 실패: {e}. 웹 로깅을 건너뜁니다.")
            self.is_active = False
    
    def _init_neptune(self, token: str, project: str, name: Optional[str] = None, **kwargs):
        """Neptune 초기화"""
        try:
            import neptune
        except ImportError:
            raise ImportError("neptune-client가 설치되지 않았습니다. 'poetry add neptune-client'로 설치하세요.")
        
        # 환경 변수에 토큰 설정
        os.environ["NEPTUNE_API_TOKEN"] = token
        
        run = neptune.init_run(
            project=project,
            name=name,
            **kwargs
        )
        self.logger_impl = run
        logger.info(f"Neptune 프로젝트: {project}, 실행 이름: {name}")
    
    def _init_wandb(self, token: str, project: str, name: Optional[str] = None, **kwargs):
        """WandB 초기화"""
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb가 설치되지 않았습니다. 'poetry add wandb'로 설치하세요.")
        
        # WandB 로그인
        wandb.login(key=token)
        
        run = wandb.init(
            project=project,
            name=name,
            **kwargs
        )
        self.logger_impl = run
        logger.info(f"WandB 프로젝트: {project}, 실행 이름: {name}")
    
    def _init_mlflow(self, token: str, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None, run_name: Optional[str] = None, **kwargs):
        """MLflow 초기화"""
        try:
            import mlflow
        except ImportError:
            raise ImportError("mlflow가 설치되지 않았습니다. 'poetry add mlflow'로 설치하세요.")
        
        # MLflow는 토큰을 직접 사용하지 않고, tracking_uri에 포함하거나 환경 변수로 설정
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name:
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except Exception:
                # 이미 존재하는 경우
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id if experiment else None
            
            if experiment_id:
                mlflow.set_experiment(experiment_name)
        
        run = mlflow.start_run(run_name=run_name, **kwargs)
        self.logger_impl = run
        logger.info(f"MLflow 추적 URI: {mlflow.get_tracking_uri()}, 실험: {experiment_name}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """하이퍼파라미터 로깅"""
        if not self.is_active:
            return
        
        try:
            if self.service == "neptune":
                self.logger_impl["parameters"] = params
            elif self.service == "wandb":
                self.logger_impl.config.update(params)
            elif self.service == "mlflow":
                import mlflow
                mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"파라미터 로깅 실패: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """메트릭 로깅"""
        if not self.is_active:
            return
        
        try:
            if self.service == "neptune":
                if step is not None:
                    for key, value in metrics.items():
                        self.logger_impl[f"metrics/{key}"].log(value, step=step)
                else:
                    for key, value in metrics.items():
                        self.logger_impl[f"metrics/{key}"].log(value)
            elif self.service == "wandb":
                if step is not None:
                    self.logger_impl.log(metrics, step=step)
                else:
                    self.logger_impl.log(metrics)
            elif self.service == "mlflow":
                import mlflow
                if step is not None:
                    mlflow.log_metrics(metrics, step=step)
                else:
                    mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"메트릭 로깅 실패: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """아티팩트(파일) 로깅"""
        if not self.is_active:
            return
        
        try:
            if self.service == "neptune":
                self.logger_impl[f"artifacts/{artifact_path or os.path.basename(local_path)}"].upload(local_path)
            elif self.service == "wandb":
                self.logger_impl.log_artifact(local_path, name=artifact_path)
            elif self.service == "mlflow":
                import mlflow
                mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"아티팩트 로깅 실패: {e}")
    
    def finish(self) -> None:
        """로깅 종료"""
        if not self.is_active:
            return
        
        try:
            if self.service == "neptune":
                self.logger_impl.stop()
            elif self.service == "wandb":
                self.logger_impl.finish()
            elif self.service == "mlflow":
                import mlflow
                mlflow.end_run()
            logger.info(f"{self.service.upper()} 로깅 종료")
        except Exception as e:
            logger.warning(f"로깅 종료 실패: {e}")


def _create_single_web_logger(web_logging_cfg) -> Optional[WebLogger]:
    """단일 웹 로깅 서비스 설정에서 WebLogger 생성"""
    service = getattr(web_logging_cfg, "service", None)
    if not service:
        return None
    
    # 토큰 가져오기 (환경 변수 또는 직접 입력)
    token = getattr(web_logging_cfg, "token", None)
    if token and isinstance(token, str) and token.startswith("${") and token.endswith("}"):
        # 환경 변수 참조 (예: ${NEPTUNE_API_TOKEN})
        env_var = token[2:-1]
        token = os.getenv(env_var)
        if not token:
            logger.warning(f"환경 변수 {env_var}를 찾을 수 없습니다.")
            return None
    
    if not token:
        logger.warning(f"{service} 토큰이 제공되지 않았습니다.")
        return None
    
    # 서비스별 설정 추출
    kwargs = {}
    if service == "neptune":
        kwargs["project"] = getattr(web_logging_cfg, "project", "lexdpr")
        kwargs["name"] = getattr(web_logging_cfg, "name", None)
    elif service == "wandb":
        kwargs["project"] = getattr(web_logging_cfg, "project", "lexdpr")
        kwargs["name"] = getattr(web_logging_cfg, "name", None)
        kwargs["entity"] = getattr(web_logging_cfg, "entity", None)
    elif service == "mlflow":
        kwargs["tracking_uri"] = getattr(web_logging_cfg, "tracking_uri", None)
        kwargs["experiment_name"] = getattr(web_logging_cfg, "experiment_name", "lexdpr")
        kwargs["run_name"] = getattr(web_logging_cfg, "run_name", None)
    
    return WebLogger(service=service, token=token, **kwargs)


class MultiWebLogger:
    """여러 웹 로깅 서비스를 동시에 관리하는 래퍼"""
    
    def __init__(self, loggers: list[WebLogger]):
        self.loggers = [lg for lg in loggers if lg and lg.is_active]
        self.is_active = len(self.loggers) > 0
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """모든 활성 웹 로거에 하이퍼파라미터 로깅"""
        for logger in self.loggers:
            logger.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """모든 활성 웹 로거에 메트릭 로깅"""
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """모든 활성 웹 로거에 아티팩트 로깅"""
        for logger in self.loggers:
            logger.log_artifact(local_path, artifact_path)
    
    def finish(self) -> None:
        """모든 활성 웹 로거 종료"""
        for logger in self.loggers:
            logger.finish()


def create_web_logger(cfg) -> Optional[MultiWebLogger]:
    """
    설정에서 웹 로거 생성 (단일 또는 다중 서비스 지원)
    
    Args:
        cfg: OmegaConf 설정 객체 (web_logging 섹션 포함)
    
    Returns:
        MultiWebLogger 인스턴스 (여러 서비스 래핑) 또는 None
    """
    web_logging_cfg = getattr(cfg, "web_logging", None)
    if not web_logging_cfg:
        return None
    
    # 리스트 형태인지 확인 (여러 서비스)
    if isinstance(web_logging_cfg, (list, tuple)):
        loggers = []
        for item in web_logging_cfg:
            logger_impl = _create_single_web_logger(item)
            if logger_impl:
                loggers.append(logger_impl)
        
        if not loggers:
            return None
        
        return MultiWebLogger(loggers)
    
    # 단일 서비스 (기존 방식)
    logger_impl = _create_single_web_logger(web_logging_cfg)
    if not logger_impl:
        return None
    
    return MultiWebLogger([logger_impl])

