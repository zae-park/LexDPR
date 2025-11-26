"""
설정 관리 CLI

사용 예시:
  poetry run config init    # 기본 설정 파일을 configs/ 디렉토리에 생성
  poetry run config show    # 현재 설정된 config 출력
"""

import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


def _get_package_configs_dir() -> Path:
    """패키지 내부의 기본 configs 디렉토리 경로 반환"""
    import lex_dpr.configs
    
    # lex_dpr.configs 모듈의 경로에서 configs 디렉토리 찾기
    configs_module_path = Path(lex_dpr.configs.__file__).parent
    return configs_module_path


def _get_user_configs_dir() -> Path:
    """사용자 프로젝트의 configs 디렉토리 경로 반환"""
    # 현재 작업 디렉토리 기준
    return Path.cwd() / "configs"


def init_configs(force: bool = False) -> None:
    """기본 설정 파일을 configs/ 디렉토리에 초기화"""
    package_configs_dir = _get_package_configs_dir()
    user_configs_dir = _get_user_configs_dir()
    
    # configs 디렉토리 생성
    user_configs_dir.mkdir(parents=True, exist_ok=True)
    
    # 기본 설정 파일 목록
    config_files = ["base.yaml", "data.yaml", "model.yaml"]
    
    copied_files = []
    skipped_files = []
    
    for config_file in config_files:
        src = package_configs_dir / config_file
        dst = user_configs_dir / config_file
        
        if not src.exists():
            print(f"경고: 패키지 내부에 {config_file} 파일이 없습니다.")
            continue
        
        if dst.exists() and not force:
            skipped_files.append(config_file)
            continue
        
        # 파일 복사
        import shutil
        shutil.copy2(src, dst)
        copied_files.append(config_file)
        print(f"✓ {config_file} 생성 완료")
    
    if copied_files:
        print(f"\n기본 설정 파일이 {user_configs_dir}에 생성되었습니다.")
        print("필요에 따라 설정 파일을 수정하세요.")
    
    if skipped_files:
        print(f"\n다음 파일들은 이미 존재하여 건너뛰었습니다: {', '.join(skipped_files)}")
        print("강제로 덮어쓰려면 --force 플래그를 사용하세요.")


def show_config() -> None:
    """현재 설정된 config 출력"""
    user_configs_dir = _get_user_configs_dir()
    
    # 사용자 configs 디렉토리가 없으면 패키지 기본값 사용
    if not user_configs_dir.exists():
        print("⚠ 사용자 configs 디렉토리가 없습니다. 패키지 기본값을 사용합니다.")
        configs_dir = _get_package_configs_dir()
    else:
        configs_dir = user_configs_dir
    
    config_files = ["base.yaml", "data.yaml", "model.yaml"]
    
    # 모든 설정 파일 로드 및 병합
    base_path = configs_dir / "base.yaml"
    data_path = configs_dir / "data.yaml"
    model_path = configs_dir / "model.yaml"
    
    if not base_path.exists():
        print(f"❌ 오류: {base_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    base = OmegaConf.load(base_path)
    
    if data_path.exists():
        data = OmegaConf.load(data_path)
        # data.yaml의 내용이 data 키 아래에 있으면 병합
        if "data" in data:
            base = OmegaConf.merge(base, data)
        else:
            # data.yaml이 평면 구조면 data 섹션에 병합
            base = OmegaConf.merge(base, {"data": data})
    
    if model_path.exists():
        model = OmegaConf.load(model_path)
        # model.yaml의 내용이 model 키 아래에 있으면 병합
        if "model" in model:
            base = OmegaConf.merge(base, model)
        else:
            # model.yaml이 평면 구조면 model 섹션에 병합
            base = OmegaConf.merge(base, {"model": model})
    
    # 설정 출력
    print("=" * 80)
    print("현재 설정 (Current Configuration)")
    print("=" * 80)
    print(f"설정 파일 위치: {configs_dir}")
    print("=" * 80)
    print(OmegaConf.to_yaml(base))
    print("=" * 80)


def main():
    """Config CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LexDPR 설정 관리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  poetry run config init          # 기본 설정 파일 생성
  poetry run config init --force  # 기존 파일 덮어쓰기
  poetry run config show          # 현재 설정 출력
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="명령어")
    
    # init 명령어
    init_parser = subparsers.add_parser("init", help="기본 설정 파일 초기화")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="기존 파일이 있어도 덮어쓰기"
    )
    
    # show 명령어
    subparsers.add_parser("show", help="현재 설정 출력")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_configs(force=args.force)
    elif args.command == "show":
        show_config()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

