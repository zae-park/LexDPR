#!/usr/bin/env python3
"""
기존 WandB run에 모델 artifact를 업로드하는 스크립트

사용법:
    poetry run python scripts/upload_artifact_to_run.py \
        --run-id abc123 \
        --model-path checkpoint/lexdpr/bi_encoder_best \
        --project lexdpr \
        --artifact-name model
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lex_dpr.utils.web_logging import upload_artifact_to_existing_run


def main():
    parser = argparse.ArgumentParser(
        description="기존 WandB run에 모델 artifact 업로드"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="WandB run ID (예: abc123 또는 entity/project/run_id)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="업로드할 모델 경로 (파일 또는 디렉토리)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="WandB 프로젝트 이름 (선택사항, run-id에 포함되어 있으면 생략 가능)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity 이름 (선택사항)"
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        default="model",
        help="아티팩트 이름 (기본값: model)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="WandB API 토큰 (기본값: WANDB_API_KEY 환경 변수에서 읽기)"
    )
    
    args = parser.parse_args()
    
    # 모델 경로 확인
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 오류: 모델 경로를 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("기존 WandB run에 모델 artifact 업로드")
    print("=" * 80)
    print()
    print(f"Run ID: {args.run_id}")
    print(f"모델 경로: {model_path}")
    print(f"프로젝트: {args.project or '(자동 감지)'}")
    print(f"Entity: {args.entity or '(자동 감지)'}")
    print(f"아티팩트 이름: {args.artifact_name}")
    print()
    
    # 토큰 가져오기
    token = args.token or os.getenv("WANDB_API_KEY")
    if not token:
        print("❌ 오류: WandB API 토큰이 필요합니다.")
        print("   --token 옵션으로 제공하거나 WANDB_API_KEY 환경 변수를 설정하세요.")
        sys.exit(1)
    
    # Artifact 업로드
    success = upload_artifact_to_existing_run(
        run_id=args.run_id,
        local_path=str(model_path),
        artifact_path=args.artifact_name,
        project=args.project,
        entity=args.entity,
        token=token,
    )
    
    if success:
        print()
        print("=" * 80)
        print("✅ 업로드 완료!")
        print("=" * 80)
        sys.exit(0)
    else:
        print()
        print("=" * 80)
        print("❌ 업로드 실패")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

