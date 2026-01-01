#!/usr/bin/env python
"""
임베딩 API 서버

사용 예시:
  poetry run python entrypoint_api.py \
    --model checkpoint/lexdpr/bi_encoder \
    --host 0.0.0.0 \
    --port 8000

API 엔드포인트:
  POST /embed/passage - Passage 임베딩
  POST /embed/query - Query 임베딩
  GET /health - 서버 상태 확인
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import List, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    print("Error: fastapi and pydantic are required. Install with: poetry add fastapi uvicorn")
    sys.exit(1)

from lex_dpr.models.encoders import BiEncoder
from lex_dpr.models.templates import TemplateMode


# 요청/응답 모델
class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: Optional[bool] = True
    batch_size: Optional[int] = 64


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# 전역 변수
encoder: Optional[BiEncoder] = None
app = FastAPI(title="LexDPR Embedding API", version="1.0.0", docs_url="/docs")


def load_encoder(
    model_path: str,
    template: str = "bge",
    max_len: Optional[int] = None,
    device: Optional[str] = None,
    peft_adapter: Optional[str] = None,
):
    """모델 로드 함수"""
    global encoder
    
    template_mode = TemplateMode.BGE if template == "bge" else TemplateMode.NONE
    max_seq_length = max_len if max_len and max_len > 0 else None
    
    print(f"[API] Loading model from {model_path}")
    encoder = BiEncoder(
        model_path,
        template=template_mode,
        normalize=True,
        max_seq_length=max_seq_length,
        peft_adapter_path=peft_adapter,
    )
    
    if device:
        encoder.model.to(device)
        print(f"[API] Using device: {device}")
    else:
        auto_device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder.model.to(auto_device)
        print(f"[API] Using device: {auto_device} (auto-detected)")
    
    print(f"[API] Model loaded successfully")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    return HealthResponse(
        status="healthy" if encoder is not None else "not_ready",
        model_loaded=encoder is not None
    )


@app.post("/embed/passage", response_model=EmbedResponse)
async def embed_passage(request: EmbedRequest):
    """Passage 임베딩 생성"""
    if encoder is None:
        logger.error("[embed/passage] Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        logger.error("[embed/passage] texts list cannot be empty")
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    
    try:
        # 임베딩 생성
        embeddings = encoder.encode_passages(
            request.texts,
            batch_size=request.batch_size or 64
        )
        
        # 정규화 (요청에서 지정한 경우)
        if not request.normalize:
            # 이미 normalize=True로 생성되었으므로, 다시 정규화하지 않음
            # 실제로는 BiEncoder의 normalize 설정을 사용
            pass
        
        # 리스트로 변환
        embeddings_list = embeddings.tolist()
        
        return EmbedResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape),
            count=len(embeddings_list)
        )
    except Exception as e:
        logger.error(f"[embed/passage] Embedding generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/embed/query", response_model=EmbedResponse)
async def embed_query(request: EmbedRequest):
    """Query 임베딩 생성"""
    if encoder is None:
        logger.error("[embed/query] Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        logger.error("[embed/query] texts list cannot be empty")
        raise HTTPException(status_code=400, detail="texts list cannot be empty")

    try:
        # 임베딩 생성
        embeddings = encoder.encode_queries(
            request.texts,
            batch_size=request.batch_size or 64
        )

        # 리스트로 변환
        embeddings_list = embeddings.tolist()

        return EmbedResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape),
            count=len(embeddings_list)
        )
    except Exception as e:
        logger.error(f"[embed/query] Embedding generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LexDPR Embedding API Server")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--template", default="bge", choices=["bge", "none"], help="Template mode")
    parser.add_argument("--max-len", type=int, default=0, help="Max sequence length (0 = use model default)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--peft-adapter", type=str, default=None, help="Path to PEFT adapter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    
    args = parser.parse_args()
    
    # 모델 로드 (서버 시작 전)
    load_encoder(
        model_path=args.model,
        template=args.template,
        max_len=args.max_len if args.max_len > 0 else None,
        device=args.device,
        peft_adapter=args.peft_adapter,
    )
    
    # uvicorn으로 서버 실행
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: poetry add uvicorn")
        sys.exit(1)
    
    print(f"[API] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

