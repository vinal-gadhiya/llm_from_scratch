import os
import torch
from fastapi import APIRouter, HTTPException

from core import config 
from serving.state import model_state

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_endpoint():
    if not model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "model_loaded": False,
                "error": model_state["error"]
            }
        )
    return {
        "status": "healthy",
        "model_loaded": True
    }