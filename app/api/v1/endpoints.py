# app/api/v1/endpoints.py
from fastapi import APIRouter
from datetime import datetime, timezone
from .schemas import EchoPayload, EchoResponse

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/echo", response_model=EchoResponse)
def echo(payload: EchoPayload):
    return {
        "message": payload.message,
        "count": payload.count,
        "timestamp": datetime.now(timezone.utc)  # Updated to use timezone-aware datetime
    }
