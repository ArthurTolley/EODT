# app/api/v1/endpoints.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/echo")
def echo(payload: dict):
    return {"you_sent": payload}
