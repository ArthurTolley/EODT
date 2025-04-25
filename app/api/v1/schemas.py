# app/api/v1/schemas.py
from pydantic import BaseModel
from datetime import datetime

class EchoPayload(BaseModel):
    message: str
    count: int

class EchoResponse(BaseModel):
    message: str
    count: int
    timestamp: datetime