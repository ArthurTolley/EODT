# app/main.py
from fastapi import FastAPI
from app.api.v1.endpoints import router as v1_router

app = FastAPI(title="EODT - Road Usability API")

app.include_router(v1_router, prefix="/api/v1")
