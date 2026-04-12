import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router


def _allowed_origins() -> list[str]:
    # Helper that expands the allowed frontend origins from a single environment variable.
    raw_value = os.getenv("CHATBOT_ALLOWED_ORIGINS", "*")
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or ["*"]


# Application setup that keeps the FastAPI entry point in one place for uvicorn.
app = FastAPI(title="BF4A Chatbot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
