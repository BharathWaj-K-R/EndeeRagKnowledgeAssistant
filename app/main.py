"""FastAPI entrypoint for the AI Knowledge Assistant."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.upload import router as upload_router
from utils.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="RAG-based document question answering system.",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health_router)
    application.include_router(upload_router, prefix="/api")
    application.include_router(chat_router, prefix="/api")
    return application


app = create_app()
