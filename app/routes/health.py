"""Health-check routes for the backend."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Return a basic health response for service monitoring."""
    return {"status": "ok"}
