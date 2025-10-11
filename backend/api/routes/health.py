"""Health check and status endpoints."""

from fastapi import APIRouter
from backend.models import HealthResponse
from backend.services import SessionService
from backend.config import settings

router = APIRouter()

# Dependency: Session service (will be injected from main)
session_service: SessionService = None


def set_session_service(service: SessionService):
    """Set the session service instance."""
    global session_service
    session_service = service


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": settings.health_check_message,
        "version": settings.health_check_version,
        "status": settings.health_check_status
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        active_sessions=session_service.get_active_session_count()
    )
