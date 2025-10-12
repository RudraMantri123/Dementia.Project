from fastapi import APIRouter
from backend.models import HealthResponse
from backend.services import SessionService
from backend.config import settings

router = APIRouter()
session_service: SessionService = None


def set_session_service(service: SessionService):
    global session_service
    session_service = service


@router.get("/")
async def root():
    return {
        "message": settings.health_check_message,
        "version": settings.health_check_version,
        "status": settings.health_check_status
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        active_sessions=session_service.get_active_session_count()
    )
