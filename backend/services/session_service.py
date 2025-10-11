"""Session management service."""

from typing import Dict
from fastapi import HTTPException
from backend.config import settings


class SessionService:
    """Manages chatbot sessions."""

    def __init__(self):
        self.sessions: Dict = {}

    def get_session(self, session_id: str):
        """Get chatbot instance for a session."""
        if session_id not in self.sessions:
            raise HTTPException(
                status_code=400,
                detail=settings.chatbot_not_initialized_msg
            )
        return self.sessions[session_id]

    def create_session(self, session_id: str, chatbot_instance):
        """Create a new session with chatbot instance."""
        self.sessions[session_id] = chatbot_instance

    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return session_id in self.sessions

    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)
