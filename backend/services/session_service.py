from typing import Dict
from fastapi import HTTPException
from backend.config import settings


class SessionService:

    def __init__(self):
        self.sessions: Dict = {}

    def get_session(self, session_id: str):
        if session_id not in self.sessions:
            raise HTTPException(status_code=400, detail=settings.chatbot_not_initialized_msg)
        return self.sessions[session_id]

    def create_session(self, session_id: str, chatbot_instance):
        self.sessions[session_id] = chatbot_instance

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get_active_session_count(self) -> int:
        return len(self.sessions)
