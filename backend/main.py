"""FastAPI backend for Dementia Chatbot - Modular Architecture."""

import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings
from backend.api.middleware import configure_cors
from backend.api.routes import chat, health
from backend.services import SessionService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Configure middleware
configure_cors(app)

# Initialize services
session_service = SessionService()

# Inject session service into route modules
chat.set_session_service(session_service)
health.set_session_service(session_service)

# Include routers
app.include_router(chat.router, tags=["Chat"])
app.include_router(health.router, tags=["Health"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
