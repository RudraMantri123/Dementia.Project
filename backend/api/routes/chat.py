"""Chat-related API endpoints."""

from fastapi import APIRouter, HTTPException
from backend.models import InitRequest, ChatRequest, ChatResponse, StatusResponse
from backend.services import ChatbotService, SessionService
from backend.config import settings

router = APIRouter()

# Dependency: Session service (will be injected from main)
session_service: SessionService = None


def set_session_service(service: SessionService):
    """Set the session service instance."""
    global session_service
    session_service = service


@router.post("/initialize")
async def initialize_chatbot(request: InitRequest):
    """Initialize the chatbot with model selection."""
    try:
        # Initialize chatbot
        chatbot = ChatbotService.initialize_chatbot(
            model_type=request.model_type,
            api_key=request.api_key,
            model_name=request.model_name
        )

        # Create session
        session_id = settings.default_session_id
        session_service.create_session(session_id, chatbot)

        return {
            "status": "success",
            "message": f"Chatbot initialized successfully with {request.model_type}",
            "session_id": session_id,
            "model_type": request.model_type,
            "model": request.model_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the chatbot."""
    try:
        # Get chatbot instance
        chatbot = session_service.get_session(request.session_id)

        # Process message
        result = ChatbotService.process_message(chatbot, request.message)

        return ChatResponse(
            response=result['response'],
            agent=result['agent'],
            intent=result['intent'],
            sources=result['num_sources']
        )

    except HTTPException as he:
        # Handle "not initialized" errors
        if he.status_code == 400 and "not initialized" in str(he.detail):
            return ChatResponse(
                response=settings.initialization_required_msg,
                agent='system',
                intent='initialization_required',
                sources=None
            )
        raise
    except Exception as e:
        # Log error but return graceful message
        import traceback
        print(f"ERROR in chat endpoint: {str(e)}")
        print(traceback.format_exc())

        return ChatResponse(
            response=settings.graceful_error_msg,
            agent='system',
            intent='error',
            sources=None
        )


@router.post("/reset", response_model=StatusResponse)
async def reset_conversation(session_id: str = settings.default_session_id):
    """Reset the conversation history."""
    try:
        # Get chatbot instance
        chatbot = session_service.get_session(session_id)

        # Reset conversation
        ChatbotService.reset_conversation(chatbot)

        return StatusResponse(
            status="success",
            message="Conversation reset successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
