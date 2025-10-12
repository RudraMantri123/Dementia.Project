from fastapi import APIRouter, HTTPException
from backend.models import InitRequest, ChatRequest, ChatResponse, StatusResponse
from backend.services import ChatbotService, SessionService
from backend.config import settings

router = APIRouter()
session_service: SessionService = None


def set_session_service(service: SessionService):
    global session_service
    session_service = service


@router.post("/initialize")
async def initialize_chatbot(request: InitRequest):
    try:
        chatbot = ChatbotService.initialize_chatbot(
            model_type=request.model_type,
            api_key=request.api_key,
            model_name=request.model_name
        )

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
    try:
        chatbot = session_service.get_session(request.session_id)
        result = ChatbotService.process_message(chatbot, request.message)

        return ChatResponse(
            response=result['response'],
            agent=result['agent'],
            intent=result['intent'],
            sources=result['num_sources']
        )
    except HTTPException as he:
        if he.status_code == 400 and "not initialized" in str(he.detail):
            return ChatResponse(
                response=settings.initialization_required_msg,
                agent='system',
                intent='initialization_required',
                sources=None
            )
        raise
    except Exception as e:
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
    try:
        chatbot = session_service.get_session(session_id)
        ChatbotService.reset_conversation(chatbot)

        return StatusResponse(status="success", message="Conversation reset successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
