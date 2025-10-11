"""FastAPI backend for Dementia Chatbot."""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base import KnowledgeBase
from src.multi_agent_system_flexible import MultiAgentChatbotFlexible
from src.agents.analyst_agent import AnalystAgent
from backend.config import settings

# Load environment variables
load_dotenv()

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Global state (in production, use Redis or database)
chatbot_instances = {}
analyst_instance = AnalystAgent()


# Pydantic Models
class InitRequest(BaseModel):
    model_type: str = settings.default_model_type  # "openai" or "ollama"
    api_key: Optional[str] = None  # Required only for OpenAI
    model_name: str = settings.default_model_name  # Default model name


class ChatRequest(BaseModel):
    message: str
    session_id: str = settings.default_session_id


class ChatResponse(BaseModel):
    response: str
    agent: str
    intent: str
    sources: Optional[int] = None


class StatsResponse(BaseModel):
    total_messages: int
    agent_distribution: dict
    intent_distribution: dict


class AnalyticsResponse(BaseModel):
    overall_sentiment: str
    sentiment_distribution: dict
    needs_support: dict
    insights: list
    detailed_metrics: Optional[dict] = None
    emotional_intensity: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    sentiment_stability: Optional[float] = None
    emotional_volatility: Optional[float] = None
    message_analyses: Optional[list] = None


# Helper function to get or create chatbot
def get_chatbot(session_id: str = settings.default_session_id) -> MultiAgentChatbotFlexible:
    """Get chatbot instance for session."""
    if session_id not in chatbot_instances:
        raise HTTPException(status_code=400, detail=settings.chatbot_not_initialized_msg)
    return chatbot_instances[session_id]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": settings.health_check_message,
        "version": settings.health_check_version,
        "status": settings.health_check_status
    }


@app.post("/initialize")
async def initialize_chatbot(request: InitRequest):
    """Initialize the chatbot with model selection."""
    try:
        # Validate based on model type
        if request.model_type == "openai":
            if not request.api_key or not request.api_key.startswith(settings.openai_api_key_prefix):
                raise HTTPException(status_code=400, detail=settings.invalid_openai_key_msg)

        # Load knowledge base
        kb = KnowledgeBase()
        vector_store_path = settings.vector_store_path

        if not os.path.exists(vector_store_path):
            raise HTTPException(
                status_code=404,
                detail=settings.vector_store_not_found_msg
            )

        kb.load(vector_store_path)

        # Create chatbot instance with flexible model support
        session_id = settings.default_session_id  # For now, single session
        chatbot_instances[session_id] = MultiAgentChatbotFlexible(
            knowledge_base=kb,
            model_type=request.model_type,
            api_key=request.api_key,
            model_name=request.model_name
        )

        return {
            "status": "success",
            "message": f"Chatbot initialized successfully with {request.model_type}",
            "session_id": session_id,
            "model_type": request.model_type,
            "model": request.model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the chatbot."""
    try:
        chatbot = get_chatbot(request.session_id)
        result = chatbot.chat(request.message)

        # DEFENSIVE: Ensure result is valid
        if not result or not isinstance(result, dict):
            return ChatResponse(
                response="I apologize, but I encountered an issue processing your message. Please try rephrasing or ask me something else.",
                agent='system',
                intent='error',
                sources=None
            )

        return ChatResponse(
            response=result.get('response', 'I apologize, but I had trouble generating a response. Please try again.'),
            agent=result.get('agent', 'system'),
            intent=result.get('intent', 'unknown'),
            sources=result.get('num_sources')
        )

    except HTTPException as he:
        # Handle "not initialized" errors with helpful message
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


@app.get("/stats", response_model=StatsResponse)
async def get_stats(session_id: str = settings.default_session_id):
    """Get conversation statistics."""
    try:
        chatbot = get_chatbot(session_id)
        stats = chatbot.get_conversation_stats()

        return StatsResponse(
            total_messages=stats.get('total_messages', 0),
            agent_distribution=stats.get('agent_distribution', {}),
            intent_distribution=stats.get('intent_distribution', {})
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics")
async def get_analytics(request: ChatRequest):
    """Get sentiment analytics for conversation."""
    try:
        chatbot = get_chatbot(request.session_id)

        # Get user messages from conversation log
        user_messages = [
            {'role': 'user', 'content': entry['user_input']}
            for entry in chatbot.conversation_log
        ]

        if not user_messages:
            return AnalyticsResponse(
                overall_sentiment="neutral",
                sentiment_distribution={},
                needs_support={},
                insights=[]
            )

        analytics = analyst_instance.analyze_conversation(user_messages)
        insights = analyst_instance.get_insights(analytics)

        return AnalyticsResponse(
            overall_sentiment=analytics.get('overall_sentiment', 'neutral'),
            sentiment_distribution=analytics.get('sentiment_distribution', {}),
            needs_support=analytics.get('needs_support', {}),
            insights=insights,
            detailed_metrics=analytics.get('detailed_metrics', {}),
            emotional_intensity=analytics.get('emotional_intensity', 0.0),
            sentiment_confidence=analytics.get('sentiment_confidence', 0.0),
            sentiment_stability=analytics.get('sentiment_stability', 0.0),
            emotional_volatility=analytics.get('emotional_volatility', 0.0),
            message_analyses=analytics.get('message_analyses', [])
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_conversation(session_id: str = settings.default_session_id):
    """Reset the conversation history."""
    try:
        chatbot = get_chatbot(session_id)
        chatbot.reset_conversation()

        return {
            "status": "success",
            "message": "Conversation reset successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(chatbot_instances)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.reload)
