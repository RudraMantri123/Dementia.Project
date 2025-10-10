"""Improved FastAPI backend with comprehensive error handling."""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base import KnowledgeBase
from src.multi_agent_system_flexible import MultiAgentChatbotFlexible
from src.agents.analyst_agent import AnalystAgent

# Import error handlers
from backend.error_handlers import (
    chatbot_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
    ChatbotException,
    InitializationError,
    ModelNotFoundError,
    ChatProcessingError,
    SessionError,
    sanitize_input,
    validate_input,
    safe_execute,
    logger
)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Dementia Support Chatbot API",
    description="Production-ready multi-agent AI system for dementia care",
    version="2.0.0"
)

# Register exception handlers
app.add_exception_handler(ChatbotException, chatbot_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
chatbot_instances = {}
analyst_instance = None

# Initialize analyst
try:
    analyst_instance = AnalystAgent()
    logger.info("Analyst agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize analyst agent: {e}")
    analyst_instance = None


# Enhanced Pydantic Models with validation
class InitRequest(BaseModel):
    model_type: str = Field(default="ollama", pattern="^(openai|ollama)$")
    api_key: Optional[str] = Field(default=None, min_length=10, max_length=200)
    model_name: str = Field(default="llama3:latest", min_length=1, max_length=100)

    @validator('api_key')
    def validate_api_key(cls, v, values):
        if values.get('model_type') == 'openai':
            if not v:
                raise ValueError("API key required for OpenAI models")
            if not v.startswith('sk-'):
                raise ValueError("Invalid OpenAI API key format")
        return v

    class Config:
        schema_extra = {
            "example": {
                "model_type": "ollama",
                "model_name": "llama3:latest"
            }
        }


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field(default="default", min_length=1, max_length=100)

    @validator('message')
    def sanitize_message(cls, v):
        return sanitize_input(v)

    class Config:
        schema_extra = {
            "example": {
                "message": "What are the symptoms of dementia?",
                "session_id": "default"
            }
        }


class ChatResponse(BaseModel):
    response: str
    agent: str
    intent: str
    sources: Optional[int] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    total_messages: int
    agent_distribution: dict
    intent_distribution: dict


class AnalyticsResponse(BaseModel):
    overall_sentiment: str
    sentiment_distribution: dict
    needs_support: dict
    insights: list


# Helper function to get or create chatbot
def get_chatbot(session_id: str = "default") -> MultiAgentChatbotFlexible:
    """
    Get chatbot instance for session with error handling.

    Args:
        session_id: Session identifier

    Returns:
        MultiAgentChatbotFlexible instance

    Raises:
        SessionError: If chatbot not initialized for session
    """
    if session_id not in chatbot_instances:
        raise SessionError(
            "Chatbot not initialized. Please initialize the chatbot first.",
            details={"session_id": session_id}
        )
    return chatbot_instances[session_id]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Dementia Support Chatbot API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/initialize")
async def initialize_chatbot(request: InitRequest):
    """
    Initialize the chatbot with model selection.

    Raises:
        InitializationError: If initialization fails
        ModelNotFoundError: If required files not found
    """
    try:
        logger.info(f"Initializing chatbot with {request.model_type}")

        # Check vector store exists
        vector_store_path = "data/vector_store"
        if not os.path.exists(vector_store_path):
            raise ModelNotFoundError(
                "Vector store not found. Please run build_knowledge_base.py first",
                details={"expected_path": vector_store_path}
            )

        # Load knowledge base
        kb = KnowledgeBase()
        try:
            kb.load(vector_store_path)
            logger.info("Knowledge base loaded successfully")
        except Exception as e:
            raise InitializationError(
                f"Failed to load knowledge base: {str(e)}",
                details={"vector_store_path": vector_store_path}
            )

        # Create chatbot instance
        session_id = "default"
        try:
            chatbot_instances[session_id] = MultiAgentChatbotFlexible(
                knowledge_base=kb,
                model_type=request.model_type,
                api_key=request.api_key,
                model_name=request.model_name
            )
            logger.info(f"Chatbot instance created for session {session_id}")
        except Exception as e:
            raise InitializationError(
                f"Failed to create chatbot instance: {str(e)}",
                details={"model_type": request.model_type}
            )

        return {
            "status": "success",
            "message": f"Chatbot initialized successfully with {request.model_type}",
            "session_id": session_id,
            "model_type": request.model_type,
            "model": request.model_name
        }

    except (InitializationError, ModelNotFoundError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}", exc_info=True)
        raise InitializationError(
            "An unexpected error occurred during initialization",
            details={"error": str(e)}
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot.

    Raises:
        SessionError: If chatbot not initialized
        ChatProcessingError: If message processing fails
    """
    try:
        logger.info(f"Processing chat message for session {request.session_id}")

        # Get chatbot instance
        try:
            chatbot = get_chatbot(request.session_id)
        except SessionError:
            # Provide user-friendly message
            return ChatResponse(
                response="Please configure the chatbot in the sidebar before starting a conversation.",
                agent='system',
                intent='initialization_required',
                sources=None
            )

        # Process message
        try:
            result = chatbot.chat(request.message)

            # Validate result
            if not result or not isinstance(result, dict):
                logger.warning("Invalid result from chatbot.chat()")
                raise ChatProcessingError("Invalid response from chatbot")

            return ChatResponse(
                response=result.get('response', 'I apologize, but I had trouble generating a response.'),
                agent=result.get('agent', 'system'),
                intent=result.get('intent', 'unknown'),
                sources=result.get('num_sources')
            )

        except Exception as e:
            logger.error(f"Error processing chat message: {e}", exc_info=True)
            return ChatResponse(
                response="I apologize, but I encountered an error processing your message. Please try again.",
                agent='system',
                intent='error',
                sources=None,
                error=str(e) if logger.level == logging.DEBUG else None
            )

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        return ChatResponse(
            response="An unexpected error occurred. Please refresh and try again.",
            agent='system',
            intent='error',
            sources=None
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(session_id: str = "default"):
    """
    Get conversation statistics.

    Raises:
        SessionError: If chatbot not initialized
    """
    try:
        chatbot = get_chatbot(session_id)
        stats = safe_execute(
            chatbot.get_conversation_stats,
            fallback_value={
                'total_messages': 0,
                'agent_distribution': {},
                'intent_distribution': {}
            },
            error_message="Error getting conversation stats"
        )

        return StatsResponse(
            total_messages=stats.get('total_messages', 0),
            agent_distribution=stats.get('agent_distribution', {}),
            intent_distribution=stats.get('intent_distribution', {})
        )

    except SessionError:
        raise


@app.post("/analytics", response_model=AnalyticsResponse)
async def get_analytics(request: ChatRequest):
    """
    Get sentiment analytics for conversation.

    Raises:
        SessionError: If chatbot not initialized
    """
    try:
        chatbot = get_chatbot(request.session_id)

        # Get user messages from conversation log
        user_messages = [
            {'role': 'user', 'content': entry['user_input']}
            for entry in getattr(chatbot, 'conversation_log', [])
        ]

        if not user_messages:
            return AnalyticsResponse(
                overall_sentiment="neutral",
                sentiment_distribution={},
                needs_support={'level': 'low'},
                insights=["Not enough messages for analysis"]
            )

        if not analyst_instance:
            logger.warning("Analyst agent not available")
            return AnalyticsResponse(
                overall_sentiment="neutral",
                sentiment_distribution={},
                needs_support={'level': 'low'},
                insights=["Analytics temporarily unavailable"]
            )

        # Analyze conversation
        analytics = safe_execute(
            analyst_instance.analyze_conversation,
            user_messages,
            fallback_value={
                'overall_sentiment': 'neutral',
                'sentiment_distribution': {},
                'needs_support': {'level': 'low'}
            },
            error_message="Error analyzing conversation"
        )

        insights = safe_execute(
            analyst_instance.get_insights,
            analytics,
            fallback_value=[],
            error_message="Error generating insights"
        )

        return AnalyticsResponse(
            overall_sentiment=analytics.get('overall_sentiment', 'neutral'),
            sentiment_distribution=analytics.get('sentiment_distribution', {}),
            needs_support=analytics.get('needs_support', {'level': 'low'}),
            insights=insights
        )

    except SessionError:
        raise
    except Exception as e:
        logger.error(f"Error in analytics endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing analytics")


@app.post("/reset")
async def reset_conversation(session_id: str = "default"):
    """
    Reset the conversation history.

    Raises:
        SessionError: If chatbot not initialized
    """
    try:
        chatbot = get_chatbot(session_id)
        safe_execute(
            chatbot.reset_conversation,
            error_message=f"Error resetting conversation for session {session_id}"
        )

        return {
            "status": "success",
            "message": "Conversation reset successfully"
        }

    except SessionError:
        raise


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        Health status with component checks
    """
    health_status = {
        "status": "healthy",
        "active_sessions": len(chatbot_instances),
        "components": {
            "analyst_agent": analyst_instance is not None,
            "vector_store": os.path.exists("data/vector_store")
        }
    }

    # Check if all components are healthy
    all_healthy = all(health_status["components"].values())
    if not all_healthy:
        health_status["status"] = "degraded"

    return health_status


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Dementia Support Chatbot API v2.0.0")
    logger.info(f"Environment: {'Production' if os.getenv('ENV') == 'production' else 'Development'}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Dementia Support Chatbot API")
    chatbot_instances.clear()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
