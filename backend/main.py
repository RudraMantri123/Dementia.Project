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

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Dementia Support Chatbot API",
    description="Multi-agent AI system for dementia care and support",
    version="2.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
chatbot_instances = {}
analyst_instance = AnalystAgent()


# Pydantic Models
class InitRequest(BaseModel):
    model_type: str = "ollama"  # "openai" or "ollama"
    api_key: Optional[str] = None  # Required only for OpenAI
    model_name: str = "llama3:latest"  # Default to free llama model (with exact tag)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


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
def get_chatbot(session_id: str = "default") -> MultiAgentChatbotFlexible:
    """Get chatbot instance for session."""
    if session_id not in chatbot_instances:
        raise HTTPException(status_code=400, detail="Chatbot not initialized. Call /initialize first.")
    return chatbot_instances[session_id]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dementia Support Chatbot API",
        "version": "1.0.0",
        "status": "online"
    }


@app.post("/initialize")
async def initialize_chatbot(request: InitRequest):
    """Initialize the chatbot with model selection."""
    try:
        # Validate based on model type
        if request.model_type == "openai":
            if not request.api_key or not request.api_key.startswith("sk-"):
                raise HTTPException(status_code=400, detail="Valid OpenAI API key required for OpenAI models")

        # Load knowledge base
        kb = KnowledgeBase()
        vector_store_path = "data/vector_store"

        if not os.path.exists(vector_store_path):
            raise HTTPException(
                status_code=404,
                detail="Vector store not found. Please run build_knowledge_base.py first"
            )

        kb.load(vector_store_path)

        # Create chatbot instance with flexible model support
        session_id = "default"  # For now, single session
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
                response="Please refresh the page and configure the chatbot in the sidebar to begin our conversation.",
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
            response="I apologize, but I encountered an unexpected error. Please try asking your question again, or start a new conversation.",
            agent='system',
            intent='error',
            sources=None
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(session_id: str = "default"):
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
async def reset_conversation(session_id: str = "default"):
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


@app.get("/ml/intervention-plan/{session_id}")
async def get_intervention_plan(session_id: str = "default"):
    """Get ML-powered intervention plan with personalized recommendations."""
    try:
        chatbot = get_chatbot(session_id)
        conversation_history = chatbot.conversation_log
        
        # Convert to format expected by analyst agent
        formatted_history = [
            {
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'message': msg.get('content', ''),
                'sentiment': msg.get('sentiment', 'neutral'),
                'timestamp': msg.get('timestamp', '')
            }
            for msg in conversation_history
        ]
        
        # Get ML intervention plan
        ml_plan = analyst_instance.get_ml_intervention_plan(formatted_history, session_id)
        
        return {
            "status": "success",
            "intervention_plan": ml_plan
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/risk-assessment/{session_id}")
async def get_risk_assessment(session_id: str = "default"):
    """Get burnout and crisis risk assessment."""
    try:
        chatbot = get_chatbot(session_id)
        conversation_history = chatbot.conversation_log
        
        formatted_history = [
            {
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'message': msg.get('content', ''),
                'sentiment': msg.get('sentiment', 'neutral'),
                'timestamp': msg.get('timestamp', '')
            }
            for msg in conversation_history
        ]
        
        # Get risk assessment
        risk_assessment = analyst_instance.get_risk_assessment(formatted_history)
        
        return {
            "status": "success",
            "risk_assessment": risk_assessment
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("=== ERROR IN RISK ASSESSMENT ===")
        traceback.print_exc()
        print("=================================")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/personalized-intervention/{session_id}")
async def get_personalized_intervention(session_id: str = "default"):
    """Get personalized therapeutic intervention recommendation."""
    try:
        chatbot = get_chatbot(session_id)
        conversation_history = chatbot.conversation_log
        
        formatted_history = [
            {
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'message': msg.get('content', ''),
                'sentiment': msg.get('sentiment', 'neutral'),
                'timestamp': msg.get('timestamp', '')
            }
            for msg in conversation_history
        ]
        
        # Get personalized intervention
        intervention = analyst_instance.get_personalized_intervention(formatted_history, session_id)
        
        return {
            "status": "success",
            "intervention": intervention
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
