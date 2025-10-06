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
from src.personalization.user_profile_manager import UserProfileManager
from src.personalization.adaptive_response import AdaptiveResponseGenerator
from src.multimodal.image_exercises import ImageExerciseGenerator
from src.analytics.longitudinal_tracker import LongitudinalTracker
from src.analytics.predictive_models import PredictiveStressModeler
from src.clinical.provider_dashboard import ProviderDashboard
from src.clinical.ehr_connector import FHIRConnector
from src.knowledge.graph_kb import GraphKnowledgeBase
from src.learning.rlhf_system import RLHFSystem

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Dementia Support Chatbot API",
    description="Multi-agent AI system for dementia care and support with advanced features",
    version="2.0.0"
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
profile_manager = UserProfileManager()
adaptive_response = AdaptiveResponseGenerator()
image_exercise_gen = ImageExerciseGenerator()
longitudinal_tracker = LongitudinalTracker()
stress_modeler = PredictiveStressModeler()
provider_dashboard = ProviderDashboard()
ehr_connector = FHIRConnector()
graph_kb = GraphKnowledgeBase()
rlhf_system = RLHFSystem()


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

        return ChatResponse(
            response=result.get('response', 'No response'),
            agent=result.get('agent', 'system'),
            intent=result.get('intent', 'unknown'),
            sources=result.get('num_sources')
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            insights=insights
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(chatbot_instances)
    }


# ============= Advanced Personalization Endpoints =============

@app.post("/profile/create")
async def create_user_profile(
    user_id: str,
    name: str = None,
    age: int = None,
    dementia_stage: str = "early"
):
    """Create user profile for personalization."""
    try:
        profile = profile_manager.create_profile(user_id, name, age, dementia_stage)
        return {
            "status": "success",
            "user_id": user_id,
            "message": "Profile created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile and personalized settings."""
    try:
        settings = profile_manager.get_personalized_settings(user_id)
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profile/{user_id}/learn")
async def learn_from_interactions(user_id: str, days: int = 30):
    """Analyze and learn from user interactions."""
    try:
        insights = profile_manager.learn_from_interactions(user_id, days)
        return {
            "status": "success",
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Multi-Modal Exercise Endpoints =============

@app.post("/exercises/image/pattern")
async def generate_pattern_exercise(difficulty: int = 3):
    """Generate pattern recognition image exercise."""
    try:
        exercise = image_exercise_gen.generate_pattern_recognition_exercise(difficulty)
        return exercise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/exercises/image/memory")
async def generate_memory_exercise(difficulty: int = 3):
    """Generate memory matching image exercise."""
    try:
        exercise = image_exercise_gen.generate_memory_matching_exercise(difficulty)
        return exercise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/exercises/image/differences")
async def generate_difference_exercise(difficulty: int = 3):
    """Generate find-the-differences exercise."""
    try:
        exercise = image_exercise_gen.generate_find_difference_exercise(difficulty)
        return exercise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/exercises/image/sequence")
async def generate_sequence_exercise(difficulty: int = 3):
    """Generate sequence ordering exercise."""
    try:
        exercise = image_exercise_gen.generate_sequence_exercise(difficulty)
        return exercise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Enhanced Analytics Endpoints =============

@app.get("/analytics/longitudinal/{user_id}")
async def get_longitudinal_analysis(user_id: str, weeks: int = 12):
    """Get longitudinal performance trends."""
    try:
        cognitive_trends = longitudinal_tracker.track_cognitive_performance(user_id, weeks=weeks)
        engagement_trends = longitudinal_tracker.track_engagement_trends(user_id, weeks=weeks)
        sentiment_trends = longitudinal_tracker.track_sentiment_trends(user_id, weeks=weeks)

        return {
            "cognitive_trends": cognitive_trends,
            "engagement_trends": engagement_trends,
            "sentiment_trends": sentiment_trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/stress-prediction/{user_id}")
async def predict_stress(user_id: str, days_ahead: int = 7):
    """Predict future stress levels."""
    try:
        prediction = stress_modeler.predict_stress(user_id, days_ahead)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Clinical Integration Endpoints =============

@app.get("/provider/dashboard/overview/{user_id}")
async def get_provider_patient_overview(user_id: str):
    """Get comprehensive patient overview for provider."""
    try:
        overview = provider_dashboard.get_patient_overview(user_id)
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provider/dashboard/trends/{user_id}")
async def get_provider_patient_trends(user_id: str, weeks: int = 12):
    """Get patient trends for provider dashboard."""
    try:
        trends = provider_dashboard.get_patient_trends(user_id, weeks)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provider/dashboard/patients")
async def get_provider_patient_list(
    provider_id: str,
    dementia_stage: str = None,
    min_cognitive_level: float = None,
    max_cognitive_level: float = None
):
    """Get list of patients for provider with filters."""
    try:
        filters = {}
        if dementia_stage:
            filters['dementia_stage'] = dementia_stage
        if min_cognitive_level is not None:
            filters['min_cognitive_level'] = min_cognitive_level
        if max_cognitive_level is not None:
            filters['max_cognitive_level'] = max_cognitive_level

        patient_list = provider_dashboard.get_patient_list(provider_id, filters)
        return {"patients": patient_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provider/dashboard/alerts/{provider_id}")
async def get_provider_alerts(provider_id: str):
    """Get clinical alerts for provider."""
    try:
        alerts = provider_dashboard.get_alerts(provider_id)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provider/dashboard/statistics/{provider_id}")
async def get_provider_statistics(provider_id: str):
    """Get aggregate statistics for provider."""
    try:
        stats = provider_dashboard.get_aggregate_statistics(provider_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provider/export-report/{user_id}")
async def export_patient_report(user_id: str):
    """Generate comprehensive patient report."""
    try:
        report = provider_dashboard.export_patient_report(user_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= EHR Integration Endpoints =============

@app.post("/ehr/sync")
async def sync_patient_from_ehr(user_id: str, fhir_patient_id: str):
    """Sync patient data from EHR system."""
    try:
        result = ehr_connector.sync_patient_to_db(user_id, fhir_patient_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ehr/patient/{fhir_patient_id}")
async def get_ehr_patient_data(fhir_patient_id: str):
    """Fetch patient data from EHR."""
    try:
        patient_data = ehr_connector.fetch_patient_data(fhir_patient_id)
        return patient_data or {"error": "Patient not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Knowledge Graph Endpoints =============

@app.get("/knowledge-graph/concept/{concept_id}")
async def get_concept_info(concept_id: str):
    """Get information about a medical concept."""
    try:
        info = graph_kb.get_concept_info(concept_id)
        return info or {"error": "Concept not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-graph/related/{concept_id}")
async def get_related_concepts(concept_id: str, max_distance: int = 2):
    """Find concepts related to given concept."""
    try:
        related = graph_kb.find_related_concepts(concept_id, max_distance=max_distance)
        return {"related_concepts": related}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-graph/treatments/{symptom_id}")
async def get_treatments_for_symptom(symptom_id: str):
    """Find treatments for a specific symptom."""
    try:
        treatments = graph_kb.find_treatments_for_symptom(symptom_id)
        return {"treatments": treatments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats():
    """Get knowledge graph statistics."""
    try:
        stats = graph_kb.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= RLHF (Feedback Learning) Endpoints =============

@app.post("/feedback/collect")
async def collect_user_feedback(
    user_id: str,
    session_id: str,
    conversation_id: int,
    feedback_type: str,
    rating: int = None,
    helpful: bool = None,
    correction: str = None,
    notes: str = None
):
    """Collect user feedback for continuous improvement."""
    try:
        result = rlhf_system.collect_feedback(
            user_id,
            session_id,
            conversation_id,
            feedback_type,
            rating,
            helpful,
            correction,
            notes
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/analysis")
async def get_feedback_analysis(agent_name: str = None, days: int = 30):
    """Analyze feedback patterns."""
    try:
        analysis = rlhf_system.analyze_feedback_patterns(agent_name, days)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/improvements/{agent_name}")
async def get_improvement_suggestions(agent_name: str):
    """Get improvement suggestions for agent."""
    try:
        suggestions = rlhf_system.get_improvement_suggestions(agent_name)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
