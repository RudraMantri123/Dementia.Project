"""Database models for advanced features."""

from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class UserProfile(Base):
    """User profile with personalization data."""

    __tablename__ = 'user_profiles'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200))
    age = Column(Integer)
    dementia_stage = Column(String(50))  # early, moderate, advanced
    language_preference = Column(String(10), default='en')

    # Personalization data
    interaction_preferences = Column(JSON)  # voice/text, response length, topics
    cognitive_level = Column(Float)  # 0.0-1.0 scale
    engagement_patterns = Column(JSON)  # time of day, frequency, duration

    # Medical information
    medical_conditions = Column(JSON)
    medications = Column(JSON)
    caregiver_info = Column(JSON)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_interaction = Column(DateTime)

    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    analytics = relationship("UserAnalytics", back_populates="user")
    activities = relationship("ActivityLog", back_populates="user")


class Conversation(Base):
    """Conversation history with full context."""

    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), ForeignKey('user_profiles.user_id'), nullable=False, index=True)
    session_id = Column(String(100), nullable=False, index=True)

    message_index = Column(Integer)  # Message order in conversation
    role = Column(String(20))  # user, assistant, system
    content = Column(Text, nullable=False)

    # Agent routing info
    agent_used = Column(String(50))
    intent = Column(String(100))
    confidence = Column(Float)

    # Sentiment analysis
    sentiment = Column(String(50))
    sentiment_score = Column(Float)
    emotion = Column(String(50))

    # Context
    context_data = Column(JSON)

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    response_time_ms = Column(Integer)

    # Relationships
    user = relationship("UserProfile", back_populates="conversations")


class UserAnalytics(Base):
    """Analytics data for longitudinal tracking."""

    __tablename__ = 'user_analytics'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), ForeignKey('user_profiles.user_id'), nullable=False, index=True)

    date = Column(DateTime, default=datetime.utcnow, index=True)

    # Engagement metrics
    total_messages = Column(Integer, default=0)
    session_count = Column(Integer, default=0)
    avg_session_duration = Column(Float)

    # Sentiment metrics
    overall_sentiment = Column(String(50))
    sentiment_distribution = Column(JSON)
    stress_level = Column(Float)  # 0.0-1.0

    # Cognitive metrics
    cognitive_exercises_completed = Column(Integer, default=0)
    cognitive_performance_score = Column(Float)

    # Agent usage
    agent_distribution = Column(JSON)

    # Intervention flags
    needs_support = Column(Boolean, default=False)
    support_level = Column(String(50))  # low, medium, high, urgent

    # Predictions
    predicted_stress_7d = Column(Float)
    predicted_engagement_7d = Column(Float)

    # Recommendations
    recommended_interventions = Column(JSON)

    # Relationships
    user = relationship("UserProfile", back_populates="analytics")


class ActivityLog(Base):
    """Activity tracking for health metrics."""

    __tablename__ = 'activity_logs'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), ForeignKey('user_profiles.user_id'), nullable=False, index=True)

    activity_type = Column(String(100))  # exercise, sleep, medication, mood
    activity_data = Column(JSON)

    # Health metrics
    heart_rate = Column(Integer)
    sleep_hours = Column(Float)
    steps = Column(Integer)
    mood_rating = Column(Integer)  # 1-10

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    notes = Column(Text)

    # Relationships
    user = relationship("UserProfile", back_populates="activities")


class CognitiveExerciseResult(Base):
    """Results from cognitive exercises."""

    __tablename__ = 'cognitive_exercise_results'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100))

    exercise_type = Column(String(100))  # memory, pattern, orientation, calculation
    difficulty_level = Column(Integer)  # 1-5

    # Exercise data
    exercise_content = Column(JSON)
    user_response = Column(Text)
    correct_answer = Column(Text)

    # Performance
    is_correct = Column(Boolean)
    completion_time_seconds = Column(Integer)
    attempts = Column(Integer)
    performance_score = Column(Float)  # 0.0-1.0

    # Feedback
    feedback = Column(Text)

    timestamp = Column(DateTime, default=datetime.utcnow)


class ClinicalData(Base):
    """Clinical data for EHR integration."""

    __tablename__ = 'clinical_data'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)

    # FHIR-compatible fields
    fhir_patient_id = Column(String(200))
    provider_id = Column(String(200))

    # Clinical assessments
    assessment_type = Column(String(100))  # MMSE, MoCA, CDR, etc.
    assessment_score = Column(Float)
    assessment_date = Column(DateTime)
    assessment_data = Column(JSON)

    # Diagnosis
    diagnosis_code = Column(String(50))  # ICD-10
    diagnosis_description = Column(Text)
    diagnosis_date = Column(DateTime)

    # Treatment
    treatment_plan = Column(JSON)
    medications = Column(JSON)

    # Clinical notes
    provider_notes = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeedbackLog(Base):
    """User feedback for reinforcement learning."""

    __tablename__ = 'feedback_logs'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100))
    conversation_id = Column(Integer, ForeignKey('conversations.id'))

    # Feedback data
    feedback_type = Column(String(50))  # rating, correction, flag
    rating = Column(Integer)  # 1-5 stars
    helpful = Column(Boolean)

    # Context
    agent_used = Column(String(50))
    query = Column(Text)
    response = Column(Text)

    # Correction data (for learning)
    correction_provided = Column(Text)
    expected_behavior = Column(Text)

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)


class KnowledgeGraphNode(Base):
    """Nodes for graph-based knowledge representation."""

    __tablename__ = 'knowledge_graph_nodes'

    id = Column(Integer, primary_key=True)
    node_id = Column(String(200), unique=True, nullable=False, index=True)

    node_type = Column(String(100))  # concept, symptom, treatment, medication, etc.
    name = Column(String(500), nullable=False)
    description = Column(Text)

    # Properties
    properties = Column(JSON)

    # Embeddings for semantic search
    embedding = Column(JSON)  # Store as JSON array

    # Metadata
    source = Column(String(500))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeGraphEdge(Base):
    """Edges for knowledge graph relationships."""

    __tablename__ = 'knowledge_graph_edges'

    id = Column(Integer, primary_key=True)

    source_node_id = Column(String(200), ForeignKey('knowledge_graph_nodes.node_id'), nullable=False, index=True)
    target_node_id = Column(String(200), ForeignKey('knowledge_graph_nodes.node_id'), nullable=False, index=True)

    relationship_type = Column(String(100))  # causes, treats, contraindicates, etc.
    weight = Column(Float, default=1.0)

    # Properties
    properties = Column(JSON)

    # Metadata
    source = Column(String(500))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelTrainingLog(Base):
    """Log for fine-tuning and model training."""

    __tablename__ = 'model_training_logs'

    id = Column(Integer, primary_key=True)

    model_name = Column(String(200), nullable=False)
    model_version = Column(String(100))
    training_type = Column(String(100))  # fine-tune, rlhf, evaluation

    # Training configuration
    config = Column(JSON)

    # Dataset info
    dataset_size = Column(Integer)
    training_samples = Column(Integer)
    validation_samples = Column(Integer)

    # Metrics
    training_metrics = Column(JSON)
    validation_metrics = Column(JSON)

    # Status
    status = Column(String(50))  # running, completed, failed
    error_message = Column(Text)

    # Metadata
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
