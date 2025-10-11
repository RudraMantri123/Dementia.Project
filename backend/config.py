"""Configuration management for the Dementia Chatbot backend."""

import os
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    api_title: str = "Dementia Support Chatbot API"
    api_description: str = "Multi-agent AI system for dementia care and support"
    api_version: str = "2.1.0"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    allow_credentials: bool = True
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]

    # Default Model Configuration
    default_model_type: str = "ollama"
    default_model_name: str = "llama3:latest"
    openai_api_key_prefix: str = "sk-"

    # Data Paths
    vector_store_path: str = "data/vector_store"
    knowledge_base_path: str = "data"
    models_path: str = "data/models"

    # Session Configuration
    default_session_id: str = "default"
    max_sessions: int = 100

    # Analytics Configuration
    min_messages_for_analytics: int = 5
    analytics_confidence_threshold: float = 0.6

    # Error Messages
    chatbot_not_initialized_msg: str = "Chatbot not initialized. Call /initialize first."
    vector_store_not_found_msg: str = "Vector store not found. Please run build_knowledge_base.py first"
    invalid_openai_key_msg: str = "Valid OpenAI API key required for OpenAI models"
    graceful_error_msg: str = "I apologize, but I encountered an unexpected error. Please try asking your question again, or start a new conversation."
    initialization_required_msg: str = "Please refresh the page and configure the chatbot in the sidebar to begin our conversation."

    # Health Check Configuration
    health_check_message: str = "Dementia Support Chatbot API"
    health_check_version: str = "1.0.0"
    health_check_status: str = "online"

    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    @field_validator('allowed_methods', mode='before')
    @classmethod
    def parse_allowed_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(',')]
        return v

    @field_validator('allowed_headers', mode='before')
    @classmethod
    def parse_allowed_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(',')]
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()



