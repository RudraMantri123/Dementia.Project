"""Constants and default values for the Dementia Chatbot backend."""

# API Configuration
API_TITLE = "Dementia Support Chatbot API"
API_DESCRIPTION = "Multi-agent AI system for dementia care and support"
API_VERSION = "2.1.0"

# Server Configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_RELOAD = True

# CORS Configuration
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173"
]
DEFAULT_ALLOW_CREDENTIALS = True
DEFAULT_ALLOWED_METHODS = ["*"]
DEFAULT_ALLOWED_HEADERS = ["*"]

# Default Model Configuration
DEFAULT_MODEL_TYPE = "ollama"
DEFAULT_MODEL_NAME = "llama3:latest"
OPENAI_API_KEY_PREFIX = "sk-"

# Data Paths
DEFAULT_VECTOR_STORE_PATH = "data/vector_store"
DEFAULT_KNOWLEDGE_BASE_PATH = "data"
DEFAULT_MODELS_PATH = "data/models"

# Session Configuration
DEFAULT_SESSION_ID = "default"
MAX_SESSIONS = 100

# Analytics Configuration
MIN_MESSAGES_FOR_ANALYTICS = 5
ANALYTICS_CONFIDENCE_THRESHOLD = 0.6

# Error Messages
CHATBOT_NOT_INITIALIZED_MSG = "Chatbot not initialized. Call /initialize first."
VECTOR_STORE_NOT_FOUND_MSG = "Vector store not found. Please run build_knowledge_base.py first"
INVALID_OPENAI_KEY_MSG = "Valid OpenAI API key required for OpenAI models"
GRACEFUL_ERROR_MSG = "I apologize, but I encountered an unexpected error. Please try asking your question again, or start a new conversation."
INITIALIZATION_REQUIRED_MSG = "Please refresh the page and configure the chatbot in the sidebar to begin our conversation."

# Health Check Configuration
HEALTH_CHECK_MESSAGE = "Dementia Support Chatbot API"
HEALTH_CHECK_VERSION = "1.0.0"
HEALTH_CHECK_STATUS = "online"



