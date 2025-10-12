import os
from typing import Dict, Any
from fastapi import HTTPException

from src.knowledge_base import KnowledgeBase
from src.multi_agent_system_flexible import MultiAgentChatbotFlexible
from backend.config import settings


class ChatbotService:

    @staticmethod
    def initialize_chatbot(model_type: str, api_key: str = None,
                          model_name: str = settings.default_model_name) -> MultiAgentChatbotFlexible:

        if model_type == "openai":
            if not api_key or not api_key.startswith(settings.openai_api_key_prefix):
                raise HTTPException(status_code=400, detail=settings.invalid_openai_key_msg)

        kb = KnowledgeBase()
        vector_store_path = settings.vector_store_path

        if not os.path.exists(vector_store_path):
            raise HTTPException(status_code=404, detail=settings.vector_store_not_found_msg)

        kb.load(vector_store_path)

        return MultiAgentChatbotFlexible(
            knowledge_base=kb,
            model_type=model_type,
            api_key=api_key,
            model_name=model_name
        )

    @staticmethod
    def process_message(chatbot: MultiAgentChatbotFlexible, message: str) -> Dict[str, Any]:
        result = chatbot.chat(message)

        if not result or not isinstance(result, dict):
            return {
                'response': 'I apologize, but I encountered an issue processing your message. Please try rephrasing or ask me something else.',
                'agent': 'system',
                'intent': 'error',
                'num_sources': None
            }

        return {
            'response': result.get('response', 'I apologize, but I had trouble generating a response. Please try again.'),
            'agent': result.get('agent', 'system'),
            'intent': result.get('intent', 'unknown'),
            'num_sources': result.get('num_sources')
        }

    @staticmethod
    def reset_conversation(chatbot: MultiAgentChatbotFlexible):
        chatbot.reset_conversation()
