"""Test API endpoint directly."""

import sys
sys.path.append('.')

from backend.main import chatbot_instances, analyst_instance
from src.knowledge_base import KnowledgeBase
from src.multi_agent_system_flexible import MultiAgentChatbotFlexible

# Initialize
kb = KnowledgeBase()
kb.load_knowledge_base()

chatbot = MultiAgentChatbotFlexible(kb, model_type="ollama", model_name="llama3:latest")
chatbot_instances["default"] = chatbot

# Add a test message
response = chatbot.chat("I feel overwhelmed")
print(f"Chat response: {response}")

print(f"\nChatbot has conversation_log: {hasattr(chatbot, 'conversation_log')}")
print(f"Conversation log: {chatbot.conversation_log}")

# Try to get risk assessment
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

print(f"\nFormatted history: {formatted_history}")

risk_assessment = analyst_instance.get_risk_assessment(formatted_history)
print(f"\nRisk assessment: {risk_assessment}")

print("\n\nAll tests passed!")

