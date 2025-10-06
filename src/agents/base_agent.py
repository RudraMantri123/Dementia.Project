"""Base agent class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_community.chat_models import ChatOpenAI


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize the base agent.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model
            temperature: Temperature for response generation
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        self.agent_name = self.__class__.__name__
        self.conversation_history = []

    @abstractmethod
    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user input and generate response.

        Args:
            user_input: The user's message
            context: Optional context information

        Returns:
            Dictionary containing response and metadata
        """
        pass

    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            role: Role of the speaker (user/assistant)
            content: Message content
        """
        self.conversation_history.append({
            'role': role,
            'content': content
        })

    def get_history(self, last_n: int = 5) -> list:
        """
        Get recent conversation history.

        Args:
            last_n: Number of recent messages to retrieve

        Returns:
            List of recent messages
        """
        return self.conversation_history[-last_n:] if self.conversation_history else []

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
