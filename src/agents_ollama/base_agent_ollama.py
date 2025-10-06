"""Base agent class for Ollama models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama


class BaseAgentOllama(ABC):
    """Abstract base class for all Ollama-based agents."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0.7
    ):
        """
        Initialize the base Ollama agent.

        Args:
            model_name: Name of the Ollama model (e.g., 'llama3.2', 'mistral', 'phi')
            temperature: Temperature for response generation
        """
        self.llm = Ollama(
            model=model_name,
            temperature=temperature
        )
        self.model_name = model_name
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
