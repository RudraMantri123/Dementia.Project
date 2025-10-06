"""Orchestrator agent for routing queries using Ollama models."""

from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate
from .base_agent_ollama import BaseAgentOllama


class AgentOrchestratorOllama(BaseAgentOllama):
    """Routes user queries to the appropriate specialized agent using Ollama."""

    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize the orchestrator with Ollama.

        Args:
            model_name: Name of the Ollama model
        """
        super().__init__(model_name, temperature=0.3)

        self.classification_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""You are an intelligent router for a dementia support chatbot.
Classify the user's intent into ONE of these categories:

1. FACTUAL - Questions about dementia facts, symptoms, treatments, caregiving information
   Examples: "What are the symptoms of dementia?", "How do I manage sundowning?", "What medications help?"

2. EMOTIONAL - Emotional support, venting, expressing feelings, seeking comfort
   Examples: "I'm feeling overwhelmed", "This is so hard", "I don't know how to cope"

3. COGNITIVE - Requests for memory exercises, brain training, cognitive activities
   Examples: "Can you give me a memory exercise?", "I want to play a game", "Help me practice remembering"

4. CASUAL - Greetings, small talk, general conversation
   Examples: "Hello", "How are you?", "Thanks for your help"

User input: {user_input}

Respond with ONLY ONE WORD: FACTUAL, EMOTIONAL, COGNITIVE, or CASUAL"""
        )

    def classify_intent(self, user_input: str) -> str:
        """
        Classify user intent.

        Args:
            user_input: User's message

        Returns:
            Intent classification (FACTUAL, EMOTIONAL, COGNITIVE, CASUAL)
        """
        try:
            prompt = self.classification_prompt.format(user_input=user_input)
            response = self.llm.invoke(prompt)
            intent = response.strip().upper()

            # Validate response
            valid_intents = ['FACTUAL', 'EMOTIONAL', 'COGNITIVE', 'CASUAL']
            if intent in valid_intents:
                return intent
            else:
                # Default to FACTUAL if unclear
                return 'FACTUAL'
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return 'FACTUAL'

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user input and determine routing.

        Args:
            user_input: User's message
            context: Optional context information

        Returns:
            Dictionary with intent and routing information
        """
        intent = self.classify_intent(user_input)

        # Map intent to agent
        agent_mapping = {
            'FACTUAL': 'knowledge',
            'EMOTIONAL': 'empathy',
            'COGNITIVE': 'cognitive',
            'CASUAL': 'empathy'  # Casual conversation handled by empathy agent
        }

        return {
            'intent': intent,
            'route_to': agent_mapping.get(intent, 'knowledge'),
            'confidence': 0.9  # Placeholder for confidence score
        }
