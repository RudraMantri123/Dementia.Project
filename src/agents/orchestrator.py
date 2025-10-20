"""Orchestrator agent for routing queries to specialized agents."""

from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent


class AgentOrchestrator(BaseAgent):
    """Routes user queries to the appropriate specialized agent."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the orchestrator.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model
        """
        super().__init__(api_key, model_name, temperature=0.3)

        self.classification_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""You are an intelligent router for a dementia support chatbot.
Classify the user's intent into ONE of these categories:

1. EMOTIONAL - Emotional support, venting, expressing feelings, seeking comfort, distress, crisis
   Examples: "I'm feeling overwhelmed", "This is so hard", "I don't know how to cope", "I feel very low", "I can't adjust to society", "I'm struggling with memory and feeling depressed"

2. FACTUAL - Questions about dementia facts, symptoms, treatments, caregiving information
   Examples: "What are the symptoms of dementia?", "How do I manage sundowning?", "What medications help?"

3. COGNITIVE - Explicit requests for memory exercises, brain training, cognitive activities (NOT emotional concerns about memory)
   Examples: "Can you give me a memory exercise?", "I want to play a game", "Help me practice remembering"

4. CASUAL - Greetings, small talk, general conversation
   Examples: "Hello", "How are you?", "Thanks for your help"

IMPORTANT ROUTING RULES:
- If someone mentions memory problems AND expresses emotional distress, route to EMOTIONAL
- If someone mentions feeling low, depressed, overwhelmed, or social isolation, route to EMOTIONAL
- If someone mentions cognitive concerns with emotional impact, route to EMOTIONAL
- Only route to COGNITIVE for explicit requests for exercises/games
- When in doubt between EMOTIONAL and other categories, choose EMOTIONAL for safety

User input: {user_input}

Respond with ONLY ONE WORD: FACTUAL, EMOTIONAL, COGNITIVE, or CASUAL"""
        )

    def classify_intent(self, user_input: str) -> str:
        """
        Classify user intent with crisis detection.

        Args:
            user_input: User's message

        Returns:
            Intent classification (FACTUAL, EMOTIONAL, COGNITIVE, CASUAL)
        """
        try:
            # First check for crisis indicators
            if self._detect_crisis_indicators(user_input):
                return 'EMOTIONAL'  # Route crisis situations to empathy agent
            
            prompt = self.classification_prompt.format(user_input=user_input)
            response = self.llm.predict(prompt)
            intent = response.strip().upper()

            # Validate response
            valid_intents = ['FACTUAL', 'EMOTIONAL', 'COGNITIVE', 'CASUAL']
            if intent in valid_intents:
                return intent
            else:
                # Default to EMOTIONAL for safety if unclear
                return 'EMOTIONAL'
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return 'EMOTIONAL'  # Default to emotional support for safety
    
    def _detect_crisis_indicators(self, user_input: str) -> bool:
        """
        Detect crisis indicators in user input.
        
        Args:
            user_input: User's message
            
        Returns:
            True if crisis indicators detected
        """
        crisis_keywords = [
            'very low', 'depressed', 'hopeless', 'can\'t cope', 'overwhelmed',
            'suicidal', 'end my life', 'kill myself', 'want to die',
            'can\'t adjust', 'social isolation', 'lonely', 'isolated',
            'crisis', 'emergency', 'help me', 'desperate'
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in crisis_keywords)

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
