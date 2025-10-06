"""Empathy agent for emotional support and casual conversation."""

from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent


class EmpathyAgent(BaseAgent):
    """Provides emotional support and handles casual conversation."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the empathy agent.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model
        """
        super().__init__(api_key, model_name, temperature=0.8)

        self.prompt_template = """You are a warm, compassionate, and empathetic support companion for people
dealing with dementia - either as patients or caregivers. Your role is to provide emotional support,
validation, and comfort.

IMPORTANT GUIDELINES:
- Show genuine empathy and understanding
- Validate feelings without judgment
- Offer comfort and encouragement
- Be warm and conversational
- Share hope while being realistic
- Encourage self-care for caregivers
- Remind users they're not alone
- Keep responses personal and caring
- When appropriate, gently suggest seeking additional support (support groups, counseling)

Recent conversation context:
{history}

User: {user_input}

Compassionate Response:"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["history", "user_input"]
        )

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process emotional input with empathy.

        Args:
            user_input: User's message
            context: Optional context information

        Returns:
            Dictionary with empathetic response
        """
        self.add_to_history('user', user_input)

        # Get recent conversation context
        history = self.get_history(last_n=6)
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in history[:-1]
        ])

        try:
            # Generate empathetic response
            prompt = self.PROMPT.format(
                history=history_text,
                user_input=user_input
            )
            response_text = self.llm.predict(prompt)

            self.add_to_history('assistant', response_text)

            return {
                'response': response_text,
                'agent': 'empathy',
                'emotion_detected': self._detect_emotion(user_input)
            }

        except Exception as e:
            error_msg = "I'm here for you. Please tell me more about what you're experiencing."
            self.add_to_history('assistant', error_msg)
            return {
                'response': error_msg,
                'agent': 'empathy',
                'error': str(e)
            }

    def _detect_emotion(self, text: str) -> str:
        """
        Enhanced emotion detection with comprehensive keyword patterns.

        Args:
            text: Input text

        Returns:
            Detected emotion category
        """
        text_lower = text.lower()

        # Enhanced keyword-based emotion detection
        stressed_keywords = [
            'overwhelmed', 'stressed', 'tired', 'exhausted', "can't cope",
            'burning out', 'breaking point', 'drowning', 'crushing',
            'drained', 'suffocating', 'burden', 'pressure', 'trapped',
            'no energy', 'stretched thin', 'breaking me', 'at my limit'
        ]

        sad_keywords = [
            'sad', 'depressed', 'crying', 'lonely', 'miss', 'heartbreak',
            'grief', 'mourning', 'lost', 'empty', 'hopeless', 'devastat',
            'broken heart', 'tears', 'sorrow', 'despair', 'anguish',
            'heartbroken', 'grieving', 'heartache'
        ]

        frustrated_keywords = [
            'angry', 'frustrated', 'unfair', 'hate', 'mad', 'annoyed',
            'irritated', 'rage', 'furious', "won't cooperate", 'stubborn',
            'system is broken', 'nothing works', 'why is this', 'fed up',
            "won't listen", 'maddening', 'useless'
        ]

        anxious_keywords = [
            'scared', 'afraid', 'worried', 'anxiety', 'fear', 'terrified',
            'panic', 'nervous', 'anxious', 'what if', 'on edge',
            'paranoid', 'dread', 'uneasy', "can't sleep", 'worry',
            'frightened', 'apprehensive', 'concerned'
        ]

        positive_keywords = [
            'happy', 'good', 'better', 'grateful', 'thankful', 'hopeful',
            'joy', 'blessed', 'appreciate', 'proud', 'wonderful',
            'encouraged', 'optimistic', 'stronger', 'peaceful', 'love',
            'thank you', 'helped', 'working', 'improvement', 'victory'
        ]

        # Count matches for each emotion
        stressed_count = sum(1 for kw in stressed_keywords if kw in text_lower)
        sad_count = sum(1 for kw in sad_keywords if kw in text_lower)
        frustrated_count = sum(1 for kw in frustrated_keywords if kw in text_lower)
        anxious_count = sum(1 for kw in anxious_keywords if kw in text_lower)
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)

        # Find emotion with highest match count
        emotion_counts = {
            'stressed': stressed_count,
            'sad': sad_count,
            'frustrated': frustrated_count,
            'anxious': anxious_count,
            'positive': positive_count
        }

        max_count = max(emotion_counts.values())
        if max_count > 0:
            return max(emotion_counts, key=emotion_counts.get)
        else:
            return 'neutral'
