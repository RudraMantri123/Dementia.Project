"""Therapeutic support agent for emotional wellness and caregiver support."""

from typing import Dict, Any, Optional, List
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent


class EmpathyAgent(BaseAgent):
    """Provides therapeutic emotional support using evidence-based techniques."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the therapeutic support agent.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model
        """
        super().__init__(api_key, model_name, temperature=0.8)

        # Crisis keywords for immediate intervention
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself',
            'self-harm', 'no reason to live', 'better off dead', 'ending it all'
        ]

        self.prompt_template = """You are a compassionate AI therapeutic support companion for dementia caregivers.

CORE PRINCIPLES:
- Keep responses SHORT (2-4 sentences maximum)
- Be warm, validating, and directly supportive
- Use evidence-based techniques: validation, CBT, mindfulness, self-compassion
- Focus on ONE key insight or suggestion per response

RESPONSE STRUCTURE:
1. Brief validation/reflection (1 sentence)
2. ONE therapeutic insight or coping strategy (1-2 sentences)
3. Optional: One simple question or action step (1 sentence)

STYLE:
- Conversational and genuine
- Avoid lengthy explanations
- No lists or bullet points
- Direct and actionable
- NEVER include meta-notes like "(Note: ...)" or explanations about your response
- NEVER add parenthetical commentary about what you're doing
- Just provide the direct response to the user

Recent conversation:
{history}

User: {user_input}

Therapeutic Response (brief, warm, and actionable - NO meta-notes or explanations):"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["history", "user_input"]
        )

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process emotional input with therapeutic support.

        Args:
            user_input: User's message
            context: Optional context information

        Returns:
            Dictionary with therapeutic response
        """
        self.add_to_history('user', user_input)

        # Check for crisis situation
        is_crisis = self._detect_crisis(user_input)

        # Get recent conversation context
        history = self.get_history(last_n=6)
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in history[:-1]
        ])

        try:
            # If crisis detected, provide immediate resources
            if is_crisis:
                response_text = self._get_crisis_response()
            else:
                # Generate therapeutic response
                prompt = self.PROMPT.format(
                    history=history_text,
                    user_input=user_input
                )
                response_text = self.llm.predict(prompt)

                # Add therapeutic disclaimer at first interaction
                if len(history) <= 1:
                    response_text = self._add_disclaimer() + "\n\n" + response_text

            self.add_to_history('assistant', response_text)

            return {
                'response': response_text,
                'agent': 'empathy_therapeutic',
                'emotion_detected': self._detect_emotion(user_input),
                'crisis_detected': is_crisis,
                'therapeutic_techniques': self._suggest_techniques(user_input)
            }

        except Exception as e:
            error_msg = "I'm here to support you. Please tell me more about what you're experiencing."
            self.add_to_history('assistant', error_msg)
            return {
                'response': error_msg,
                'agent': 'empathy_therapeutic',
                'error': str(e)
            }

    def _detect_crisis(self, text: str) -> bool:
        """
        Detect crisis situations requiring immediate professional help.

        Args:
            text: User input text

        Returns:
            True if crisis keywords detected
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)

    def _get_crisis_response(self) -> str:
        """
        Provide immediate crisis resources and support.

        Returns:
            Crisis intervention response with resources
        """
        return """I'm deeply concerned about what you've shared. Please reach out for immediate professional help:

**National Suicide Prevention: 988** (call or text 24/7)
**Crisis Text Line: Text HELLO to 741741**

If you're in immediate danger, call 911 or go to your nearest emergency room.

You are not alone. What you're feeling is temporary. Trained counselors are available right now to help you through this."""

    def _add_disclaimer(self) -> str:
        """Add therapeutic disclaimer for first interaction."""
        return """*Note: I'm an AI support companion, not a licensed therapist. For professional mental health care, please consult a licensed professional. Crisis help: 988 (24/7)*"""

    def _suggest_techniques(self, text: str) -> List[str]:
        """
        Suggest appropriate therapeutic techniques based on emotional state.

        Args:
            text: User input

        Returns:
            List of applicable therapeutic techniques
        """
        emotion = self._detect_emotion(text)
        techniques = []

        if emotion == 'anxious':
            techniques = ['breathing_exercises', 'grounding', 'mindfulness']
        elif emotion == 'stressed':
            techniques = ['self_compassion', 'boundary_setting', 'stress_management']
        elif emotion == 'sad':
            techniques = ['validation', 'grief_support', 'self_care']
        elif emotion == 'frustrated':
            techniques = ['cognitive_reframing', 'problem_solving', 'emotional_expression']
        else:
            techniques = ['active_listening', 'validation']

        return techniques

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
