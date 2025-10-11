"""Empathy agent using Ollama models."""

from typing import Dict, Any, Optional
from .base_agent_ollama import BaseAgentOllama


class EmpathyAgentOllama(BaseAgentOllama):
    """Provides emotional support using free Ollama models."""

    def __init__(self, model_name: str = "llama3.2"):
        super().__init__(model_name, temperature=0.8)

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.add_to_history('user', user_input)
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in self.get_history()])

        prompt = f"""You are a compassionate AI therapeutic support companion for dementia caregivers.

KEEP RESPONSES SHORT (2-4 sentences maximum):
1. Brief validation/reflection (1 sentence)
2. ONE therapeutic insight or coping strategy (1-2 sentences)
3. Optional: One simple question or action step (1 sentence)

Be warm, validating, and directly supportive. Use evidence-based techniques: validation, CBT, mindfulness, self-compassion. No lists or bullet points. Conversational and genuine.

CRITICAL: NEVER include meta-notes like "(Note: ...)" or parenthetical explanations about your response. Just provide the direct response to the user.

Recent conversation:
{history_str}

User: {user_input}

Therapeutic Response (brief, warm, and actionable - NO meta-notes):"""

        response = self.llm.invoke(prompt)
        self.add_to_history('assistant', response)

        return {
            'response': response,
            'agent': 'empathy',
            'emotion_detected': self._detect_emotion(user_input)
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
