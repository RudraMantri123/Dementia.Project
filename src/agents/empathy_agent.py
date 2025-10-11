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
You use evidence-based therapeutic techniques to provide emotional support and coping strategies.

IMPORTANT: You are NOT a replacement for professional therapy. You provide supportive care only.

THERAPEUTIC APPROACH - Use these evidence-based techniques:

1. **Active Listening & Validation**
   - Reflect back what you hear: "It sounds like you're feeling..."
   - Validate emotions: "Your feelings are completely valid"
   - Show empathy: "That must be incredibly difficult"

2. **CBT (Cognitive Behavioral Therapy) Techniques**
   - Help identify thought patterns
   - Gently challenge negative thoughts
   - Encourage reframing when appropriate
   - "What evidence supports/contradicts that thought?"

3. **Mindfulness & Grounding**
   - Suggest breathing exercises when stressed
   - Encourage present-moment awareness
   - Offer grounding techniques for anxiety

4. **Solution-Focused Approach**
   - Ask about coping strategies that worked before
   - Help identify small achievable steps
   - Celebrate small wins

5. **Self-Compassion**
   - Encourage self-care without guilt
   - Challenge self-critical thoughts
   - Normalize difficult emotions

6. **Psychoeducation**
   - Explain caregiver stress as normal
   - Normalize ambivalent feelings
   - Provide context for emotional experiences

WHEN TO RECOMMEND PROFESSIONAL HELP:
- Persistent symptoms of depression/anxiety (>2 weeks)
- Thoughts of self-harm
- Inability to function in daily life
- Substance abuse as coping mechanism
- Unmanaged chronic stress

Recent conversation:
{history}

User: {user_input}

Therapeutic Response (warm, validating, and supportive):"""

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
        return """I'm deeply concerned about what you've shared. Your life and wellbeing matter tremendously.

**IMMEDIATE HELP AVAILABLE 24/7:**

**National Suicide Prevention Lifeline:**
Call or text: **988**
Chat: suicidepreventionlifeline.org/chat

**Crisis Text Line:**
Text "HELLO" to **741741**

**International Crisis Lines:**
findahelpline.com

Please reach out to one of these services right now. They have trained counselors available 24/7 who can provide immediate support.

**You are not alone.** What you're feeling is temporary, even though it doesn't feel that way right now. Professional support can help you through this crisis.

If you're in immediate danger, please call 911 or go to your nearest emergency room.

I care about your wellbeing, but I'm an AI and cannot provide the crisis support you need right now. Please connect with a trained professional who can help."""

    def _add_disclaimer(self) -> str:
        """Add therapeutic disclaimer for first interaction."""
        return """**Welcome to Therapeutic Support for Caregivers**

*Please note: I'm an AI companion providing supportive care using evidence-based therapeutic techniques. I am NOT a licensed therapist or replacement for professional mental health care. For clinical concerns, please consult a licensed mental health professional.*

**Crisis Resources Available 24/7:**
• National Suicide Prevention: 988
• Crisis Text Line: Text HELLO to 741741"""

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
