"""Enhanced Cognitive Agent with fine-tuned model support."""

from typing import Dict, Any, Optional
import os
import json
from .cognitive_agent import CognitiveAgent
import openai


class CognitiveAgentFineTuned(CognitiveAgent):
    """
    Enhanced cognitive agent that uses a fine-tuned model for exercise generation.

    This agent extends the base CognitiveAgent with:
    - Fine-tuned LLM for better exercise quality
    - Dynamic exercise generation
    - Improved personalization
    - Quality evaluation
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        fine_tuned_model: str = None,
        use_fine_tuned: bool = True
    ):
        """
        Initialize the fine-tuned cognitive agent.

        Args:
            api_key: OpenAI API key
            model_name: Base model name (fallback)
            fine_tuned_model: Fine-tuned model ID
            use_fine_tuned: Whether to use fine-tuned model
        """
        super().__init__(api_key, model_name)

        self.fine_tuned_model = fine_tuned_model
        self.use_fine_tuned = use_fine_tuned and fine_tuned_model is not None

        # Load fine-tuned model info if available
        if not self.fine_tuned_model:
            self.fine_tuned_model = self._load_model_info()

        if self.use_fine_tuned and self.fine_tuned_model:
            print(f"✓ Using fine-tuned model: {self.fine_tuned_model}")
        else:
            print(f"⚠ Using base model: {model_name}")

    def _load_model_info(self) -> Optional[str]:
        """Load fine-tuned model ID from saved info."""
        info_file = "data/fine_tuning/model_info.json"

        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                return info.get('fine_tuned_model')
            except Exception as e:
                print(f"Warning: Could not load model info: {e}")

        return None

    def _generate_exercise_with_llm(
        self,
        exercise_type: str,
        difficulty: str = "medium",
        theme: str = None,
        personalization: Dict = None
    ) -> Dict[str, Any]:
        """
        Generate exercise using fine-tuned LLM.

        Args:
            exercise_type: Type of exercise (memory_list, story_recall, etc.)
            difficulty: Difficulty level (easy, medium, hard)
            theme: Optional theme for the exercise
            personalization: Optional personalization parameters

        Returns:
            Generated exercise dict
        """
        # Determine which model to use
        model = self.fine_tuned_model if self.use_fine_tuned else self.model_name

        # Build prompt based on exercise type
        prompt = self._build_generation_prompt(
            exercise_type,
            difficulty,
            theme,
            personalization
        )

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert cognitive therapist specializing in dementia care.
Generate engaging, appropriate, and therapeutically valuable cognitive exercises for individuals with dementia.
Your exercises should be: clear, age-appropriate, culturally sensitive, encouraging, and designed to support memory, attention, and cognitive function."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )

            exercise_text = response.choices[0].message.content

            return {
                'response': exercise_text,
                'agent': 'cognitive',
                'exercise_type': exercise_type,
                'generated_by': 'fine_tuned_model' if self.use_fine_tuned else 'base_model',
                'model_used': model
            }

        except Exception as e:
            print(f"Error generating exercise with LLM: {e}")
            # Fallback to base implementation
            return self._fallback_exercise(exercise_type)

    def _build_generation_prompt(
        self,
        exercise_type: str,
        difficulty: str,
        theme: str,
        personalization: Dict
    ) -> str:
        """
        Build prompt for exercise generation.

        Args:
            exercise_type: Type of exercise
            difficulty: Difficulty level
            theme: Theme for exercise
            personalization: Personalization parameters

        Returns:
            Formatted prompt string
        """
        prompts = {
            'memory_list': f"Generate a memory list exercise about {theme or 'common household items'} for {difficulty} difficulty dementia patients. Include 4-6 items with emojis and memory tips.",

            'story_recall': f"Create a detailed story recall exercise about {theme or 'a daily activity'} suitable for someone with {difficulty}-stage dementia. Include specific names, times, colors, and numbers. The story should be 200-300 words.",

            'pattern_recognition': f"Design a pattern recognition exercise with {difficulty} difficulty. Use visual elements or numbers. Include hints and encouraging language.",

            'word_association': f"Create a word association exercise using the theme '{theme or 'family'}'. Include prompts for memories and related words.",

            'orientation': f"Generate an orientation exercise focusing on {theme or 'time awareness'}. Include 4-5 questions that help maintain awareness."
        }

        base_prompt = prompts.get(exercise_type, f"Generate a {exercise_type} exercise for dementia patients")

        # Add personalization if provided
        if personalization:
            if 'interests' in personalization:
                base_prompt += f"\n\nPatient interests: {', '.join(personalization['interests'])}"
            if 'cognitive_level' in personalization:
                base_prompt += f"\n\nCognitive level: {personalization['cognitive_level']}/10"

        return base_prompt

    def _fallback_exercise(self, exercise_type: str) -> Dict[str, Any]:
        """Fallback to base implementation if LLM generation fails."""
        exercise_methods = {
            'memory_list': self._memory_list_exercise,
            'story_recall': self._story_recall_exercise,
            'pattern_recognition': self._pattern_recognition_exercise,
            'word_association': self._word_association_exercise,
            'orientation': self._orientation_exercise
        }

        method = exercise_methods.get(exercise_type, self._memory_list_exercise)
        return method()

    def generate_personalized_exercise(
        self,
        exercise_type: str = None,
        user_profile: Dict = None,
        difficulty: str = "medium",
        theme: str = None
    ) -> Dict[str, Any]:
        """
        Generate personalized exercise based on user profile.

        Args:
            exercise_type: Type of exercise (auto-selected if None)
            user_profile: User profile with preferences and history
            difficulty: Difficulty level
            theme: Optional theme

        Returns:
            Generated exercise
        """
        # Auto-select exercise type if not specified
        if not exercise_type:
            exercise_type = self._select_exercise_type(user_profile)

        # Extract personalization info
        personalization = {}
        if user_profile:
            personalization = {
                'interests': user_profile.get('interests', []),
                'cognitive_level': user_profile.get('cognitive_level', 5),
                'previous_exercises': user_profile.get('exercise_history', [])
            }

        # Generate with fine-tuned model if available
        if self.use_fine_tuned:
            return self._generate_exercise_with_llm(
                exercise_type,
                difficulty,
                theme,
                personalization
            )
        else:
            # Use base implementation
            return self._fallback_exercise(exercise_type)

    def _select_exercise_type(self, user_profile: Dict = None) -> str:
        """
        Intelligently select exercise type based on user profile.

        Args:
            user_profile: User profile with history

        Returns:
            Selected exercise type
        """
        import random

        available_types = [
            'memory_list',
            'story_recall',
            'pattern_recognition',
            'word_association',
            'orientation'
        ]

        if not user_profile or 'exercise_history' not in user_profile:
            return random.choice(available_types)

        # Avoid repeating recent exercises
        recent_exercises = user_profile.get('exercise_history', [])[-3:]
        recent_types = [ex.get('type') for ex in recent_exercises]

        # Prefer types not recently used
        preferred_types = [t for t in available_types if t not in recent_types]

        if preferred_types:
            return random.choice(preferred_types)
        else:
            return random.choice(available_types)

    def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process request with enhanced fine-tuned capabilities.

        Args:
            user_input: User's request
            context: Optional context with exercise state and user profile

        Returns:
            Dictionary with exercise and response
        """
        # Check if continuing an active exercise
        if context and context.get('exercise_state'):
            return self._continue_exercise(user_input, context['exercise_state'])

        # Check if user profile is provided for personalization
        user_profile = context.get('user_profile') if context else None

        # Detect if user is requesting specific type or theme
        exercise_type, theme = self._parse_user_request(user_input)

        # Generate personalized exercise
        if self.use_fine_tuned and (exercise_type or theme):
            difficulty = self._detect_difficulty(user_input, user_profile)

            return self.generate_personalized_exercise(
                exercise_type=exercise_type,
                user_profile=user_profile,
                difficulty=difficulty,
                theme=theme
            )
        else:
            # Use base implementation
            return self._start_new_exercise(user_input)

    def _parse_user_request(self, user_input: str) -> tuple:
        """
        Parse user request for exercise type and theme.

        Returns:
            Tuple of (exercise_type, theme)
        """
        user_lower = user_input.lower()

        # Detect exercise type
        exercise_type = None
        if 'memory' in user_lower or 'list' in user_lower:
            exercise_type = 'memory_list'
        elif 'story' in user_lower:
            exercise_type = 'story_recall'
        elif 'pattern' in user_lower:
            exercise_type = 'pattern_recognition'
        elif 'word' in user_lower or 'association' in user_lower:
            exercise_type = 'word_association'
        elif 'orientation' in user_lower:
            exercise_type = 'orientation'

        # Detect theme
        theme = None
        theme_keywords = {
            'food': ['food', 'meal', 'cooking', 'grocery'],
            'family': ['family', 'relatives', 'children'],
            'nature': ['nature', 'garden', 'flowers', 'animals'],
            'music': ['music', 'song', 'instrument'],
            'travel': ['travel', 'vacation', 'trip']
        }

        for theme_name, keywords in theme_keywords.items():
            if any(kw in user_lower for kw in keywords):
                theme = theme_name
                break

        return exercise_type, theme

    def _detect_difficulty(self, user_input: str, user_profile: Dict = None) -> str:
        """Detect difficulty level from user input or profile."""
        user_lower = user_input.lower()

        if 'easy' in user_lower or 'simple' in user_lower:
            return 'easy'
        elif 'hard' in user_lower or 'difficult' in user_lower or 'challenging' in user_lower:
            return 'hard'
        elif user_profile and 'cognitive_level' in user_profile:
            level = user_profile['cognitive_level']
            if level >= 7:
                return 'easy'
            elif level >= 4:
                return 'medium'
            else:
                return 'hard'

        return 'medium'
