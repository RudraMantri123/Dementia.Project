"""Adaptive response generation based on user profiles."""

from typing import Dict, Any, Optional
from src.personalization.user_profile_manager import UserProfileManager


class AdaptiveResponseGenerator:
    """Generates personalized responses based on user profile."""

    def __init__(self):
        """Initialize adaptive response generator."""
        self.profile_manager = UserProfileManager()

    def adapt_response(
        self,
        response: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Adapt response based on user profile and preferences.

        Args:
            response: Original response from agent
            user_id: User identifier
            context: Additional context

        Returns:
            Adapted response
        """
        settings = self.profile_manager.get_personalized_settings(user_id)

        # Get preferences
        preferences = settings.get('interaction_preferences', {})
        response_length = preferences.get('response_length', 'medium')
        dementia_stage = settings.get('dementia_stage', 'early')
        cognitive_level = settings.get('cognitive_level', 0.5)

        # Apply adaptations
        adapted_response = response

        # Adjust for response length preference
        if response_length == 'short' and len(response) > 300:
            adapted_response = self._shorten_response(response)
        elif response_length == 'long' and len(response) < 200:
            adapted_response = self._elaborate_response(response)

        # Adjust for cognitive level
        if cognitive_level < 0.4 or dementia_stage == 'advanced':
            adapted_response = self._simplify_for_cognitive_level(adapted_response)

        # Add personalization
        user_name = settings.get('name')
        if user_name and context and context.get('use_name', False):
            adapted_response = f"{user_name}, {adapted_response}"

        return adapted_response

    def _shorten_response(self, response: str) -> str:
        """Shorten response while maintaining key information."""
        sentences = response.split('. ')
        if len(sentences) <= 2:
            return response

        # Keep first sentence and key information
        shortened = sentences[0]
        if not shortened.endswith('.'):
            shortened += '.'

        # Add most important follow-up if needed
        if len(sentences) > 1:
            # Heuristic: sentences with keywords are more important
            important_keywords = ['important', 'remember', 'should', 'must', 'key', 'essential']
            for sentence in sentences[1:3]:
                if any(keyword in sentence.lower() for keyword in important_keywords):
                    shortened += ' ' + sentence + '.'
                    break

        return shortened

    def _elaborate_response(self, response: str) -> str:
        """Add elaboration to short responses."""
        # Add supportive follow-up
        elaborations = [
            " I'm here to help you with any questions you might have about this topic.",
            " Feel free to ask me if you'd like more details or clarification.",
            " Would you like me to explain anything in more detail?"
        ]

        # Choose elaboration based on response content
        if '?' not in response:
            return response + elaborations[0]
        else:
            return response

    def _simplify_for_cognitive_level(self, response: str) -> str:
        """
        Simplify language for users with lower cognitive levels.

        Strategies:
        - Use shorter sentences
        - Replace complex words
        - Add clear structure
        """
        # Split into sentences
        sentences = response.split('. ')

        simplified_sentences = []
        for sentence in sentences:
            # Replace complex terms with simpler ones
            simplified = self._replace_complex_terms(sentence)

            # Break long sentences
            if len(simplified.split()) > 15:
                simplified = self._break_long_sentence(simplified)

            simplified_sentences.append(simplified)

        # Join with clear structure
        result = '. '.join(simplified_sentences)

        # Ensure proper punctuation
        if not result.endswith('.'):
            result += '.'

        return result

    def _replace_complex_terms(self, text: str) -> str:
        """Replace complex medical terms with simpler alternatives."""
        replacements = {
            'cognitive decline': 'memory changes',
            'neurological': 'brain-related',
            'pharmaceutical': 'medicine',
            'diagnosis': 'finding out what's wrong',
            'symptoms': 'signs',
            'progression': 'changes over time',
            'dementia': 'memory problems',
            'Alzheimer\'s': 'a type of memory problem',
            'caregiver': 'person who helps',
            'therapeutic': 'helpful',
            'intervention': 'help',
            'assessment': 'check-up'
        }

        result = text
        for complex_term, simple_term in replacements.items():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(complex_term), re.IGNORECASE)
            result = pattern.sub(simple_term, result)

        return result

    def _break_long_sentence(self, sentence: str) -> str:
        """Break long sentences into shorter ones."""
        # Split on conjunctions
        for conjunction in [' and ', ' but ', ' or ', ' so ']:
            if conjunction in sentence:
                parts = sentence.split(conjunction, 1)
                return f"{parts[0]}. {parts[1].capitalize()}"

        return sentence

    def get_response_guidelines(self, user_id: str) -> Dict[str, Any]:
        """
        Get response generation guidelines for user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with guidelines
        """
        settings = self.profile_manager.get_personalized_settings(user_id)

        return {
            'response_length': settings.get('interaction_preferences', {}).get('response_length', 'medium'),
            'simplification_level': self._get_simplification_level(settings['cognitive_level']),
            'use_name': settings.get('name') is not None,
            'preferred_topics': settings.get('interaction_preferences', {}).get('preferred_topics', []),
            'dementia_stage': settings.get('dementia_stage', 'early'),
            'recommendations': self._get_response_recommendations(settings)
        }

    def _get_simplification_level(self, cognitive_level: float) -> str:
        """Get appropriate simplification level."""
        if cognitive_level >= 0.7:
            return 'minimal'
        elif cognitive_level >= 0.4:
            return 'moderate'
        else:
            return 'high'

    def _get_response_recommendations(self, settings: Dict[str, Any]) -> List[str]:
        """Get recommendations for response generation."""
        recommendations = []

        cognitive_level = settings.get('cognitive_level', 0.5)
        dementia_stage = settings.get('dementia_stage', 'early')

        if cognitive_level < 0.4:
            recommendations.append("Use simple, clear language")
            recommendations.append("Keep sentences short (under 15 words)")
            recommendations.append("Avoid medical jargon")

        if dementia_stage == 'advanced':
            recommendations.append("Use familiar, concrete examples")
            recommendations.append("Repeat key information")
            recommendations.append("Provide reassurance")

        preferred_topics = settings.get('interaction_preferences', {}).get('preferred_topics', [])
        if preferred_topics:
            recommendations.append(f"Reference familiar topics: {', '.join(preferred_topics[:3])}")

        return recommendations
