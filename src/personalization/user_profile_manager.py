"""User profile management and learning system."""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import Counter
import json

from src.database.models import UserProfile, Conversation, CognitiveExerciseResult
from src.database.connection import get_db_manager


class UserProfileManager:
    """Manages user profiles and learns from interaction patterns."""

    def __init__(self):
        """Initialize profile manager."""
        self.db_manager = get_db_manager()

    def create_profile(
        self,
        user_id: str,
        name: str = None,
        age: int = None,
        dementia_stage: str = "early",
        **kwargs
    ) -> UserProfile:
        """
        Create a new user profile.

        Args:
            user_id: Unique user identifier
            name: User's name
            age: User's age
            dementia_stage: Stage of dementia (early, moderate, advanced)
            **kwargs: Additional profile data

        Returns:
            Created UserProfile instance
        """
        with self.db_manager.get_session() as session:
            profile = UserProfile(
                user_id=user_id,
                name=name,
                age=age,
                dementia_stage=dementia_stage,
                interaction_preferences={
                    'voice_enabled': False,
                    'response_length': 'medium',  # short, medium, long
                    'preferred_topics': [],
                    'preferred_time': None,
                    'exercise_difficulty': 3  # 1-5 scale
                },
                cognitive_level=0.5,  # Default to medium
                engagement_patterns={
                    'avg_session_length': 0,
                    'preferred_hours': [],
                    'interaction_frequency': 0
                },
                medical_conditions=kwargs.get('medical_conditions', []),
                medications=kwargs.get('medications', []),
                caregiver_info=kwargs.get('caregiver_info', {})
            )
            session.add(profile)
            session.commit()
            session.refresh(profile)
            return profile

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by ID.

        Args:
            user_id: User identifier

        Returns:
            UserProfile or None if not found
        """
        with self.db_manager.get_session() as session:
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()
            if profile:
                # Detach from session to avoid lazy loading issues
                session.expunge(profile)
            return profile

    def update_profile(self, user_id: str, **kwargs) -> Optional[UserProfile]:
        """
        Update user profile.

        Args:
            user_id: User identifier
            **kwargs: Fields to update

        Returns:
            Updated UserProfile or None
        """
        with self.db_manager.get_session() as session:
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()
            if profile:
                for key, value in kwargs.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                profile.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(profile)
                session.expunge(profile)
            return profile

    def update_last_interaction(self, user_id: str):
        """Update last interaction timestamp."""
        self.update_profile(user_id, last_interaction=datetime.utcnow())

    def learn_from_interactions(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze user interactions and update profile with learned preferences.

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Dictionary with learned insights
        """
        with self.db_manager.get_session() as session:
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()
            if not profile:
                return {}

            # Get recent conversations
            since_date = datetime.utcnow() - timedelta(days=days)
            conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.timestamp >= since_date
            ).all()

            if not conversations:
                return {'message': 'No recent conversations to analyze'}

            # Analyze patterns
            insights = self._analyze_interaction_patterns(conversations)

            # Update profile with learned preferences
            profile.interaction_preferences = {
                **profile.interaction_preferences,
                'preferred_topics': insights['preferred_topics'],
                'response_length': insights['preferred_response_length']
            }

            profile.engagement_patterns = {
                **profile.engagement_patterns,
                'avg_session_length': insights['avg_session_length'],
                'preferred_hours': insights['preferred_hours'],
                'interaction_frequency': insights['interaction_frequency']
            }

            # Update cognitive level based on exercise performance
            cognitive_level = self._calculate_cognitive_level(user_id, session)
            if cognitive_level is not None:
                profile.cognitive_level = cognitive_level

            profile.updated_at = datetime.utcnow()
            session.commit()

            return insights

    def _analyze_interaction_patterns(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """
        Analyze conversation patterns to extract preferences.

        Args:
            conversations: List of Conversation objects

        Returns:
            Dictionary with analyzed patterns
        """
        # Extract topics (intents)
        intents = [c.intent for c in conversations if c.intent]
        topic_counts = Counter(intents)
        preferred_topics = [topic for topic, _ in topic_counts.most_common(3)]

        # Analyze response lengths
        response_lengths = [len(c.content) for c in conversations if c.role == 'assistant']
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

        if avg_length < 200:
            preferred_response_length = 'short'
        elif avg_length < 500:
            preferred_response_length = 'medium'
        else:
            preferred_response_length = 'long'

        # Analyze timing patterns
        hours = [c.timestamp.hour for c in conversations]
        hour_counts = Counter(hours)
        preferred_hours = [hour for hour, _ in hour_counts.most_common(3)]

        # Calculate session metrics
        session_ids = set(c.session_id for c in conversations)
        avg_messages_per_session = len(conversations) / len(session_ids) if session_ids else 0

        # Calculate interaction frequency (messages per day)
        date_range = (max(c.timestamp for c in conversations) - min(c.timestamp for c in conversations)).days
        interaction_frequency = len(conversations) / max(date_range, 1)

        return {
            'preferred_topics': preferred_topics,
            'preferred_response_length': preferred_response_length,
            'preferred_hours': preferred_hours,
            'avg_session_length': avg_messages_per_session,
            'interaction_frequency': interaction_frequency,
            'total_interactions': len(conversations),
            'unique_sessions': len(session_ids)
        }

    def _calculate_cognitive_level(self, user_id: str, session) -> Optional[float]:
        """
        Calculate cognitive level based on exercise performance.

        Args:
            user_id: User identifier
            session: Database session

        Returns:
            Cognitive level (0.0-1.0) or None
        """
        # Get recent exercise results
        since_date = datetime.utcnow() - timedelta(days=30)
        exercises = session.query(CognitiveExerciseResult).filter(
            CognitiveExerciseResult.user_id == user_id,
            CognitiveExerciseResult.timestamp >= since_date
        ).all()

        if not exercises:
            return None

        # Calculate average performance
        scores = [e.performance_score for e in exercises if e.performance_score is not None]
        if not scores:
            return None

        avg_score = sum(scores) / len(scores)

        # Adjust for difficulty
        weighted_scores = []
        for exercise in exercises:
            if exercise.performance_score is not None:
                difficulty_weight = exercise.difficulty_level / 5.0
                weighted_score = exercise.performance_score * difficulty_weight
                weighted_scores.append(weighted_score)

        if weighted_scores:
            return sum(weighted_scores) / len(weighted_scores)

        return avg_score

    def get_personalized_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Get personalized settings for user interactions.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with personalized settings
        """
        profile = self.get_profile(user_id)
        if not profile:
            return self._get_default_settings()

        return {
            'user_id': user_id,
            'name': profile.name,
            'dementia_stage': profile.dementia_stage,
            'cognitive_level': profile.cognitive_level,
            'interaction_preferences': profile.interaction_preferences,
            'engagement_patterns': profile.engagement_patterns,
            'medical_conditions': profile.medical_conditions,
            'recommended_exercise_difficulty': self._get_recommended_difficulty(profile.cognitive_level)
        }

    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for new users."""
        return {
            'dementia_stage': 'early',
            'cognitive_level': 0.5,
            'interaction_preferences': {
                'voice_enabled': False,
                'response_length': 'medium',
                'preferred_topics': [],
                'exercise_difficulty': 3
            },
            'engagement_patterns': {},
            'medical_conditions': [],
            'recommended_exercise_difficulty': 3
        }

    def _get_recommended_difficulty(self, cognitive_level: float) -> int:
        """
        Get recommended exercise difficulty based on cognitive level.

        Args:
            cognitive_level: User's cognitive level (0.0-1.0)

        Returns:
            Difficulty level (1-5)
        """
        if cognitive_level >= 0.8:
            return 5
        elif cognitive_level >= 0.6:
            return 4
        elif cognitive_level >= 0.4:
            return 3
        elif cognitive_level >= 0.2:
            return 2
        else:
            return 1

    def get_user_statistics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive user statistics.

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Dictionary with user statistics
        """
        with self.db_manager.get_session() as session:
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()
            if not profile:
                return {}

            since_date = datetime.utcnow() - timedelta(days=days)

            # Conversation statistics
            conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.timestamp >= since_date
            ).all()

            # Exercise statistics
            exercises = session.query(CognitiveExerciseResult).filter(
                CognitiveExerciseResult.user_id == user_id,
                CognitiveExerciseResult.timestamp >= since_date
            ).all()

            # Agent usage
            agent_counts = Counter(c.agent_used for c in conversations if c.agent_used)

            # Sentiment distribution
            sentiment_counts = Counter(c.sentiment for c in conversations if c.sentiment)

            return {
                'profile': {
                    'name': profile.name,
                    'age': profile.age,
                    'dementia_stage': profile.dementia_stage,
                    'cognitive_level': profile.cognitive_level
                },
                'engagement': {
                    'total_conversations': len(conversations),
                    'total_exercises': len(exercises),
                    'exercises_completed': sum(1 for e in exercises if e.is_correct),
                    'last_interaction': profile.last_interaction.isoformat() if profile.last_interaction else None
                },
                'performance': {
                    'avg_exercise_score': sum(e.performance_score for e in exercises if e.performance_score) / len(exercises) if exercises else 0,
                    'cognitive_level': profile.cognitive_level
                },
                'preferences': {
                    'agent_distribution': dict(agent_counts),
                    'sentiment_distribution': dict(sentiment_counts),
                    'interaction_preferences': profile.interaction_preferences
                }
            }
