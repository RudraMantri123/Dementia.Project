"""Reinforcement Learning from Human Feedback (RLHF) system."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pickle

from src.database.models import FeedbackLog, Conversation
from src.database.connection import get_db_manager


class RLHFSystem:
    """Reinforcement learning system for continuous improvement from feedback."""

    def __init__(self, reward_model_path: str = "data/models/reward_model.pkl"):
        """
        Initialize RLHF system.

        Args:
            reward_model_path: Path to reward model
        """
        self.reward_model_path = reward_model_path
        self.reward_model = None
        self.db_manager = get_db_manager()

    def collect_feedback(
        self,
        user_id: str,
        session_id: str,
        conversation_id: int,
        feedback_type: str,
        rating: Optional[int] = None,
        helpful: Optional[bool] = None,
        correction: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect user feedback on agent responses.

        Args:
            user_id: User identifier
            session_id: Session identifier
            conversation_id: Conversation ID
            feedback_type: Type of feedback (rating, correction, flag)
            rating: Rating (1-5 stars)
            helpful: Boolean helpfulness indicator
            correction: User-provided correction
            notes: Additional notes

        Returns:
            Feedback receipt confirmation
        """
        with self.db_manager.get_session() as session:
            # Get conversation details
            conversation = session.query(Conversation).filter_by(
                id=conversation_id
            ).first()

            if not conversation:
                return {'error': 'Conversation not found'}

            # Create feedback entry
            feedback = FeedbackLog(
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
                feedback_type=feedback_type,
                rating=rating,
                helpful=helpful,
                agent_used=conversation.agent_used,
                query=conversation.content if conversation.role == 'user' else None,
                response=conversation.content if conversation.role == 'assistant' else None,
                correction_provided=correction,
                notes=notes
            )

            session.add(feedback)
            session.commit()
            session.refresh(feedback)

            return {
                'status': 'success',
                'feedback_id': feedback.id,
                'timestamp': feedback.timestamp.isoformat()
            }

    def analyze_feedback_patterns(
        self,
        agent_name: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze feedback patterns to identify improvement areas.

        Args:
            agent_name: Filter by specific agent
            days: Number of days to analyze

        Returns:
            Analysis results
        """
        since_date = datetime.utcnow() - timedelta(days=days)

        with self.db_manager.get_session() as session:
            query = session.query(FeedbackLog).filter(
                FeedbackLog.timestamp >= since_date
            )

            if agent_name:
                query = query.filter(FeedbackLog.agent_used == agent_name)

            feedback_items = query.all()

            if not feedback_items:
                return {'message': 'No feedback data available'}

            # Analyze ratings
            ratings = [f.rating for f in feedback_items if f.rating is not None]
            helpful_responses = [f.helpful for f in feedback_items if f.helpful is not None]

            # Agent performance
            agent_stats = defaultdict(lambda: {'ratings': [], 'helpful': []})

            for feedback in feedback_items:
                agent = feedback.agent_used or 'unknown'

                if feedback.rating is not None:
                    agent_stats[agent]['ratings'].append(feedback.rating)

                if feedback.helpful is not None:
                    agent_stats[agent]['helpful'].append(feedback.helpful)

            # Compute metrics per agent
            agent_metrics = {}
            for agent, stats in agent_stats.items():
                if stats['ratings']:
                    avg_rating = np.mean(stats['ratings'])
                    rating_std = np.std(stats['ratings'])
                else:
                    avg_rating = None
                    rating_std = None

                if stats['helpful']:
                    helpful_ratio = sum(stats['helpful']) / len(stats['helpful'])
                else:
                    helpful_ratio = None

                agent_metrics[agent] = {
                    'avg_rating': avg_rating,
                    'rating_std': rating_std,
                    'helpful_ratio': helpful_ratio,
                    'total_feedback': len(stats['ratings']) + len(stats['helpful'])
                }

            # Identify improvement areas
            improvement_areas = []

            for agent, metrics in agent_metrics.items():
                if metrics['avg_rating'] and metrics['avg_rating'] < 3.5:
                    improvement_areas.append(f"{agent} agent has low ratings")

                if metrics['helpful_ratio'] and metrics['helpful_ratio'] < 0.6:
                    improvement_areas.append(f"{agent} agent has low helpfulness ratio")

            return {
                'date_range': {
                    'start': since_date.isoformat(),
                    'end': datetime.utcnow().isoformat()
                },
                'total_feedback': len(feedback_items),
                'overall_stats': {
                    'avg_rating': np.mean(ratings) if ratings else None,
                    'helpful_ratio': sum(helpful_responses) / len(helpful_responses) if helpful_responses else None
                },
                'agent_metrics': agent_metrics,
                'improvement_areas': improvement_areas
            }

    def get_corrections(
        self,
        agent_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get user-provided corrections for training.

        Args:
            agent_name: Filter by agent
            limit: Maximum number of corrections to return

        Returns:
            List of corrections
        """
        with self.db_manager.get_session() as session:
            query = session.query(FeedbackLog).filter(
                FeedbackLog.correction_provided.isnot(None)
            )

            if agent_name:
                query = query.filter(FeedbackLog.agent_used == agent_name)

            feedback_items = query.order_by(
                FeedbackLog.timestamp.desc()
            ).limit(limit).all()

            corrections = []

            for feedback in feedback_items:
                corrections.append({
                    'feedback_id': feedback.id,
                    'agent': feedback.agent_used,
                    'query': feedback.query,
                    'original_response': feedback.response,
                    'correction': feedback.correction_provided,
                    'timestamp': feedback.timestamp.isoformat()
                })

            return corrections

    def compute_reward(
        self,
        feedback: FeedbackLog
    ) -> float:
        """
        Compute reward value from feedback.

        Args:
            feedback: Feedback log entry

        Returns:
            Reward value (-1 to 1)
        """
        reward = 0.0

        # Rating-based reward
        if feedback.rating is not None:
            # Normalize rating (1-5) to (-1 to 1)
            reward += (feedback.rating - 3) / 2

        # Helpfulness-based reward
        if feedback.helpful is not None:
            reward += 0.5 if feedback.helpful else -0.5

        # Correction provided (indicates dissatisfaction)
        if feedback.correction_provided:
            reward -= 0.3

        # Normalize to [-1, 1]
        reward = np.clip(reward, -1.0, 1.0)

        return float(reward)

    def train_reward_model(self, min_samples: int = 100) -> Dict[str, Any]:
        """
        Train reward model from collected feedback.

        Args:
            min_samples: Minimum samples required for training

        Returns:
            Training results
        """
        with self.db_manager.get_session() as session:
            feedback_items = session.query(FeedbackLog).all()

            if len(feedback_items) < min_samples:
                return {
                    'error': f'Insufficient feedback samples. Need {min_samples}, have {len(feedback_items)}'
                }

            # Prepare training data
            rewards = []
            features = []

            for feedback in feedback_items:
                reward = self.compute_reward(feedback)
                rewards.append(reward)

                # Extract features (simplified)
                feature_vec = self._extract_feedback_features(feedback)
                features.append(feature_vec)

            # Train simple reward model (could be replaced with neural network)
            # For now, use statistical aggregation
            self.reward_model = {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'agent_rewards': self._compute_agent_rewards(feedback_items)
            }

            # Save model
            self._save_reward_model()

            return {
                'status': 'success',
                'samples_used': len(feedback_items),
                'mean_reward': self.reward_model['mean_reward'],
                'std_reward': self.reward_model['std_reward']
            }

    def get_improvement_suggestions(
        self,
        agent_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get specific improvement suggestions for an agent.

        Args:
            agent_name: Agent to analyze

        Returns:
            List of improvement suggestions
        """
        feedback_analysis = self.analyze_feedback_patterns(agent_name=agent_name)

        if 'message' in feedback_analysis:
            return []

        suggestions = []

        agent_metrics = feedback_analysis.get('agent_metrics', {}).get(agent_name, {})

        # Low rating suggestions
        avg_rating = agent_metrics.get('avg_rating')
        if avg_rating and avg_rating < 3.5:
            suggestions.append({
                'priority': 'high',
                'category': 'response_quality',
                'suggestion': f'Improve response quality. Current avg rating: {avg_rating:.2f}/5',
                'action': 'Review low-rated responses and identify common issues'
            })

        # Low helpfulness suggestions
        helpful_ratio = agent_metrics.get('helpful_ratio')
        if helpful_ratio and helpful_ratio < 0.6:
            suggestions.append({
                'priority': 'high',
                'category': 'helpfulness',
                'suggestion': f'Increase helpfulness. Current ratio: {helpful_ratio:.1%}',
                'action': 'Make responses more actionable and specific'
            })

        # Get common correction patterns
        corrections = self.get_corrections(agent_name=agent_name, limit=20)

        if corrections:
            suggestions.append({
                'priority': 'medium',
                'category': 'accuracy',
                'suggestion': f'{len(corrections)} corrections provided by users',
                'action': 'Review corrections and update knowledge base'
            })

        return suggestions

    def _extract_feedback_features(self, feedback: FeedbackLog) -> List[float]:
        """Extract features from feedback for reward modeling."""
        features = []

        # Agent encoding (one-hot style)
        agent_types = ['knowledge', 'empathy', 'cognitive', 'analyst']
        agent_encoding = [1.0 if feedback.agent_used == a else 0.0 for a in agent_types]
        features.extend(agent_encoding)

        # Feedback type encoding
        feedback_types = ['rating', 'correction', 'flag']
        feedback_encoding = [1.0 if feedback.feedback_type == f else 0.0 for f in feedback_types]
        features.extend(feedback_encoding)

        # Numeric features
        features.append(feedback.rating / 5.0 if feedback.rating else 0.5)
        features.append(1.0 if feedback.helpful else 0.0 if feedback.helpful is False else 0.5)
        features.append(1.0 if feedback.correction_provided else 0.0)

        return features

    def _compute_agent_rewards(self, feedback_items: List[FeedbackLog]) -> Dict[str, float]:
        """Compute average reward per agent."""
        agent_rewards = defaultdict(list)

        for feedback in feedback_items:
            reward = self.compute_reward(feedback)
            agent = feedback.agent_used or 'unknown'
            agent_rewards[agent].append(reward)

        return {
            agent: float(np.mean(rewards))
            for agent, rewards in agent_rewards.items()
        }

    def _save_reward_model(self):
        """Save reward model to disk."""
        import os
        os.makedirs(os.path.dirname(self.reward_model_path), exist_ok=True)

        with open(self.reward_model_path, 'wb') as f:
            pickle.dump(self.reward_model, f)

    def _load_reward_model(self) -> bool:
        """Load reward model from disk."""
        try:
            with open(self.reward_model_path, 'rb') as f:
                self.reward_model = pickle.load(f)
            return True
        except FileNotFoundError:
            return False
