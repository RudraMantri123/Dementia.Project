"""Longitudinal trend analysis for tracking user progress over time."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from src.database.models import UserAnalytics, CognitiveExerciseResult, Conversation
from src.database.connection import get_db_manager


class LongitudinalTracker:
    """Tracks and analyzes user progress over time."""

    def __init__(self):
        """Initialize longitudinal tracker."""
        self.db_manager = get_db_manager()

    def track_cognitive_performance(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        weeks: int = 12
    ) -> Dict[str, Any]:
        """
        Track cognitive performance over time.

        Args:
            user_id: User identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            weeks: Number of weeks to analyze if dates not provided

        Returns:
            Dictionary with performance trends
        """
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(weeks=weeks)

        with self.db_manager.get_session() as session:
            # Get exercise results
            exercises = session.query(CognitiveExerciseResult).filter(
                CognitiveExerciseResult.user_id == user_id,
                CognitiveExerciseResult.timestamp >= start_date,
                CognitiveExerciseResult.timestamp <= end_date
            ).order_by(CognitiveExerciseResult.timestamp).all()

            if not exercises:
                return {'message': 'No exercise data available'}

            # Analyze trends
            weekly_data = self._aggregate_by_week(exercises)
            trend_analysis = self._analyze_trend(weekly_data)
            performance_categories = self._categorize_performance(exercises)

            return {
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_exercises': len(exercises),
                'weekly_data': weekly_data,
                'trend_analysis': trend_analysis,
                'performance_categories': performance_categories,
                'current_level': self._calculate_current_level(exercises[-10:] if len(exercises) >= 10 else exercises),
                'recommendations': self._generate_recommendations(trend_analysis, performance_categories)
            }

    def track_engagement_trends(
        self,
        user_id: str,
        weeks: int = 12
    ) -> Dict[str, Any]:
        """
        Track user engagement trends over time.

        Args:
            user_id: User identifier
            weeks: Number of weeks to analyze

        Returns:
            Dictionary with engagement trends
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(weeks=weeks)

        with self.db_manager.get_session() as session:
            # Get conversations
            conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.timestamp >= start_date,
                Conversation.timestamp <= end_date,
                Conversation.role == 'user'
            ).order_by(Conversation.timestamp).all()

            if not conversations:
                return {'message': 'No conversation data available'}

            # Analyze engagement
            weekly_engagement = self._calculate_weekly_engagement(conversations)
            trend = self._detect_engagement_trend(weekly_engagement)

            return {
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_interactions': len(conversations),
                'weekly_engagement': weekly_engagement,
                'engagement_trend': trend,
                'average_engagement_score': np.mean([w['engagement_score'] for w in weekly_engagement]),
                'consistency_score': self._calculate_consistency_score(weekly_engagement)
            }

    def track_sentiment_trends(
        self,
        user_id: str,
        weeks: int = 12
    ) -> Dict[str, Any]:
        """
        Track sentiment trends over time.

        Args:
            user_id: User identifier
            weeks: Number of weeks to analyze

        Returns:
            Dictionary with sentiment trends
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(weeks=weeks)

        with self.db_manager.get_session() as session:
            # Get conversations with sentiment
            conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.timestamp >= start_date,
                Conversation.timestamp <= end_date,
                Conversation.role == 'user',
                Conversation.sentiment_score.isnot(None)
            ).order_by(Conversation.timestamp).all()

            if not conversations:
                return {'message': 'No sentiment data available'}

            # Analyze sentiment trends
            weekly_sentiment = self._calculate_weekly_sentiment(conversations)
            trend = self._analyze_sentiment_trend(weekly_sentiment)

            # Detect significant changes
            change_points = self._detect_sentiment_changes(weekly_sentiment)

            return {
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'weekly_sentiment': weekly_sentiment,
                'sentiment_trend': trend,
                'average_sentiment': np.mean([w['avg_score'] for w in weekly_sentiment]),
                'volatility': np.std([w['avg_score'] for w in weekly_sentiment]),
                'significant_changes': change_points
            }

    def _aggregate_by_week(self, exercises: List[CognitiveExerciseResult]) -> List[Dict[str, Any]]:
        """Aggregate exercise data by week."""
        weekly_data = defaultdict(list)

        for exercise in exercises:
            week_start = exercise.timestamp - timedelta(days=exercise.timestamp.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            weekly_data[week_key].append(exercise)

        result = []
        for week_key in sorted(weekly_data.keys()):
            week_exercises = weekly_data[week_key]

            # Calculate metrics
            scores = [e.performance_score for e in week_exercises if e.performance_score is not None]
            completion_times = [e.completion_time_seconds for e in week_exercises if e.completion_time_seconds is not None]

            result.append({
                'week_start': week_key,
                'total_exercises': len(week_exercises),
                'avg_score': np.mean(scores) if scores else 0,
                'avg_completion_time': np.mean(completion_times) if completion_times else 0,
                'success_rate': sum(1 for e in week_exercises if e.is_correct) / len(week_exercises) if week_exercises else 0
            })

        return result

    def _analyze_trend(self, weekly_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trend in weekly data."""
        if len(weekly_data) < 3:
            return {'trend': 'insufficient_data'}

        scores = [w['avg_score'] for w in weekly_data]

        # Calculate linear regression
        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)
        slope = coeffs[0]

        # Determine trend
        if slope > 0.02:
            trend = 'improving'
        elif slope < -0.02:
            trend = 'declining'
        else:
            trend = 'stable'

        # Calculate confidence
        variance = np.var(scores)
        confidence = 'high' if variance < 0.05 else 'medium' if variance < 0.1 else 'low'

        return {
            'trend': trend,
            'slope': float(slope),
            'confidence': confidence,
            'variance': float(variance)
        }

    def _categorize_performance(self, exercises: List[CognitiveExerciseResult]) -> Dict[str, Any]:
        """Categorize performance by exercise type and difficulty."""
        by_type = defaultdict(list)
        by_difficulty = defaultdict(list)

        for exercise in exercises:
            if exercise.performance_score is not None:
                by_type[exercise.exercise_type].append(exercise.performance_score)
                by_difficulty[exercise.difficulty_level].append(exercise.performance_score)

        return {
            'by_type': {ex_type: {
                'avg_score': np.mean(scores),
                'count': len(scores)
            } for ex_type, scores in by_type.items()},
            'by_difficulty': {diff: {
                'avg_score': np.mean(scores),
                'count': len(scores)
            } for diff, scores in by_difficulty.items()}
        }

    def _calculate_current_level(self, recent_exercises: List[CognitiveExerciseResult]) -> Dict[str, Any]:
        """Calculate current cognitive level based on recent performance."""
        if not recent_exercises:
            return {'level': 'unknown'}

        scores = [e.performance_score for e in recent_exercises if e.performance_score is not None]

        if not scores:
            return {'level': 'unknown'}

        avg_score = np.mean(scores)

        if avg_score >= 0.8:
            level = 'high'
        elif avg_score >= 0.6:
            level = 'medium-high'
        elif avg_score >= 0.4:
            level = 'medium'
        elif avg_score >= 0.2:
            level = 'medium-low'
        else:
            level = 'low'

        return {
            'level': level,
            'score': float(avg_score),
            'based_on': len(scores)
        }

    def _generate_recommendations(
        self,
        trend_analysis: Dict[str, Any],
        performance_categories: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        trend = trend_analysis.get('trend')

        if trend == 'declining':
            recommendations.append("Performance appears to be declining. Consider consulting healthcare provider.")
            recommendations.append("Reduce difficulty level to maintain engagement.")
        elif trend == 'improving':
            recommendations.append("Excellent progress! Continue with current exercise routine.")
            recommendations.append("Consider gradually increasing difficulty level.")
        elif trend == 'stable':
            recommendations.append("Performance is stable. Maintain current routine.")

        # Type-specific recommendations
        by_type = performance_categories.get('by_type', {})
        for ex_type, data in by_type.items():
            if data['avg_score'] < 0.4:
                recommendations.append(f"Focus on improving {ex_type} exercises.")

        return recommendations

    def _calculate_weekly_engagement(self, conversations: List[Conversation]) -> List[Dict[str, Any]]:
        """Calculate engagement metrics by week."""
        weekly_data = defaultdict(list)

        for conv in conversations:
            week_start = conv.timestamp - timedelta(days=conv.timestamp.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            weekly_data[week_key].append(conv)

        result = []
        for week_key in sorted(weekly_data.keys()):
            week_convs = weekly_data[week_key]

            # Calculate engagement score
            avg_length = np.mean([len(c.content) for c in week_convs])
            message_count = len(week_convs)

            engagement_score = min((message_count / 10) * 0.5 + (avg_length / 200) * 0.5, 1.0)

            result.append({
                'week_start': week_key,
                'message_count': message_count,
                'avg_message_length': avg_length,
                'engagement_score': engagement_score
            })

        return result

    def _detect_engagement_trend(self, weekly_engagement: List[Dict[str, Any]]) -> str:
        """Detect trend in engagement."""
        if len(weekly_engagement) < 3:
            return 'insufficient_data'

        scores = [w['engagement_score'] for w in weekly_engagement]

        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)

        if coeffs[0] > 0.02:
            return 'increasing'
        elif coeffs[0] < -0.02:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_consistency_score(self, weekly_engagement: List[Dict[str, Any]]) -> float:
        """Calculate consistency score (lower variance = higher consistency)."""
        scores = [w['engagement_score'] for w in weekly_engagement]
        variance = np.var(scores)

        # Convert variance to consistency score (0-1, higher is better)
        consistency = max(0, 1 - variance * 2)

        return round(consistency, 3)

    def _calculate_weekly_sentiment(self, conversations: List[Conversation]) -> List[Dict[str, Any]]:
        """Calculate sentiment metrics by week."""
        weekly_data = defaultdict(list)

        for conv in conversations:
            week_start = conv.timestamp - timedelta(days=conv.timestamp.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            weekly_data[week_key].append(conv)

        result = []
        for week_key in sorted(weekly_data.keys()):
            week_convs = weekly_data[week_key]

            scores = [c.sentiment_score for c in week_convs if c.sentiment_score is not None]

            result.append({
                'week_start': week_key,
                'avg_score': np.mean(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'sample_size': len(scores)
            })

        return result

    def _analyze_sentiment_trend(self, weekly_sentiment: List[Dict[str, Any]]) -> str:
        """Analyze sentiment trend."""
        if len(weekly_sentiment) < 3:
            return 'insufficient_data'

        scores = [w['avg_score'] for w in weekly_sentiment]

        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)

        if coeffs[0] > 0.02:
            return 'improving'
        elif coeffs[0] < -0.02:
            return 'declining'
        else:
            return 'stable'

    def _detect_sentiment_changes(self, weekly_sentiment: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect significant changes in sentiment."""
        if len(weekly_sentiment) < 3:
            return []

        changes = []
        scores = [w['avg_score'] for w in weekly_sentiment]

        for i in range(1, len(scores)):
            change = scores[i] - scores[i-1]

            if abs(change) > 0.2:  # Significant change threshold
                changes.append({
                    'week': weekly_sentiment[i]['week_start'],
                    'change': float(change),
                    'direction': 'positive' if change > 0 else 'negative',
                    'magnitude': 'large' if abs(change) > 0.4 else 'moderate'
                })

        return changes
