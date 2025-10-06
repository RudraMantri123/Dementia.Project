"""Intervention recommendation engine based on analytics."""

from typing import Dict, Any, List
from datetime import datetime, timedelta

from src.database.models import UserAnalytics, UserProfile
from src.database.connection import get_db_manager
from src.analytics.conversation_analyzer import ConversationAnalyzer
from src.analytics.longitudinal_tracker import LongitudinalTracker
from src.analytics.predictive_models import PredictiveStressModeler


class InterventionEngine:
    """Generates personalized intervention recommendations."""

    def __init__(self):
        """Initialize intervention engine."""
        self.db_manager = get_db_manager()
        self.conversation_analyzer = ConversationAnalyzer()
        self.longitudinal_tracker = LongitudinalTracker()
        self.stress_modeler = PredictiveStressModeler()

    def generate_recommendations(
        self,
        user_id: str,
        include_predictive: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive intervention recommendations.

        Args:
            user_id: User identifier
            include_predictive: Whether to include predictive modeling

        Returns:
            Dictionary with recommendations
        """
        # Get user profile
        with self.db_manager.get_session() as session:
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()

            if not profile:
                return {'error': 'User profile not found'}

            # Get latest analytics
            latest_analytics = session.query(UserAnalytics).filter_by(
                user_id=user_id
            ).order_by(UserAnalytics.date.desc()).first()

        # Gather insights
        conversation_insights = self.conversation_analyzer.get_conversation_insights(user_id)
        cognitive_trends = self.longitudinal_tracker.track_cognitive_performance(user_id)
        engagement_trends = self.longitudinal_tracker.track_engagement_trends(user_id)
        sentiment_trends = self.longitudinal_tracker.track_sentiment_trends(user_id)

        # Generate recommendations
        recommendations = []
        priority_levels = []

        # Cognitive performance recommendations
        if 'message' not in cognitive_trends:
            cog_recs, cog_priority = self._analyze_cognitive_trends(cognitive_trends)
            recommendations.extend(cog_recs)
            priority_levels.extend(cog_priority)

        # Engagement recommendations
        if 'message' not in engagement_trends:
            eng_recs, eng_priority = self._analyze_engagement_trends(engagement_trends)
            recommendations.extend(eng_recs)
            priority_levels.extend(eng_priority)

        # Sentiment recommendations
        if 'message' not in sentiment_trends:
            sent_recs, sent_priority = self._analyze_sentiment_trends(sentiment_trends)
            recommendations.extend(sent_recs)
            priority_levels.extend(sent_priority)

        # Predictive recommendations
        if include_predictive:
            try:
                stress_prediction = self.stress_modeler.predict_stress(user_id, days_ahead=7)
                if 'error' not in stress_prediction:
                    pred_recs, pred_priority = self._analyze_stress_prediction(stress_prediction)
                    recommendations.extend(pred_recs)
                    priority_levels.extend(pred_priority)
            except Exception as e:
                pass  # Skip if prediction fails

        # Support level determination
        support_level = self._determine_support_level(
            profile,
            latest_analytics,
            engagement_trends,
            sentiment_trends
        )

        # Categorize recommendations
        categorized = self._categorize_recommendations(recommendations, priority_levels)

        return {
            'user_id': user_id,
            'generated_at': datetime.utcnow().isoformat(),
            'support_level': support_level,
            'overall_priority': self._calculate_overall_priority(priority_levels),
            'recommendations': categorized,
            'action_items': self._generate_action_items(categorized, support_level),
            'follow_up_date': (datetime.utcnow() + timedelta(days=7)).isoformat()
        }

    def _analyze_cognitive_trends(
        self,
        cognitive_trends: Dict[str, Any]
    ) -> Tuple[List[str], List[int]]:
        """Analyze cognitive trends and generate recommendations."""
        recommendations = []
        priorities = []

        trend = cognitive_trends.get('trend_analysis', {}).get('trend')

        if trend == 'declining':
            recommendations.append(
                "Cognitive performance is declining. Recommend neurological assessment."
            )
            priorities.append(3)  # High priority

            recommendations.append(
                "Adjust exercise difficulty to maintain engagement and prevent frustration."
            )
            priorities.append(2)

            recommendations.append(
                "Increase frequency of cognitive exercises to 2-3 times daily."
            )
            priorities.append(2)

        elif trend == 'stable':
            current_level = cognitive_trends.get('current_level', {})

            if current_level.get('level') in ['low', 'medium-low']:
                recommendations.append(
                    "Maintain current exercise difficulty but increase frequency."
                )
                priorities.append(1)

        elif trend == 'improving':
            recommendations.append(
                "Consider gradually increasing exercise difficulty to maintain challenge."
            )
            priorities.append(1)

        # Type-specific recommendations
        performance_categories = cognitive_trends.get('performance_categories', {})
        by_type = performance_categories.get('by_type', {})

        for ex_type, data in by_type.items():
            if data['avg_score'] < 0.4:
                recommendations.append(
                    f"Focus on {ex_type} exercises with simplified versions."
                )
                priorities.append(2)

        return recommendations, priorities

    def _analyze_engagement_trends(
        self,
        engagement_trends: Dict[str, Any]
    ) -> Tuple[List[str], List[int]]:
        """Analyze engagement trends and generate recommendations."""
        recommendations = []
        priorities = []

        trend = engagement_trends.get('engagement_trend')
        avg_engagement = engagement_trends.get('average_engagement_score', 0)

        if trend == 'decreasing' or avg_engagement < 0.3:
            recommendations.append(
                "Low engagement detected. Consider more interactive content or caregiver involvement."
            )
            priorities.append(2)

            recommendations.append(
                "Schedule sessions during peak activity hours based on user patterns."
            )
            priorities.append(2)

        consistency = engagement_trends.get('consistency_score', 0)

        if consistency < 0.5:
            recommendations.append(
                "Inconsistent engagement. Establish a regular daily routine for interactions."
            )
            priorities.append(2)

        return recommendations, priorities

    def _analyze_sentiment_trends(
        self,
        sentiment_trends: Dict[str, Any]
    ) -> Tuple[List[str], List[int]]:
        """Analyze sentiment trends and generate recommendations."""
        recommendations = []
        priorities = []

        trend = sentiment_trends.get('sentiment_trend')
        avg_sentiment = sentiment_trends.get('average_sentiment', 0.5)

        if trend == 'declining' or avg_sentiment < 0.3:
            recommendations.append(
                "Declining emotional well-being. Recommend caregiver support group or counseling."
            )
            priorities.append(3)  # High priority

            recommendations.append(
                "Increase empathetic agent interactions and emotional support content."
            )
            priorities.append(2)

        volatility = sentiment_trends.get('volatility', 0)

        if volatility > 0.3:
            recommendations.append(
                "High emotional volatility detected. Monitor for mood disorders and consult healthcare provider."
            )
            priorities.append(3)

        # Check for significant negative changes
        significant_changes = sentiment_trends.get('significant_changes', [])

        negative_changes = [c for c in significant_changes if c['direction'] == 'negative']

        if negative_changes:
            recommendations.append(
                f"Recent negative emotional shift detected. Immediate follow-up recommended."
            )
            priorities.append(3)

        return recommendations, priorities

    def _analyze_stress_prediction(
        self,
        stress_prediction: Dict[str, Any]
    ) -> Tuple[List[str], List[int]]:
        """Analyze stress prediction and generate recommendations."""
        recommendations = []
        priorities = []

        risk_level = stress_prediction.get('risk_level')
        predicted_stress = stress_prediction.get('predicted_stress_level', 0)

        if risk_level == 'high' or predicted_stress > 0.7:
            recommendations.append(
                "High stress predicted for next week. Proactive intervention recommended."
            )
            priorities.append(3)

            recommendations.append(
                "Schedule caregiver check-in and consider respite care options."
            )
            priorities.append(3)

        elif risk_level == 'medium':
            recommendations.append(
                "Moderate stress predicted. Monitor closely and increase support activities."
            )
            priorities.append(2)

        return recommendations, priorities

    def _determine_support_level(
        self,
        profile: UserProfile,
        analytics: UserAnalytics,
        engagement_trends: Dict[str, Any],
        sentiment_trends: Dict[str, Any]
    ) -> str:
        """Determine overall support level needed."""
        score = 0

        # Dementia stage factor
        stage_scores = {'early': 1, 'moderate': 2, 'advanced': 3}
        score += stage_scores.get(profile.dementia_stage, 1)

        # Engagement factor
        avg_engagement = engagement_trends.get('average_engagement_score', 0.5)
        if avg_engagement < 0.3:
            score += 2
        elif avg_engagement < 0.5:
            score += 1

        # Sentiment factor
        avg_sentiment = sentiment_trends.get('average_sentiment', 0.5)
        if avg_sentiment < 0.3:
            score += 2
        elif avg_sentiment < 0.5:
            score += 1

        # Analytics factor
        if analytics and analytics.needs_support:
            score += 2

        # Determine level
        if score >= 6:
            return 'urgent'
        elif score >= 4:
            return 'high'
        elif score >= 2:
            return 'medium'
        else:
            return 'low'

    def _calculate_overall_priority(self, priorities: List[int]) -> str:
        """Calculate overall priority from individual priorities."""
        if not priorities:
            return 'low'

        max_priority = max(priorities)

        if max_priority >= 3:
            return 'high'
        elif max_priority >= 2:
            return 'medium'
        else:
            return 'low'

    def _categorize_recommendations(
        self,
        recommendations: List[str],
        priorities: List[int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize recommendations by type and priority."""
        categories = {
            'immediate': [],  # Priority 3
            'short_term': [],  # Priority 2
            'ongoing': []      # Priority 1
        }

        for rec, priority in zip(recommendations, priorities):
            item = {'recommendation': rec, 'priority': priority}

            if priority >= 3:
                categories['immediate'].append(item)
            elif priority >= 2:
                categories['short_term'].append(item)
            else:
                categories['ongoing'].append(item)

        return categories

    def _generate_action_items(
        self,
        categorized: Dict[str, List[Dict[str, Any]]],
        support_level: str
    ) -> List[Dict[str, Any]]:
        """Generate specific action items based on recommendations."""
        action_items = []

        # Immediate actions
        if categorized['immediate']:
            action_items.append({
                'action': 'Schedule healthcare provider consultation',
                'timeframe': 'within 48 hours',
                'assignee': 'caregiver',
                'priority': 'high'
            })

        # Support level actions
        if support_level in ['high', 'urgent']:
            action_items.append({
                'action': 'Increase monitoring frequency to daily',
                'timeframe': 'immediate',
                'assignee': 'caregiver',
                'priority': 'high'
            })

            action_items.append({
                'action': 'Consider respite care or additional support services',
                'timeframe': 'within 1 week',
                'assignee': 'caregiver',
                'priority': 'high'
            })

        # Standard actions
        action_items.append({
            'action': 'Review and adjust exercise difficulty based on performance',
            'timeframe': 'within 3 days',
            'assignee': 'system',
            'priority': 'medium'
        })

        action_items.append({
            'action': 'Follow up on recommendations',
            'timeframe': 'within 1 week',
            'assignee': 'caregiver',
            'priority': 'medium'
        })

        return action_items
