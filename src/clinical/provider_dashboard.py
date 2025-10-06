"""Healthcare Provider Dashboard for clinical monitoring."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from src.database.models import UserProfile, Conversation, CognitiveExerciseResult, ClinicalData
from src.database.connection import get_db_manager
from src.analytics.longitudinal_tracker import LongitudinalTracker
from src.analytics.predictive_models import PredictiveStressModeler
from src.personalization.user_profile_manager import UserProfileManager


class ProviderDashboard:
    """Dashboard interface for healthcare providers."""

    def __init__(self):
        """Initialize provider dashboard."""
        self.db_manager = get_db_manager()
        self.longitudinal_tracker = LongitudinalTracker()
        self.stress_modeler = PredictiveStressModeler()
        self.profile_manager = UserProfileManager()

    def get_patient_overview(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive patient overview.

        Args:
            user_id: Patient identifier

        Returns:
            Patient overview data
        """
        with self.db_manager.get_session() as session:
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()

            if not profile:
                return {'error': 'Patient not found'}

            # Get recent activity
            recent_conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.timestamp >= datetime.utcnow() - timedelta(days=30),
                Conversation.role == 'user'
            ).count()

            recent_exercises = session.query(CognitiveExerciseResult).filter(
                CognitiveExerciseResult.user_id == user_id,
                CognitiveExerciseResult.timestamp >= datetime.utcnow() - timedelta(days=30)
            ).count()

            # Get clinical data
            clinical_records = session.query(ClinicalData).filter_by(
                user_id=user_id
            ).order_by(ClinicalData.assessment_date.desc()).limit(5).all()

            return {
                'patient_info': {
                    'user_id': user_id,
                    'name': profile.name,
                    'age': profile.age,
                    'dementia_stage': profile.dementia_stage,
                    'cognitive_level': profile.cognitive_level,
                    'last_interaction': profile.last_interaction.isoformat() if profile.last_interaction else None
                },
                'recent_activity': {
                    'conversations_30d': recent_conversations,
                    'exercises_30d': recent_exercises,
                    'last_seen': profile.last_interaction.isoformat() if profile.last_interaction else None
                },
                'medical_info': {
                    'medical_conditions': profile.medical_conditions or [],
                    'medications': profile.medications or [],
                    'caregiver_info': profile.caregiver_info or {}
                },
                'clinical_assessments': [
                    {
                        'type': record.assessment_type,
                        'score': record.assessment_score,
                        'date': record.assessment_date.isoformat() if record.assessment_date else None,
                        'diagnosis': record.diagnosis_description
                    }
                    for record in clinical_records
                ]
            }

    def get_patient_trends(self, user_id: str, weeks: int = 12) -> Dict[str, Any]:
        """
        Get patient progress trends.

        Args:
            user_id: Patient identifier
            weeks: Number of weeks to analyze

        Returns:
            Trend analysis data
        """
        # Get cognitive performance trends
        cognitive_trends = self.longitudinal_tracker.track_cognitive_performance(
            user_id, weeks=weeks
        )

        # Get engagement trends
        engagement_trends = self.longitudinal_tracker.track_engagement_trends(
            user_id, weeks=weeks
        )

        # Get sentiment trends
        sentiment_trends = self.longitudinal_tracker.track_sentiment_trends(
            user_id, weeks=weeks
        )

        # Get stress prediction
        stress_prediction = self.stress_modeler.predict_stress(user_id)

        return {
            'cognitive_performance': cognitive_trends,
            'engagement': engagement_trends,
            'sentiment': sentiment_trends,
            'stress_prediction': stress_prediction,
            'summary': self._generate_trend_summary(
                cognitive_trends,
                engagement_trends,
                sentiment_trends
            )
        }

    def get_patient_list(
        self,
        provider_id: str,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of patients for provider.

        Args:
            provider_id: Provider identifier
            filters: Optional filters (dementia_stage, risk_level, etc.)

        Returns:
            List of patient summaries
        """
        with self.db_manager.get_session() as session:
            query = session.query(UserProfile)

            # Apply filters
            if filters:
                if 'dementia_stage' in filters:
                    query = query.filter_by(dementia_stage=filters['dementia_stage'])

                if 'min_cognitive_level' in filters:
                    query = query.filter(
                        UserProfile.cognitive_level >= filters['min_cognitive_level']
                    )

                if 'max_cognitive_level' in filters:
                    query = query.filter(
                        UserProfile.cognitive_level <= filters['max_cognitive_level']
                    )

            patients = query.all()

            patient_list = []

            for patient in patients:
                # Get last interaction
                last_conversation = session.query(Conversation).filter(
                    Conversation.user_id == patient.user_id,
                    Conversation.role == 'user'
                ).order_by(Conversation.timestamp.desc()).first()

                # Calculate risk score
                risk_score = self._calculate_risk_score(patient, session)

                patient_list.append({
                    'user_id': patient.user_id,
                    'name': patient.name,
                    'age': patient.age,
                    'dementia_stage': patient.dementia_stage,
                    'cognitive_level': patient.cognitive_level,
                    'last_interaction': last_conversation.timestamp.isoformat() if last_conversation else None,
                    'risk_level': self._categorize_risk(risk_score),
                    'risk_score': risk_score,
                    'needs_attention': risk_score > 0.7
                })

            # Sort by risk score (descending)
            patient_list.sort(key=lambda x: x['risk_score'], reverse=True)

            return patient_list

    def get_alerts(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Get clinical alerts for provider.

        Args:
            provider_id: Provider identifier

        Returns:
            List of alerts
        """
        alerts = []

        with self.db_manager.get_session() as session:
            # Get all patients
            patients = session.query(UserProfile).all()

            for patient in patients:
                # Check for inactivity
                if patient.last_interaction:
                    days_inactive = (datetime.utcnow() - patient.last_interaction).days

                    if days_inactive > 7:
                        alerts.append({
                            'type': 'inactivity',
                            'severity': 'medium' if days_inactive < 14 else 'high',
                            'patient_id': patient.user_id,
                            'patient_name': patient.name,
                            'message': f'Patient inactive for {days_inactive} days',
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Check for declining cognitive performance
                cognitive_trends = self.longitudinal_tracker.track_cognitive_performance(
                    patient.user_id, weeks=4
                )

                if cognitive_trends.get('trend_analysis', {}).get('trend') == 'declining':
                    alerts.append({
                        'type': 'cognitive_decline',
                        'severity': 'high',
                        'patient_id': patient.user_id,
                        'patient_name': patient.name,
                        'message': 'Cognitive performance declining',
                        'timestamp': datetime.utcnow().isoformat()
                    })

                # Check for high stress prediction
                stress_prediction = self.stress_modeler.predict_stress(patient.user_id)

                if stress_prediction.get('risk_level') == 'high':
                    alerts.append({
                        'type': 'high_stress',
                        'severity': 'high',
                        'patient_id': patient.user_id,
                        'patient_name': patient.name,
                        'message': f'High stress predicted: {stress_prediction.get("predicted_stress_level", 0):.2f}',
                        'timestamp': datetime.utcnow().isoformat()
                    })

        # Sort by severity and timestamp
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        alerts.sort(key=lambda x: (severity_order.get(x['severity'], 3), x['timestamp']), reverse=True)

        return alerts

    def get_aggregate_statistics(self, provider_id: str) -> Dict[str, Any]:
        """
        Get aggregate statistics across all patients.

        Args:
            provider_id: Provider identifier

        Returns:
            Aggregate statistics
        """
        with self.db_manager.get_session() as session:
            patients = session.query(UserProfile).all()

            if not patients:
                return {'total_patients': 0}

            # Aggregate metrics
            total_patients = len(patients)
            avg_cognitive_level = sum(p.cognitive_level for p in patients) / total_patients

            # Dementia stage distribution
            stage_distribution = defaultdict(int)
            for patient in patients:
                stage_distribution[patient.dementia_stage] += 1

            # Activity metrics
            active_30d = 0
            for patient in patients:
                if patient.last_interaction and \
                   (datetime.utcnow() - patient.last_interaction).days <= 30:
                    active_30d += 1

            # High-risk patients
            high_risk_count = 0
            for patient in patients:
                risk_score = self._calculate_risk_score(patient, session)
                if risk_score > 0.7:
                    high_risk_count += 1

            return {
                'total_patients': total_patients,
                'active_patients_30d': active_30d,
                'avg_cognitive_level': round(avg_cognitive_level, 3),
                'stage_distribution': dict(stage_distribution),
                'high_risk_patients': high_risk_count,
                'engagement_rate': round(active_30d / total_patients, 3) if total_patients > 0 else 0
            }

    def get_patient_conversation_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get patient conversation history.

        Args:
            user_id: Patient identifier
            limit: Maximum number of conversations

        Returns:
            Conversation history
        """
        with self.db_manager.get_session() as session:
            conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(
                Conversation.timestamp.desc()
            ).limit(limit).all()

            return [
                {
                    'id': conv.id,
                    'role': conv.role,
                    'content': conv.content,
                    'agent_used': conv.agent_used,
                    'intent': conv.intent,
                    'sentiment': conv.sentiment,
                    'sentiment_score': conv.sentiment_score,
                    'timestamp': conv.timestamp.isoformat()
                }
                for conv in conversations
            ]

    def export_patient_report(self, user_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive patient report.

        Args:
            user_id: Patient identifier

        Returns:
            Complete patient report
        """
        overview = self.get_patient_overview(user_id)
        trends = self.get_patient_trends(user_id, weeks=12)
        statistics = self.profile_manager.get_user_statistics(user_id, days=90)

        return {
            'report_generated': datetime.utcnow().isoformat(),
            'patient_overview': overview,
            'performance_trends': trends,
            'detailed_statistics': statistics,
            'recommendations': self._generate_clinical_recommendations(
                overview,
                trends,
                statistics
            )
        }

    def _calculate_risk_score(self, patient: UserProfile, session) -> float:
        """Calculate overall risk score for patient."""
        risk_score = 0.0

        # Factor 1: Low cognitive level
        if patient.cognitive_level < 0.3:
            risk_score += 0.3
        elif patient.cognitive_level < 0.5:
            risk_score += 0.1

        # Factor 2: Inactivity
        if patient.last_interaction:
            days_inactive = (datetime.utcnow() - patient.last_interaction).days
            if days_inactive > 14:
                risk_score += 0.3
            elif days_inactive > 7:
                risk_score += 0.15

        # Factor 3: Advanced dementia stage
        if patient.dementia_stage == 'advanced':
            risk_score += 0.2
        elif patient.dementia_stage == 'moderate':
            risk_score += 0.1

        # Factor 4: Recent performance decline
        recent_exercises = session.query(CognitiveExerciseResult).filter(
            CognitiveExerciseResult.user_id == patient.user_id,
            CognitiveExerciseResult.timestamp >= datetime.utcnow() - timedelta(days=30)
        ).all()

        if recent_exercises:
            avg_score = sum(e.performance_score for e in recent_exercises if e.performance_score) / len(recent_exercises)
            if avg_score < 0.4:
                risk_score += 0.2

        return min(risk_score, 1.0)

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level."""
        if risk_score >= 0.7:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def _generate_trend_summary(
        self,
        cognitive_trends: Dict[str, Any],
        engagement_trends: Dict[str, Any],
        sentiment_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of trends."""
        summary = {
            'cognitive_status': 'unknown',
            'engagement_status': 'unknown',
            'sentiment_status': 'unknown',
            'overall_assessment': 'unknown',
            'key_points': []
        }

        # Cognitive status
        cog_trend = cognitive_trends.get('trend_analysis', {}).get('trend', 'unknown')
        summary['cognitive_status'] = cog_trend

        if cog_trend == 'improving':
            summary['key_points'].append('✓ Cognitive performance is improving')
        elif cog_trend == 'declining':
            summary['key_points'].append('⚠ Cognitive performance is declining')

        # Engagement status
        eng_trend = engagement_trends.get('engagement_trend', 'unknown')
        summary['engagement_status'] = eng_trend

        if eng_trend == 'increasing':
            summary['key_points'].append('✓ User engagement is increasing')
        elif eng_trend == 'decreasing':
            summary['key_points'].append('⚠ User engagement is decreasing')

        # Sentiment status
        sent_trend = sentiment_trends.get('sentiment_trend', 'unknown')
        summary['sentiment_status'] = sent_trend

        if sent_trend == 'improving':
            summary['key_points'].append('✓ Emotional state is improving')
        elif sent_trend == 'declining':
            summary['key_points'].append('⚠ Emotional state is declining')

        # Overall assessment
        if cog_trend == 'improving' and eng_trend == 'increasing':
            summary['overall_assessment'] = 'positive'
        elif cog_trend == 'declining' or eng_trend == 'decreasing':
            summary['overall_assessment'] = 'concerning'
        else:
            summary['overall_assessment'] = 'stable'

        return summary

    def _generate_clinical_recommendations(
        self,
        overview: Dict[str, Any],
        trends: Dict[str, Any],
        statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []

        # Based on cognitive trends
        cog_trend = trends.get('cognitive_performance', {}).get('trend_analysis', {}).get('trend')
        if cog_trend == 'declining':
            recommendations.append("Consider adjusting medication or treatment plan due to declining cognitive performance")
            recommendations.append("Schedule comprehensive cognitive assessment")

        # Based on engagement
        engagement_rate = statistics.get('engagement', {}).get('total_conversations', 0)
        if engagement_rate < 10:
            recommendations.append("Increase engagement activities - current interaction frequency is low")

        # Based on stress prediction
        stress_pred = trends.get('stress_prediction', {})
        if stress_pred.get('risk_level') == 'high':
            recommendations.append("High caregiver stress predicted - recommend respite care or support services")

        # Based on cognitive level
        cognitive_level = overview.get('patient_info', {}).get('cognitive_level', 0.5)
        if cognitive_level < 0.3:
            recommendations.append("Consider more intensive support services due to low cognitive function")

        return recommendations

