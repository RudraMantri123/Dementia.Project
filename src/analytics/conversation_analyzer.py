"""Conversation history analysis with pattern detection."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.database.models import Conversation, UserProfile
from src.database.connection import get_db_manager


class ConversationAnalyzer:
    """Analyzes conversation patterns and detects trends."""

    def __init__(self):
        """Initialize analyzer."""
        self.db_manager = get_db_manager()

    def analyze_conversation_patterns(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze conversation patterns for a user.

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Dictionary with pattern analysis
        """
        with self.db_manager.get_session() as session:
            since_date = datetime.utcnow() - timedelta(days=days)
            conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.timestamp >= since_date
            ).order_by(Conversation.timestamp).all()

            if not conversations:
                return {'message': 'No conversations found'}

            # Extract patterns
            temporal_patterns = self._analyze_temporal_patterns(conversations)
            topic_patterns = self._analyze_topic_patterns(conversations)
            sentiment_patterns = self._analyze_sentiment_patterns(conversations)
            engagement_patterns = self._analyze_engagement_patterns(conversations)
            repetition_patterns = self._detect_repetitions(conversations)

            return {
                'total_conversations': len(conversations),
                'date_range': {
                    'start': conversations[0].timestamp.isoformat(),
                    'end': conversations[-1].timestamp.isoformat()
                },
                'temporal_patterns': temporal_patterns,
                'topic_patterns': topic_patterns,
                'sentiment_patterns': sentiment_patterns,
                'engagement_patterns': engagement_patterns,
                'repetition_patterns': repetition_patterns,
                'anomalies': self._detect_anomalies(conversations)
            }

    def _analyze_temporal_patterns(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze time-based patterns."""
        # Hour distribution
        hours = [c.timestamp.hour for c in conversations]
        hour_counts = Counter(hours)

        # Day of week distribution
        days = [c.timestamp.weekday() for c in conversations]
        day_counts = Counter(days)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Session duration analysis
        sessions = defaultdict(list)
        for conv in conversations:
            sessions[conv.session_id].append(conv.timestamp)

        session_durations = []
        for session_times in sessions.values():
            if len(session_times) > 1:
                duration = (max(session_times) - min(session_times)).total_seconds() / 60
                session_durations.append(duration)

        return {
            'peak_hours': [hour for hour, _ in hour_counts.most_common(3)],
            'hourly_distribution': dict(hour_counts),
            'peak_days': [day_names[day] for day, _ in day_counts.most_common(3)],
            'daily_distribution': {day_names[day]: count for day, count in day_counts.items()},
            'avg_session_duration_minutes': np.mean(session_durations) if session_durations else 0,
            'total_sessions': len(sessions)
        }

    def _analyze_topic_patterns(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze topic patterns and clusters."""
        # Intent/topic distribution
        intents = [c.intent for c in conversations if c.intent and c.role == 'user']
        intent_counts = Counter(intents)

        # Agent usage
        agents = [c.agent_used for c in conversations if c.agent_used]
        agent_counts = Counter(agents)

        # Extract user messages for topic clustering
        user_messages = [c.content for c in conversations if c.role == 'user' and c.content]

        topic_clusters = {}
        if len(user_messages) >= 5:
            topic_clusters = self._cluster_topics(user_messages)

        return {
            'top_intents': dict(intent_counts.most_common(5)),
            'agent_distribution': dict(agent_counts),
            'topic_clusters': topic_clusters,
            'topic_diversity': len(set(intents)) / len(intents) if intents else 0
        }

    def _cluster_topics(self, messages: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster messages into topics using K-means."""
        if len(messages) < n_clusters:
            n_clusters = len(messages)

        try:
            # Vectorize messages
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            vectors = vectorizer.fit_transform(messages)

            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)

            # Get top terms per cluster
            feature_names = vectorizer.get_feature_names_out()
            clusters = {}

            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                cluster_size = sum(1 for label in labels if label == i)

                clusters[f'cluster_{i}'] = {
                    'top_terms': top_terms,
                    'size': cluster_size,
                    'percentage': (cluster_size / len(messages)) * 100
                }

            return clusters

        except Exception as e:
            return {'error': str(e)}

    def _analyze_sentiment_patterns(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze sentiment patterns over time."""
        user_convs = [c for c in conversations if c.role == 'user']

        sentiments = [c.sentiment for c in user_convs if c.sentiment]
        sentiment_counts = Counter(sentiments)

        # Sentiment scores over time
        sentiment_scores = [(c.timestamp, c.sentiment_score) for c in user_convs if c.sentiment_score is not None]

        if sentiment_scores:
            timestamps, scores = zip(*sentiment_scores)
            avg_score = np.mean(scores)

            # Detect trends
            if len(scores) >= 3:
                # Simple linear trend
                x = np.arange(len(scores))
                coeffs = np.polyfit(x, scores, 1)
                trend = 'improving' if coeffs[0] > 0.01 else 'declining' if coeffs[0] < -0.01 else 'stable'
            else:
                trend = 'insufficient_data'
        else:
            avg_score = None
            trend = 'no_data'

        return {
            'sentiment_distribution': dict(sentiment_counts),
            'average_sentiment_score': avg_score,
            'sentiment_trend': trend,
            'total_analyzed': len(sentiments)
        }

    def _analyze_engagement_patterns(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze user engagement patterns."""
        user_convs = [c for c in conversations if c.role == 'user']

        # Message length analysis
        message_lengths = [len(c.content) for c in user_convs]

        # Response time analysis (if available)
        response_times = [c.response_time_ms for c in conversations if c.response_time_ms]

        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(conversations)

        return {
            'avg_message_length': np.mean(message_lengths) if message_lengths else 0,
            'message_length_std': np.std(message_lengths) if message_lengths else 0,
            'avg_response_time_ms': np.mean(response_times) if response_times else 0,
            'engagement_score': engagement_score,
            'total_user_messages': len(user_convs)
        }

    def _calculate_engagement_score(self, conversations: List[Conversation]) -> float:
        """
        Calculate overall engagement score (0-1).

        Factors:
        - Message frequency
        - Message length
        - Session duration
        - Topic diversity
        """
        if not conversations:
            return 0.0

        user_convs = [c for c in conversations if c.role == 'user']

        # Frequency score
        time_span = (conversations[-1].timestamp - conversations[0].timestamp).total_seconds() / 3600
        messages_per_hour = len(user_convs) / max(time_span, 1)
        frequency_score = min(messages_per_hour / 5, 1.0)  # Normalize to 0-1

        # Length score (engagement indicator)
        avg_length = np.mean([len(c.content) for c in user_convs])
        length_score = min(avg_length / 200, 1.0)

        # Diversity score
        intents = [c.intent for c in user_convs if c.intent]
        diversity_score = len(set(intents)) / max(len(intents), 1) if intents else 0.5

        # Overall engagement score
        engagement_score = (frequency_score * 0.4 + length_score * 0.3 + diversity_score * 0.3)

        return round(engagement_score, 3)

    def _detect_repetitions(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Detect repetitive patterns in conversations."""
        user_convs = [c for c in conversations if c.role == 'user']

        if len(user_convs) < 3:
            return {'repetition_detected': False}

        # Check for repeated questions/phrases
        messages = [c.content.lower() for c in user_convs]
        message_counts = Counter(messages)

        repeated_messages = {msg: count for msg, count in message_counts.items() if count > 2}

        # Check for similar messages using TF-IDF similarity
        if len(messages) >= 5:
            try:
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(messages)
                similarity_matrix = cosine_similarity(vectors)

                # Find highly similar pairs
                similar_pairs = []
                for i in range(len(messages)):
                    for j in range(i + 1, len(messages)):
                        if similarity_matrix[i][j] > 0.8:  # 80% similarity threshold
                            similar_pairs.append({
                                'message1': messages[i][:100],
                                'message2': messages[j][:100],
                                'similarity': float(similarity_matrix[i][j])
                            })

                repetition_score = len(similar_pairs) / len(messages)

            except Exception:
                similar_pairs = []
                repetition_score = 0

        else:
            similar_pairs = []
            repetition_score = 0

        return {
            'repetition_detected': len(repeated_messages) > 0 or repetition_score > 0.2,
            'exact_repetitions': len(repeated_messages),
            'repeated_phrases': list(repeated_messages.keys())[:5],
            'similar_message_pairs': similar_pairs[:5],
            'repetition_score': round(repetition_score, 3)
        }

    def _detect_anomalies(self, conversations: List[Conversation]) -> List[Dict[str, Any]]:
        """Detect anomalies in conversation patterns."""
        anomalies = []

        # Check for sudden sentiment changes
        user_convs = [c for c in conversations if c.role == 'user' and c.sentiment_score is not None]

        if len(user_convs) >= 5:
            scores = [c.sentiment_score for c in user_convs]
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            for i, conv in enumerate(user_convs):
                # Detect outliers (more than 2 standard deviations)
                if abs(conv.sentiment_score - mean_score) > 2 * std_score:
                    anomalies.append({
                        'type': 'sentiment_outlier',
                        'timestamp': conv.timestamp.isoformat(),
                        'sentiment_score': conv.sentiment_score,
                        'message': conv.content[:100]
                    })

        # Check for unusual gaps in conversation
        timestamps = [c.timestamp for c in conversations]
        for i in range(1, len(timestamps)):
            gap_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
            if gap_hours > 48:  # More than 2 days
                anomalies.append({
                    'type': 'conversation_gap',
                    'gap_hours': round(gap_hours, 1),
                    'start': timestamps[i-1].isoformat(),
                    'end': timestamps[i].isoformat()
                })

        return anomalies[:10]  # Return top 10 anomalies

    def get_conversation_insights(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get high-level insights from conversation analysis.

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Dictionary with insights and recommendations
        """
        patterns = self.analyze_conversation_patterns(user_id, days)

        if 'message' in patterns:
            return patterns

        insights = []
        recommendations = []

        # Engagement insights
        engagement_score = patterns['engagement_patterns']['engagement_score']
        if engagement_score < 0.3:
            insights.append("Low engagement detected")
            recommendations.append("Consider more interactive content or shorter sessions")
        elif engagement_score > 0.7:
            insights.append("High engagement observed")

        # Sentiment insights
        sentiment_trend = patterns['sentiment_patterns']['sentiment_trend']
        if sentiment_trend == 'declining':
            insights.append("Sentiment appears to be declining")
            recommendations.append("Increased emotional support may be needed")
        elif sentiment_trend == 'improving':
            insights.append("Sentiment is improving over time")

        # Repetition insights
        if patterns['repetition_patterns']['repetition_detected']:
            insights.append("Repetitive conversation patterns detected")
            recommendations.append("May indicate memory challenges - provide consistent reminders")

        # Temporal insights
        peak_hours = patterns['temporal_patterns']['peak_hours']
        if peak_hours:
            insights.append(f"Most active during hours: {peak_hours}")
            recommendations.append(f"Schedule important interactions during peak hours: {peak_hours}")

        # Topic diversity
        topic_diversity = patterns['topic_patterns']['topic_diversity']
        if topic_diversity < 0.3:
            insights.append("Limited topic diversity")
            recommendations.append("Introduce new conversation topics or activities")

        return {
            'summary': {
                'engagement_level': 'high' if engagement_score > 0.7 else 'medium' if engagement_score > 0.4 else 'low',
                'sentiment_trend': sentiment_trend,
                'primary_concerns': list(patterns['topic_patterns']['top_intents'].keys())[:3]
            },
            'insights': insights,
            'recommendations': recommendations,
            'detailed_patterns': patterns
        }
