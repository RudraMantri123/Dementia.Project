"""Advanced personalized insights engine for dynamic, meaningful recommendations."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import json


class PersonalizedInsightsEngine:
    """Generates deeply personalized, dynamic insights based on individual user patterns."""
    
    def __init__(self):
        """Initialize the personalized insights engine."""
        self.user_profiles = {}
        self.conversation_patterns = defaultdict(list)
        self.emotional_journeys = defaultdict(list)
        self.intervention_history = defaultdict(list)
        self.personalization_weights = {}
        
    def analyze_conversation_depth(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation patterns for deep personalization."""
        if not conversation_history:
            return self._get_default_insights()
        
        # Extract user messages with timestamps
        user_messages = [
            {
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', datetime.now().isoformat()),
                'sentiment': msg.get('sentiment', 'neutral')
            }
            for msg in conversation_history 
            if msg.get('role') == 'user'
        ]
        
        if not user_messages:
            return self._get_default_insights()
        
        # Deep pattern analysis
        patterns = self._extract_deep_patterns(user_messages)
        emotional_arc = self._analyze_emotional_journey(user_messages)
        communication_style = self._analyze_communication_style(user_messages)
        support_needs = self._identify_support_needs(user_messages)
        
        return {
            'patterns': patterns,
            'emotional_arc': emotional_arc,
            'communication_style': communication_style,
            'support_needs': support_needs,
            'personalized_recommendations': self._generate_personalized_recommendations(
                patterns, emotional_arc, communication_style, support_needs
            ),
            'insight_confidence': self._calculate_insight_confidence(user_messages)
        }
    
    def _extract_deep_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract deep behavioral and emotional patterns."""
        patterns = {
            'temporal_patterns': self._analyze_temporal_patterns(messages),
            'emotional_patterns': self._analyze_emotional_patterns(messages),
            'linguistic_patterns': self._analyze_linguistic_patterns(messages),
            'topic_patterns': self._analyze_topic_patterns(messages),
            'crisis_indicators': self._detect_crisis_indicators(messages)
        }
        return patterns
    
    def _analyze_temporal_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze when and how often user engages."""
        if len(messages) < 2:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}
        
        timestamps = [datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')) for msg in messages]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 for i in range(len(timestamps)-1)]
        
        # Engagement frequency
        avg_gap = np.mean(time_diffs) if time_diffs else 24
        if avg_gap < 1:
            frequency_pattern = 'high_engagement'
        elif avg_gap < 6:
            frequency_pattern = 'moderate_engagement'
        elif avg_gap < 24:
            frequency_pattern = 'daily_engagement'
        else:
            frequency_pattern = 'sporadic_engagement'
        
        # Time of day patterns
        hours = [ts.hour for ts in timestamps]
        hour_counter = Counter(hours)
        most_active_hour = hour_counter.most_common(1)[0][0] if hour_counter else 12
        
        if 6 <= most_active_hour < 12:
            time_preference = 'morning_person'
        elif 12 <= most_active_hour < 18:
            time_preference = 'afternoon_person'
        elif 18 <= most_active_hour < 22:
            time_preference = 'evening_person'
        else:
            time_preference = 'night_person'
        
        return {
            'frequency_pattern': frequency_pattern,
            'time_preference': time_preference,
            'avg_response_time_hours': round(avg_gap, 1),
            'most_active_hour': most_active_hour,
            'engagement_consistency': 1.0 - (np.std(time_diffs) / np.mean(time_diffs)) if time_diffs and np.mean(time_diffs) > 0 else 0.0
        }
    
    def _analyze_emotional_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotional patterns and transitions."""
        sentiments = [msg['sentiment'] for msg in messages]
        sentiment_counter = Counter(sentiments)
        
        # Emotional diversity
        emotional_diversity = len(sentiment_counter) / 6.0  # 6 possible emotions
        
        # Emotional transitions
        transitions = []
        for i in range(len(sentiments) - 1):
            transitions.append(f"{sentiments[i]}_to_{sentiments[i+1]}")
        
        transition_counter = Counter(transitions)
        
        # Identify emotional cycles
        emotional_cycles = self._identify_emotional_cycles(sentiments)
        
        # Emotional volatility
        volatility = self._calculate_emotional_volatility(sentiments)
        
        return {
            'dominant_emotion': sentiment_counter.most_common(1)[0][0] if sentiment_counter else 'neutral',
            'emotional_diversity': round(emotional_diversity, 2),
            'emotional_volatility': volatility,
            'common_transitions': dict(transition_counter.most_common(3)),
            'emotional_cycles': emotional_cycles,
            'sentiment_distribution': dict(sentiment_counter)
        }
    
    def _analyze_linguistic_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze linguistic patterns and communication style."""
        all_text = ' '.join([msg['content'] for msg in messages])
        
        # Message length patterns
        lengths = [len(msg['content']) for msg in messages]
        avg_length = np.mean(lengths)
        
        if avg_length < 50:
            communication_style = 'concise'
        elif avg_length < 150:
            communication_style = 'moderate'
        else:
            communication_style = 'detailed'
        
        # Language complexity
        avg_words = np.mean([len(msg['content'].split()) for msg in messages])
        question_ratio = sum(1 for msg in messages if '?' in msg['content']) / len(messages)
        exclamation_ratio = sum(1 for msg in messages if '!' in msg['content']) / len(messages)
        
        # Emotional language indicators
        emotional_words = self._count_emotional_words(all_text)
        
        return {
            'communication_style': communication_style,
            'avg_message_length': round(avg_length, 1),
            'avg_words_per_message': round(avg_words, 1),
            'question_frequency': round(question_ratio, 2),
            'exclamation_frequency': round(exclamation_ratio, 2),
            'emotional_language_indicators': emotional_words,
            'language_complexity': 'simple' if avg_words < 10 else 'moderate' if avg_words < 20 else 'complex'
        }
    
    def _analyze_topic_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recurring topics and themes with clinical dementia awareness."""
        all_text = ' '.join([msg['content'].lower() for msg in messages])
        
        # Topic categories with clinical awareness
        topic_categories = {
            'caregiving_challenges': ['overwhelmed', 'tired', 'exhausted', 'difficult', 'hard', 'struggle'],
            'emotional_support': ['sad', 'lonely', 'depressed', 'anxious', 'worried', 'scared'],
            'practical_help': ['how', 'what', 'when', 'where', 'help', 'advice', 'information'],
            'family_dynamics': ['family', 'children', 'spouse', 'parent', 'relationship'],
            'medical_concerns': ['doctor', 'medication', 'symptoms', 'health', 'medical'],
            'future_planning': ['future', 'plan', 'decision', 'choice', 'next'],
            'self_care': ['break', 'rest', 'time', 'myself', 'self-care', 'respite'],
            'dementia_symptoms': ['forget', 'memory', 'confused', 'lost', 'wandering', 'agitation', 'sundowning'],
            'normal_aging': ['sometimes', 'occasionally', 'once in a while', 'temporary', 'brief']
        }
        
        topic_scores = {}
        for category, keywords in topic_categories.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            topic_scores[category] = score
        
        # Analyze dementia vs normal aging patterns
        dementia_concerns = self._analyze_dementia_vs_normal_aging(all_text)
        
        primary_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'primary_topics': [topic[0] for topic in primary_topics if topic[1] > 0],
            'topic_scores': topic_scores,
            'conversation_focus': primary_topics[0][0] if primary_topics and primary_topics[0][1] > 0 else 'general',
            'dementia_analysis': dementia_concerns
        }
    
    def _detect_crisis_indicators(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential crisis indicators with high sensitivity."""
        crisis_keywords = {
            'immediate_risk': ['suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself'],
            'emotional_crisis': ['can\'t cope', 'breaking down', 'falling apart', 'losing it', 'desperate'],
            'isolation': ['alone', 'nobody cares', 'no one understands', 'isolated', 'abandoned'],
            'hopelessness': ['hopeless', 'no point', 'give up', 'nothing matters', 'pointless'],
            'overwhelm': ['too much', 'can\'t handle', 'drowning', 'suffocating', 'trapped']
        }
        
        all_text = ' '.join([msg['content'].lower() for msg in messages])
        crisis_indicators = {}
        
        for category, keywords in crisis_keywords.items():
            found_keywords = [kw for kw in keywords if kw in all_text]
            crisis_indicators[category] = {
                'detected': len(found_keywords) > 0,
                'keywords_found': found_keywords,
                'severity': len(found_keywords)
            }
        
        overall_crisis_score = sum(indicator['severity'] for indicator in crisis_indicators.values())
        
        return {
            'crisis_indicators': crisis_indicators,
            'overall_crisis_score': overall_crisis_score,
            'crisis_level': 'high' if overall_crisis_score >= 3 else 'moderate' if overall_crisis_score >= 1 else 'low',
            'requires_immediate_attention': overall_crisis_score >= 3
        }
    
    def _analyze_emotional_journey(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the emotional journey and progression."""
        if len(messages) < 3:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}
        
        sentiments = [msg['sentiment'] for msg in messages]
        
        # Emotional progression
        first_half = sentiments[:len(sentiments)//2]
        second_half = sentiments[len(sentiments)//2:]
        
        first_avg = self._sentiment_to_numeric(first_half)
        second_avg = self._sentiment_to_numeric(second_half)
        
        progression = second_avg - first_avg
        
        if progression > 0.3:
            journey_pattern = 'improving'
        elif progression < -0.3:
            journey_pattern = 'declining'
        else:
            journey_pattern = 'stable'
        
        # Emotional resilience indicators
        resilience_indicators = self._assess_emotional_resilience(sentiments)
        
        return {
            'journey_pattern': journey_pattern,
            'emotional_progression': round(progression, 2),
            'resilience_indicators': resilience_indicators,
            'journey_confidence': min(len(messages) / 10.0, 1.0)
        }
    
    def _analyze_communication_style(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual communication preferences and style."""
        if not messages:
            return {'style': 'unknown', 'confidence': 0.0}
        
        # Analyze response patterns
        response_patterns = self._analyze_response_patterns(messages)
        
        # Communication preferences
        preferences = self._identify_communication_preferences(messages)
        
        return {
            'response_patterns': response_patterns,
            'preferences': preferences,
            'style_confidence': min(len(messages) / 5.0, 1.0)
        }
    
    def _identify_support_needs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify specific support needs based on conversation content."""
        all_text = ' '.join([msg['content'].lower() for msg in messages])
        
        support_categories = {
            'emotional_support': ['feel', 'emotion', 'sad', 'anxious', 'overwhelmed', 'lonely'],
            'practical_guidance': ['how', 'what', 'when', 'where', 'advice', 'help', 'guide'],
            'crisis_intervention': ['crisis', 'emergency', 'urgent', 'immediate', 'help now'],
            'peer_connection': ['others', 'group', 'community', 'similar', 'understand'],
            'respite_care': ['break', 'rest', 'time off', 'respite', 'exhausted', 'burnt out'],
            'medical_support': ['doctor', 'medical', 'health', 'symptoms', 'medication'],
            'family_support': ['family', 'children', 'spouse', 'relationship', 'communication']
        }
        
        support_scores = {}
        for category, keywords in support_categories.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            support_scores[category] = score
        
        # Identify primary support needs
        primary_needs = sorted(support_scores.items(), key=lambda x: x[1], reverse=True)
        primary_need = primary_needs[0][0] if primary_needs and primary_needs[0][1] > 0 else 'general_support'
        
        return {
            'primary_support_need': primary_need,
            'support_scores': support_scores,
            'support_urgency': 'high' if support_scores.get('crisis_intervention', 0) > 0 else 'moderate' if max(support_scores.values()) > 2 else 'low'
        }
    
    def _generate_personalized_recommendations(self, patterns: Dict, emotional_arc: Dict, 
                                             communication_style: Dict, support_needs: Dict) -> List[Dict[str, Any]]:
        """Generate highly personalized, dynamic recommendations."""
        recommendations = []
        
        # Time-based recommendations
        temporal = patterns.get('temporal_patterns', {})
        if temporal.get('time_preference') == 'morning_person':
            recommendations.append({
                'type': 'timing',
                'title': 'Morning Check-ins',
                'description': 'Based on your activity patterns, you seem most engaged in the morning. Consider scheduling important conversations or support sessions during this time.',
                'confidence': 0.8,
                'personalization_level': 'high'
            })
        
        # Emotional pattern recommendations
        emotional = patterns.get('emotional_patterns', {})
        if emotional.get('emotional_volatility', 0) > 0.7:
            recommendations.append({
                'type': 'emotional_support',
                'title': 'Emotional Stability Support',
                'description': 'I notice you experience significant emotional changes. Consider mindfulness techniques or breathing exercises to help manage emotional transitions.',
                'confidence': 0.9,
                'personalization_level': 'high'
            })
        
        # Communication style recommendations
        comm_style = communication_style.get('preferences', {})
        if comm_style.get('communication_style') == 'concise':
            recommendations.append({
                'type': 'communication',
                'title': 'Brief, Focused Responses',
                'description': 'You prefer concise communication. I\'ll keep my responses brief and focused on the most important points.',
                'confidence': 0.85,
                'personalization_level': 'high'
            })
        
        # Support needs recommendations
        support = support_needs.get('primary_support_need', 'general_support')
        if support == 'emotional_support':
            recommendations.append({
                'type': 'support',
                'title': 'Enhanced Emotional Support',
                'description': 'Your conversations suggest you need additional emotional support. Consider connecting with a support group or counselor who specializes in caregiver emotional wellness.',
                'confidence': 0.9,
                'personalization_level': 'high'
            })
        
        # Clinical dementia vs normal aging recommendations
        dementia_analysis = patterns.get('dementia_analysis', {})
        if dementia_analysis:
            assessment = dementia_analysis.get('assessment', 'insufficient_data')
            clinical_guidance = dementia_analysis.get('clinical_guidance', {})
            
            if assessment == 'concerning_patterns':
                recommendations.append({
                    'type': 'clinical',
                    'title': 'Professional Evaluation Recommended',
                    'description': f"Based on conversation patterns, {clinical_guidance.get('action', 'professional evaluation is recommended')}. Timeline: {clinical_guidance.get('timeline', 'within 2-4 weeks')}.",
                    'confidence': 0.95,
                    'personalization_level': 'critical',
                    'urgency': 'high',
                    'clinical_resources': clinical_guidance.get('resources', [])
                })
            elif assessment == 'mixed_patterns':
                recommendations.append({
                    'type': 'clinical',
                    'title': 'Close Monitoring Recommended',
                    'description': f"Mixed patterns detected - {clinical_guidance.get('action', 'close monitoring recommended')}. Continue documenting specific incidents.",
                    'confidence': 0.85,
                    'personalization_level': 'high',
                    'urgency': 'moderate',
                    'watch_for': clinical_guidance.get('watch_for', [])
                })
            elif assessment == 'normal_aging':
                recommendations.append({
                    'type': 'clinical',
                    'title': 'Normal Aging Patterns',
                    'description': f"Conversation patterns are consistent with normal aging. {clinical_guidance.get('action', 'Continue normal monitoring')}.",
                    'confidence': 0.8,
                    'personalization_level': 'high',
                    'urgency': 'low',
                    'reassurance': clinical_guidance.get('reassurance', [])
                })
        
        # Crisis intervention recommendations
        crisis = patterns.get('crisis_indicators', {})
        if crisis.get('crisis_level') == 'high':
            recommendations.append({
                'type': 'crisis_intervention',
                'title': 'Immediate Support Needed',
                'description': 'Your recent messages indicate you may be in crisis. Please consider reaching out to a mental health professional or crisis hotline immediately.',
                'confidence': 1.0,
                'personalization_level': 'critical',
                'urgency': 'immediate'
            })
        
        return recommendations
    
    def _calculate_insight_confidence(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate confidence in insights based on data quality and quantity."""
        if not messages:
            return 0.0
        
        # Base confidence on message count
        message_confidence = min(len(messages) / 10.0, 1.0)
        
        # Adjust for message quality (length, diversity)
        avg_length = np.mean([len(msg['content']) for msg in messages])
        length_confidence = min(avg_length / 100.0, 1.0)
        
        # Diversity confidence
        unique_sentiments = len(set(msg['sentiment'] for msg in messages))
        diversity_confidence = min(unique_sentiments / 4.0, 1.0)
        
        return round((message_confidence + length_confidence + diversity_confidence) / 3.0, 2)
    
    def _get_default_insights(self) -> Dict[str, Any]:
        """Return default insights when insufficient data is available."""
        return {
            'patterns': {'pattern': 'insufficient_data', 'confidence': 0.0},
            'emotional_arc': {'pattern': 'insufficient_data', 'confidence': 0.0},
            'communication_style': {'style': 'unknown', 'confidence': 0.0},
            'support_needs': {'primary_support_need': 'general_support', 'confidence': 0.0},
            'personalized_recommendations': [{
                'type': 'general',
                'title': 'Getting Started',
                'description': 'Continue our conversation so I can provide more personalized insights and recommendations.',
                'confidence': 0.5,
                'personalization_level': 'low'
            }],
            'insight_confidence': 0.0
        }
    
    # Helper methods
    def _sentiment_to_numeric(self, sentiments: List[str]) -> float:
        """Convert sentiment to numeric value for analysis."""
        sentiment_map = {
            'positive': 1.0,
            'neutral': 0.0,
            'sad': -0.8,
            'anxious': -0.6,
            'frustrated': -0.7,
            'stressed': -0.9
        }
        return np.mean([sentiment_map.get(s, 0.0) for s in sentiments])
    
    def _calculate_emotional_volatility(self, sentiments: List[str]) -> float:
        """Calculate emotional volatility."""
        if len(sentiments) < 2:
            return 0.0
        
        numeric_sentiments = [self._sentiment_to_numeric([s]) for s in sentiments]
        return np.std(numeric_sentiments)
    
    def _identify_emotional_cycles(self, sentiments: List[str]) -> List[str]:
        """Identify recurring emotional patterns."""
        if len(sentiments) < 4:
            return []
        
        cycles = []
        # Look for 2-3 message cycles
        for cycle_length in [2, 3]:
            for i in range(len(sentiments) - cycle_length):
                cycle = sentiments[i:i+cycle_length]
                if sentiments[i:i+cycle_length] == sentiments[i+cycle_length:i+2*cycle_length]:
                    cycles.append('_'.join(cycle))
        
        return list(set(cycles))
    
    def _assess_emotional_resilience(self, sentiments: List[str]) -> Dict[str, Any]:
        """Assess emotional resilience indicators."""
        if len(sentiments) < 3:
            return {'resilience_level': 'unknown', 'indicators': []}
        
        # Look for recovery patterns
        recovery_indicators = 0
        for i in range(len(sentiments) - 2):
            if sentiments[i] in ['sad', 'anxious', 'stressed'] and sentiments[i+1] in ['neutral', 'positive']:
                recovery_indicators += 1
        
        resilience_ratio = recovery_indicators / max(len(sentiments) - 2, 1)
        
        if resilience_ratio > 0.5:
            resilience_level = 'high'
        elif resilience_ratio > 0.2:
            resilience_level = 'moderate'
        else:
            resilience_level = 'low'
        
        return {
            'resilience_level': resilience_level,
            'recovery_indicators': recovery_indicators,
            'resilience_ratio': round(resilience_ratio, 2)
        }
    
    def _analyze_response_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how user responds to different types of interactions."""
        # This would analyze response patterns to different agent types
        # For now, return basic analysis
        return {
            'response_consistency': 0.7,
            'preferred_interaction_type': 'conversational'
        }
    
    def _identify_communication_preferences(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify specific communication preferences."""
        all_text = ' '.join([msg['content'] for msg in messages])
        
        preferences = {
            'communication_style': 'conversational',
            'detail_preference': 'moderate',
            'question_frequency': 'moderate'
        }
        
        # Analyze for specific preferences
        if '?' in all_text and all_text.count('?') / len(messages) > 0.3:
            preferences['question_frequency'] = 'high'
        elif all_text.count('?') / len(messages) < 0.1:
            preferences['question_frequency'] = 'low'
        
        return preferences
    
    def _count_emotional_words(self, text: str) -> Dict[str, int]:
        """Count emotional language indicators."""
        emotional_categories = {
            'intensity_words': ['very', 'extremely', 'completely', 'totally', 'absolutely'],
            'negative_words': ['terrible', 'awful', 'horrible', 'devastating', 'crushing'],
            'positive_words': ['wonderful', 'amazing', 'great', 'excellent', 'fantastic'],
            'uncertainty_words': ['maybe', 'perhaps', 'might', 'could', 'possibly']
        }
        
        word_counts = {}
        for category, words in emotional_categories.items():
            count = sum(1 for word in words if word in text.lower())
            word_counts[category] = count
        
        return word_counts
    
    def _analyze_dementia_vs_normal_aging(self, text: str) -> Dict[str, Any]:
        """Analyze conversation for dementia symptoms vs normal aging patterns."""
        
        # Clinical patterns based on the provided reference
        dementia_indicators = {
            'poor_judgment': {
                'keywords': ['bad decision', 'poor judgment', 'terrible choice', 'awful decision', 'wrong choice'],
                'context': 'repeated poor decision-making'
            },
            'budget_management': {
                'keywords': ['can\'t manage money', 'budget problems', 'financial confusion', 'bills unpaid', 'money trouble'],
                'context': 'inability to manage finances'
            },
            'time_disorientation': {
                'keywords': ['don\'t know what day', 'lost track of time', 'what season is it', 'confused about date'],
                'context': 'losing track of dates/seasons'
            },
            'conversation_difficulty': {
                'keywords': ['can\'t find words', 'hard to talk', 'conversation problems', 'speech difficulty'],
                'context': 'difficulty having conversations'
            },
            'misplacing_things': {
                'keywords': ['can\'t find things', 'lost and can\'t retrace', 'put things in wrong place', 'misplaced items'],
                'context': 'misplacing things without retracing steps'
            }
        }
        
        normal_aging_indicators = {
            'occasional_judgment': {
                'keywords': ['bad decision once', 'mistake sometimes', 'wrong choice occasionally'],
                'context': 'occasional poor decisions'
            },
            'occasional_payments': {
                'keywords': ['missed payment once', 'forgot bill sometimes', 'late payment occasionally'],
                'context': 'occasional missed payments'
            },
            'temporary_time_confusion': {
                'keywords': ['forgot day but remembered', 'confused briefly', 'temporary confusion'],
                'context': 'temporary time confusion with recovery'
            },
            'word_finding': {
                'keywords': ['forgot word but remembered', 'tip of tongue', 'word finding difficulty'],
                'context': 'occasional word-finding difficulties'
            },
            'occasional_misplacing': {
                'keywords': ['lost things sometimes', 'misplaced occasionally', 'found it later'],
                'context': 'occasional misplacing with recovery'
            }
        }
        
        # Analyze for dementia patterns
        dementia_scores = {}
        for category, data in dementia_indicators.items():
            score = sum(1 for keyword in data['keywords'] if keyword in text)
            dementia_scores[category] = {
                'score': score,
                'context': data['context'],
                'severity': 'high' if score >= 2 else 'moderate' if score >= 1 else 'low'
            }
        
        # Analyze for normal aging patterns
        normal_aging_scores = {}
        for category, data in normal_aging_indicators.items():
            score = sum(1 for keyword in data['keywords'] if keyword in text)
            normal_aging_scores[category] = {
                'score': score,
                'context': data['context'],
                'pattern': 'normal_aging' if score >= 1 else 'none'
            }
        
        # Calculate overall assessment
        total_dementia_score = sum(data['score'] for data in dementia_scores.values())
        total_normal_aging_score = sum(data['score'] for data in normal_aging_scores.values())
        
        # Determine assessment
        if total_dementia_score >= 3:
            assessment = 'concerning_patterns'
            recommendation = 'Consider professional evaluation for dementia assessment'
        elif total_dementia_score >= 1 and total_normal_aging_score >= 1:
            assessment = 'mixed_patterns'
            recommendation = 'Monitor patterns closely, may indicate early changes'
        elif total_normal_aging_score >= 2:
            assessment = 'normal_aging'
            recommendation = 'Patterns consistent with normal aging'
        else:
            assessment = 'insufficient_data'
            recommendation = 'Continue monitoring for patterns'
        
        return {
            'dementia_indicators': dementia_scores,
            'normal_aging_indicators': normal_aging_scores,
            'total_dementia_score': total_dementia_score,
            'total_normal_aging_score': total_normal_aging_score,
            'assessment': assessment,
            'recommendation': recommendation,
            'clinical_guidance': self._get_clinical_guidance(assessment, total_dementia_score, total_normal_aging_score)
        }
    
    def _get_clinical_guidance(self, assessment: str, dementia_score: int, normal_score: int) -> Dict[str, Any]:
        """Provide clinical guidance based on assessment."""
        
        guidance_map = {
            'concerning_patterns': {
                'urgency': 'high',
                'action': 'Professional evaluation recommended',
                'timeline': 'within 2-4 weeks',
                'resources': [
                    'Schedule appointment with neurologist or geriatrician',
                    'Consider neuropsychological testing',
                    'Document specific examples of concerning behaviors',
                    'Discuss with primary care physician'
                ],
                'red_flags': [
                    'Multiple cognitive domains affected',
                    'Progressive decline in function',
                    'Impact on daily activities',
                    'Safety concerns'
                ]
            },
            'mixed_patterns': {
                'urgency': 'moderate',
                'action': 'Close monitoring recommended',
                'timeline': 'ongoing observation',
                'resources': [
                    'Keep detailed log of concerning incidents',
                    'Monitor for progression of symptoms',
                    'Consider baseline cognitive assessment',
                    'Discuss concerns with healthcare provider'
                ],
                'watch_for': [
                    'Increase in frequency of concerning behaviors',
                    'New types of cognitive difficulties',
                    'Impact on independence',
                    'Family member concerns'
                ]
            },
            'normal_aging': {
                'urgency': 'low',
                'action': 'Continue normal monitoring',
                'timeline': 'routine check-ups',
                'resources': [
                    'Maintain healthy lifestyle habits',
                    'Stay mentally and socially active',
                    'Regular health check-ups',
                    'Family awareness of normal aging changes'
                ],
                'reassurance': [
                    'Changes are consistent with normal aging',
                    'No immediate cause for concern',
                    'Continue monitoring for any changes',
                    'Focus on maintaining cognitive health'
                ]
            },
            'insufficient_data': {
                'urgency': 'low',
                'action': 'Continue gathering information',
                'timeline': 'ongoing observation',
                'resources': [
                    'Document any concerning incidents',
                    'Monitor patterns over time',
                    'Discuss any concerns with healthcare provider',
                    'Stay informed about normal aging vs dementia'
                ],
                'guidance': [
                    'Insufficient information for assessment',
                    'Continue monitoring and documentation',
                    'Seek professional guidance if concerns arise',
                    'Focus on maintaining overall health'
                ]
            }
        }
        
        return guidance_map.get(assessment, guidance_map['insufficient_data'])
