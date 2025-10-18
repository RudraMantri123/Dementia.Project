"""Personalization engine for adaptive therapeutic interventions."""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
import json


class PersonalizationEngine:
    """Personalizes therapeutic approaches based on user response patterns."""
    
    def __init__(self):
        """Initialize personalization engine."""
        self.user_profiles = {}
        self.technique_effectiveness = defaultdict(lambda: defaultdict(float))
        self.response_patterns = defaultdict(list)
        
    def create_user_profile(self, user_id: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create or update user profile from conversation history."""
        profile = {
            'user_id': user_id,
            'dominant_emotions': self._identify_dominant_emotions(conversation_history),
            'preferred_techniques': self._identify_preferred_techniques(conversation_history),
            'communication_style': self._analyze_communication_style(conversation_history),
            'engagement_level': self._calculate_engagement(conversation_history),
            'response_to_interventions': self._track_intervention_responses(conversation_history),
            'crisis_triggers': self._identify_triggers(conversation_history),
            'support_needs': self._assess_support_needs(conversation_history),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def _identify_dominant_emotions(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Identify user's dominant emotional states."""
        emotion_counts = defaultdict(int)
        total = len(history)
        
        if total == 0:
            return {}
        
        for msg in history:
            sentiment = msg.get('sentiment', 'neutral')
            emotion_counts[sentiment] += 1
        
        return {
            emotion: count / total 
            for emotion, count in emotion_counts.items()
        }
    
    def _identify_preferred_techniques(self, history: List[Dict[str, Any]]) -> List[str]:
        """Identify which therapeutic techniques user responds to best."""
        technique_scores = defaultdict(float)
        
        for i, msg in enumerate(history):
            if msg.get('agent') == 'therapeutic':
                technique = msg.get('technique_used', 'general')
                
                # Check if next user message shows improvement
                if i + 1 < len(history):
                    current_sentiment = msg.get('sentiment', 'neutral')
                    next_sentiment = history[i + 1].get('sentiment', 'neutral')
                    
                    # Score improvement
                    if self._is_positive_shift(current_sentiment, next_sentiment):
                        technique_scores[technique] += 1.0
                    elif current_sentiment == next_sentiment:
                        technique_scores[technique] += 0.5
        
        # Return top techniques
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)
        return [technique for technique, _ in sorted_techniques[:3]]
    
    def _is_positive_shift(self, current: str, next_sentiment: str) -> bool:
        """Check if sentiment shifted positively."""
        emotion_hierarchy = {
            'positive': 5,
            'neutral': 4,
            'frustrated': 3,
            'anxious': 2,
            'sad': 1,
            'stressed': 0
        }
        
        current_score = emotion_hierarchy.get(current, 3)
        next_score = emotion_hierarchy.get(next_sentiment, 3)
        
        return next_score > current_score
    
    def _analyze_communication_style(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's communication style."""
        if not history:
            return {'style': 'unknown', 'avg_length': 0, 'complexity': 'medium'}
        
        user_messages = [msg for msg in history if msg.get('role') == 'user']
        
        if not user_messages:
            return {'style': 'unknown', 'avg_length': 0, 'complexity': 'medium'}
        
        avg_length = np.mean([len(msg.get('message', '')) for msg in user_messages])
        
        # Determine style
        if avg_length < 50:
            style = 'brief'
        elif avg_length < 150:
            style = 'conversational'
        else:
            style = 'expressive'
        
        # Assess complexity
        avg_words = np.mean([len(msg.get('message', '').split()) for msg in user_messages])
        if avg_words < 10:
            complexity = 'simple'
        elif avg_words < 30:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        return {
            'style': style,
            'avg_length': float(avg_length),
            'avg_words': float(avg_words),
            'complexity': complexity
        }
    
    def _calculate_engagement(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate user engagement metrics."""
        if not history:
            return {'level': 'low', 'score': 0.0}
        
        # Metrics
        total_messages = len([msg for msg in history if msg.get('role') == 'user'])
        response_rate = total_messages / max(1, len(history) / 2)
        
        # Check for follow-up questions
        follow_ups = sum(1 for msg in history 
                        if msg.get('role') == 'user' and '?' in msg.get('message', ''))
        
        # Engagement score
        score = min(1.0, (response_rate * 0.6 + (follow_ups / max(1, total_messages)) * 0.4))
        
        if score > 0.7:
            level = 'high'
        elif score > 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': float(score),
            'total_messages': total_messages,
            'follow_up_questions': follow_ups
        }
    
    def _track_intervention_responses(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track how user responds to different interventions."""
        responses = defaultdict(list)
        
        for i, msg in enumerate(history):
            intervention_type = msg.get('intervention_type')
            
            if intervention_type and i + 1 < len(history):
                current_sentiment = msg.get('sentiment', 'neutral')
                next_sentiment = history[i + 1].get('sentiment', 'neutral')
                
                improved = self._is_positive_shift(current_sentiment, next_sentiment)
                responses[intervention_type].append(1 if improved else 0)
        
        # Calculate success rates
        success_rates = {
            intervention: np.mean(outcomes) if outcomes else 0.5
            for intervention, outcomes in responses.items()
        }
        
        return {
            'success_rates': success_rates,
            'most_effective': max(success_rates.items(), key=lambda x: x[1])[0] if success_rates else 'unknown'
        }
    
    def _identify_triggers(self, history: List[Dict[str, Any]]) -> List[str]:
        """Identify potential crisis triggers from conversation patterns."""
        triggers = []
        trigger_keywords = {
            'overwhelm': ['overwhelmed', 'too much', "can't cope"],
            'isolation': ['alone', 'lonely', 'no one understands'],
            'exhaustion': ['exhausted', 'tired', 'drained', 'burnt out'],
            'grief': ['miss', 'lost', 'grief', 'mourning'],
            'guilt': ['guilty', 'fault', 'should have'],
            'financial': ['money', 'afford', 'expensive', 'cost']
        }
        
        for msg in history:
            message_text = msg.get('message', '').lower()
            sentiment = msg.get('sentiment', '')
            
            if sentiment in ['stressed', 'sad', 'anxious']:
                for trigger_type, keywords in trigger_keywords.items():
                    if any(keyword in message_text for keyword in keywords):
                        if trigger_type not in triggers:
                            triggers.append(trigger_type)
        
        return triggers
    
    def _assess_support_needs(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess user's primary support needs."""
        needs = {
            'emotional_support': 0,
            'practical_information': 0,
            'crisis_intervention': 0,
            'peer_connection': 0,
            'respite_care': 0
        }
        
        keywords_map = {
            'emotional_support': ['feel', 'emotion', 'overwhelmed', 'scared', 'sad'],
            'practical_information': ['how', 'what', 'when', 'information', 'guide'],
            'crisis_intervention': ['crisis', 'emergency', 'help', 'suicide', 'harm'],
            'peer_connection': ['others', 'group', 'support group', 'community'],
            'respite_care': ['break', 'rest', 'time off', 'respite', 'exhausted']
        }
        
        for msg in history:
            if msg.get('role') == 'user':
                message_text = msg.get('message', '').lower()
                
                for need_type, keywords in keywords_map.items():
                    if any(keyword in message_text for keyword in keywords):
                        needs[need_type] += 1
        
        # Normalize
        total = sum(needs.values()) or 1
        needs = {k: v / total for k, v in needs.items()}
        
        primary_need = max(needs.items(), key=lambda x: x[1])[0]
        
        return {
            'distribution': needs,
            'primary': primary_need
        }
    
    def personalize_response(self, user_id: str, base_response: str, 
                           context: Dict[str, Any]) -> str:
        """Personalize response based on user profile."""
        profile = self.user_profiles.get(user_id)
        
        if not profile:
            return base_response
        
        # Adjust based on communication style
        style = profile['communication_style']['style']
        
        if style == 'brief':
            # Keep response concise
            sentences = base_response.split('. ')
            return '. '.join(sentences[:2]) + '.'
        elif style == 'expressive':
            # Add empathetic elaboration
            return base_response
        
        return base_response
    
    def suggest_optimal_timing(self, user_id: str) -> Dict[str, Any]:
        """Suggest optimal timing for interventions based on patterns."""
        profile = self.user_profiles.get(user_id)
        
        if not profile:
            return {
                'next_checkin': '24 hours',
                'optimal_time': 'afternoon',
                'frequency': 'daily'
            }
        
        engagement = profile['engagement_level']['level']
        
        if engagement == 'high':
            frequency = 'multiple_daily'
            next_checkin = '8 hours'
        elif engagement == 'medium':
            frequency = 'daily'
            next_checkin = '24 hours'
        else:
            frequency = 'every_2_days'
            next_checkin = '48 hours'
        
        return {
            'next_checkin': next_checkin,
            'optimal_time': 'afternoon',
            'frequency': frequency
        }
    
    def get_personalized_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get personalized recommendations for user."""
        profile = self.user_profiles.get(user_id)
        
        if not profile:
            return {
                'techniques': ['cbt', 'mindfulness'],
                'focus_areas': ['stress_management'],
                'avoid': []
            }
        
        return {
            'techniques': profile['preferred_techniques'],
            'focus_areas': self._recommend_focus_areas(profile),
            'triggers_to_avoid': profile['crisis_triggers'],
            'communication_approach': profile['communication_style']['style'],
            'primary_need': profile['support_needs']['primary']
        }
    
    def _recommend_focus_areas(self, profile: Dict[str, Any]) -> List[str]:
        """Recommend focus areas based on profile."""
        focus_areas = []
        
        dominant = profile['dominant_emotions']
        
        if dominant.get('stressed', 0) > 0.3:
            focus_areas.append('stress_management')
        if dominant.get('sad', 0) > 0.3:
            focus_areas.append('grief_processing')
        if dominant.get('anxious', 0) > 0.3:
            focus_areas.append('anxiety_reduction')
        
        if 'isolation' in profile['crisis_triggers']:
            focus_areas.append('social_connection')
        
        return focus_areas if focus_areas else ['general_wellbeing']

