"""Test ML intervention system."""

import sys
sys.path.append('.')

from src.ml.intervention_predictor import InterventionPredictor
from src.agents.analyst_agent import AnalystAgent

# Create sample conversation
conversation = [
    {'role': 'user', 'content': 'I feel overwhelmed', 'message': 'I feel overwhelmed', 'sentiment': 'stressed', 'timestamp': '2025-10-17T10:00:00'},
    {'role': 'assistant', 'content': 'I understand', 'message': 'I understand', 'sentiment': 'neutral', 'timestamp': '2025-10-17T10:01:00'},
    {'role': 'user', 'content': 'I am so sad', 'message': 'I am so sad', 'sentiment': 'sad', 'timestamp': '2025-10-17T10:02:00'},
]

# Test intervention predictor directly
print("Testing Intervention Predictor...")
predictor = InterventionPredictor()

sentiment_data = {
    'sentiment_breakdown': {
        'stressed': 1,
        'sad': 1,
        'neutral': 0,
        'anxious': 0,
        'frustrated': 0,
        'positive': 0
    },
    'intensity_metrics': {
        'avg_intensity': 7.0
    },
    'stability_metrics': {
        'stability_score': 0.6
    }
}

features = predictor.extract_features(conversation, sentiment_data)
print(f"Features extracted: {features}")

burnout_risk = predictor.predict_burnout_risk(features)
print(f"Burnout risk: {burnout_risk}")

crisis_risk = predictor.predict_crisis_risk(features)
print(f"Crisis risk: {crisis_risk}")

intervention = predictor.recommend_intervention(features)
print(f"Intervention: {intervention}")

# Test analyst agent
print("\n\nTesting Analyst Agent...")
analyst = AnalystAgent()

risk_assessment = analyst.get_risk_assessment(conversation)
print(f"Risk assessment: {risk_assessment}")

print("\n\nAll tests passed!")

