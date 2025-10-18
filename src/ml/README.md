# ML Intervention Layer

This module implements advanced machine learning models for intelligent therapeutic interventions in the Dementia Support Chatbot.

## Components

### 1. Intervention Predictor (`intervention_predictor.py`)

**Purpose**: Predicts burnout risk, crisis probability, and recommends optimal therapeutic interventions.

**Features**:
- **Burnout Risk Prediction**: 4-level classification (low, moderate, high, critical)
- **Crisis Detection**: Binary classification with probability scores
- **Intervention Recommendation**: 4 intervention types (CBT, Mindfulness, Peer Support, Crisis Intervention)

**Models**:
- Gradient Boosting for burnout prediction (n_estimators=200, learning_rate=0.1)
- Random Forest for crisis detection (n_estimators=150, max_depth=10)
- Gradient Boosting for intervention recommendation (n_estimators=100)

**Feature Extraction** (10 features):
1. Negative emotion ratio
2. Emotional intensity (0-10 scale)
3. Sentiment stability (0-1)
4. Conversation frequency (messages/day)
5. Negative trend (slope of emotional decline)
6. Stress ratio
7. Sadness ratio
8. Anxiety ratio
9. Support-seeking behavior
10. Duration (days in system)

### 2. Personalization Engine (`personalization_engine.py`)

**Purpose**: Creates personalized user profiles and adapts therapeutic approaches based on individual response patterns.

**Features**:
- **User Profiling**: Tracks dominant emotions, preferred techniques, communication style
- **Technique Effectiveness**: Monitors which therapeutic approaches work best for each user
- **Trigger Identification**: Detects crisis triggers (overwhelm, isolation, exhaustion, grief, guilt, financial)
- **Support Needs Assessment**: Identifies primary needs (emotional, practical, crisis, peer, respite)
- **Optimal Timing**: Suggests best times for interventions based on engagement patterns

**Personalization Dimensions**:
- Communication style: brief, conversational, expressive
- Complexity level: simple, medium, complex
- Engagement level: low, medium, high
- Preferred techniques: CBT, mindfulness, validation, etc.

### 3. Integration with Analyst Agent

The ML layer is integrated into the `AnalystAgent` through three main methods:

```python
# Complete intervention plan
ml_plan = analyst.get_ml_intervention_plan(conversation_history, session_id)

# Risk assessment only
risk = analyst.get_risk_assessment(conversation_history)

# Personalized intervention
intervention = analyst.get_personalized_intervention(conversation_history, session_id)
```

## API Endpoints

### GET /ml/intervention-plan/{session_id}
Returns comprehensive intervention plan including:
- Burnout risk assessment
- Crisis risk evaluation
- Recommended interventions
- User profile
- Personalized recommendations
- Optimal timing suggestions

### GET /ml/risk-assessment/{session_id}
Returns risk assessment including:
- Burnout risk (level, probability, factors)
- Crisis risk (status, urgency, probability)
- Action required flag

### GET /ml/personalized-intervention/{session_id}
Returns personalized intervention recommendation:
- Intervention type and details
- Confidence score
- Preferred techniques
- Communication approach

## Usage Example

```python
from src.ml import InterventionPredictor, PersonalizationEngine

# Initialize
predictor = InterventionPredictor()
personalizer = PersonalizationEngine()

# Extract features from conversation
features = predictor.extract_features(conversation_history, sentiment_data)

# Get predictions
burnout_risk = predictor.predict_burnout_risk(features)
crisis_risk = predictor.predict_crisis_risk(features)
intervention = predictor.recommend_intervention(features)

# Create user profile
profile = personalizer.create_user_profile(user_id, conversation_history)

# Get personalized recommendations
recommendations = personalizer.get_personalized_recommendations(user_id)
```

## Model Training

Models are trained with synthetic but realistic data representing common caregiver patterns. The system includes:
- 1000 training samples per model
- Realistic feature distributions
- Balanced class representations
- Cross-validation for hyperparameter tuning

Models are automatically saved to `data/models/intervention_model.pkl` and loaded on initialization.

## Performance Characteristics

- **Inference Time**: <100ms per prediction
- **Memory Footprint**: ~50MB (all models loaded)
- **Feature Extraction**: <50ms
- **Profile Creation**: <100ms

## Future Enhancements

1. **Online Learning**: Continuous model updates from user feedback
2. **Multi-modal Input**: Integrate voice prosody and facial expressions
3. **Longitudinal Analysis**: Track outcomes over weeks/months
4. **Federated Learning**: Privacy-preserving learning across users
5. **Transfer Learning**: Adapt to new user populations quickly

