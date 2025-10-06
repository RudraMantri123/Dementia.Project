"""Predictive stress modeling with ML."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.database.models import Conversation, UserAnalytics, CognitiveExerciseResult
from src.database.connection import get_db_manager


class PredictiveStressModeler:
    """Predicts future stress levels using machine learning."""

    def __init__(self, model_path: str = "data/models/stress_predictor.pkl"):
        """
        Initialize predictive stress modeler.

        Args:
            model_path: Path to save/load model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.db_manager = get_db_manager()

    def extract_features(self, user_id: str, days: int = 30) -> Optional[np.ndarray]:
        """
        Extract features for stress prediction.

        Args:
            user_id: User identifier
            days: Number of days of historical data

        Returns:
            Feature array or None
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        with self.db_manager.get_session() as session:
            # Get conversations
            conversations = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.timestamp >= start_date,
                Conversation.role == 'user'
            ).all()

            if not conversations:
                return None

            # Get exercises
            exercises = session.query(CognitiveExerciseResult).filter(
                CognitiveExerciseResult.user_id == user_id,
                CognitiveExerciseResult.timestamp >= start_date
            ).all()

            # Extract features
            features = self._compute_features(conversations, exercises)

            return np.array(features).reshape(1, -1)

    def _compute_features(
        self,
        conversations: List[Conversation],
        exercises: List[CognitiveExerciseResult]
    ) -> List[float]:
        """Compute feature vector from data."""
        features = []

        # Conversation features
        if conversations:
            # Sentiment features
            sentiments = [c.sentiment_score for c in conversations if c.sentiment_score is not None]
            if sentiments:
                features.extend([
                    np.mean(sentiments),  # Average sentiment
                    np.std(sentiments),   # Sentiment variability
                    min(sentiments),      # Minimum sentiment
                    max(sentiments)       # Maximum sentiment
                ])
            else:
                features.extend([0.5, 0, 0, 1])  # Default values

            # Engagement features
            message_lengths = [len(c.content) for c in conversations]
            features.extend([
                len(conversations),          # Message count
                np.mean(message_lengths),    # Average message length
                np.std(message_lengths)      # Message length variability
            ])

            # Temporal features
            timestamps = [c.timestamp.hour for c in conversations]
            features.extend([
                len(set(timestamps)),        # Active hours diversity
                np.std(timestamps)           # Time variability
            ])

            # Negative emotion indicators
            negative_sentiments = [s for s in sentiments if s < 0.4]
            features.append(len(negative_sentiments) / len(conversations))  # Negative sentiment ratio

        else:
            features.extend([0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # Defaults

        # Exercise features
        if exercises:
            scores = [e.performance_score for e in exercises if e.performance_score is not None]
            if scores:
                features.extend([
                    len(exercises),          # Exercise count
                    np.mean(scores),         # Average performance
                    np.std(scores),          # Performance variability
                    min(scores),             # Minimum performance
                    max(scores)              # Maximum performance
                ])
            else:
                features.extend([0, 0.5, 0, 0, 1])
        else:
            features.extend([0, 0.5, 0, 0, 1])  # Defaults

        # Additional stress indicators
        # Repetition score (simplified)
        if conversations:
            contents = [c.content.lower() for c in conversations]
            unique_ratio = len(set(contents)) / len(contents)
            features.append(1 - unique_ratio)  # Higher = more repetition = potential stress
        else:
            features.append(0)

        return features

    def train_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train stress prediction model.

        Args:
            training_data: List of training samples with features and labels

        Returns:
            Dictionary with training metrics
        """
        if len(training_data) < 10:
            raise ValueError("Insufficient training data (minimum 10 samples required)")

        # Prepare data
        X = np.array([sample['features'] for sample in training_data])
        y = np.array([sample['stress_level'] for sample in training_data])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model (using RandomForest)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save model
        self._save_model()

        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'r2_score': float(r2),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def predict_stress(
        self,
        user_id: str,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Predict stress level for upcoming period.

        Args:
            user_id: User identifier
            days_ahead: Number of days to predict ahead

        Returns:
            Dictionary with prediction
        """
        if self.model is None:
            # Try to load existing model
            if not self._load_model():
                return {
                    'error': 'Model not trained. Please train model first.',
                    'prediction': None
                }

        # Extract features
        features = self.extract_features(user_id, days=30)

        if features is None:
            return {
                'error': 'Insufficient data for prediction',
                'prediction': None
            }

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]

        # Ensure prediction is in valid range [0, 1]
        prediction = np.clip(prediction, 0, 1)

        # Get confidence estimate (using prediction variance from trees)
        if hasattr(self.model, 'estimators_'):
            predictions = [tree.predict(features_scaled)[0] for tree in self.model.estimators_]
            confidence = 1 - np.std(predictions)
        else:
            confidence = 0.7  # Default confidence

        # Generate recommendation
        recommendation = self._get_stress_recommendation(prediction)

        return {
            'user_id': user_id,
            'predicted_stress_level': float(prediction),
            'confidence': float(confidence),
            'days_ahead': days_ahead,
            'risk_level': self._categorize_stress(prediction),
            'recommendation': recommendation,
            'prediction_date': (datetime.utcnow() + timedelta(days=days_ahead)).isoformat()
        }

    def _categorize_stress(self, stress_level: float) -> str:
        """Categorize stress level."""
        if stress_level >= 0.75:
            return 'high'
        elif stress_level >= 0.5:
            return 'medium'
        elif stress_level >= 0.25:
            return 'low'
        else:
            return 'minimal'

    def _get_stress_recommendation(self, stress_level: float) -> str:
        """Get recommendation based on predicted stress."""
        if stress_level >= 0.75:
            return "High stress predicted. Recommend immediate caregiver consultation and increased support."
        elif stress_level >= 0.5:
            return "Moderate stress predicted. Monitor closely and consider additional support activities."
        elif stress_level >= 0.25:
            return "Low stress predicted. Continue current support routine."
        else:
            return "Minimal stress predicted. Maintain current engagement level."

    def _save_model(self):
        """Save model and scaler to disk."""
        import os
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)

    def _load_model(self) -> bool:
        """Load model from disk. Returns True if successful."""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
            return True
        except FileNotFoundError:
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}

        feature_names = [
            'avg_sentiment', 'sentiment_std', 'min_sentiment', 'max_sentiment',
            'message_count', 'avg_message_length', 'message_length_std',
            'active_hours_diversity', 'time_variability', 'negative_ratio',
            'exercise_count', 'avg_performance', 'performance_std',
            'min_performance', 'max_performance', 'repetition_score'
        ]

        importance = self.model.feature_importances_

        return {name: float(imp) for name, imp in zip(feature_names, importance)}
