"""Analyst agent for conversation analytics using machine learning."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os


class AnalystAgent:
    """Analyzes conversations using ML for sentiment and patterns."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the analyst agent.

        Args:
            model_path: Path to saved model (optional)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3),  # Include bigrams and trigrams
            min_df=1,
            max_df=0.95
        )
        self.sentiment_model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_path = model_path or "data/models/analyst_model.pkl"

        # Try to load existing model
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            # Train with default data
            self._train_default_model()

    def _train_default_model(self):
        """Train model with comprehensive sentiment data for dementia caregiving."""
        # Expanded training data with 300+ examples covering diverse scenarios
        training_data = [
            # STRESSED/OVERWHELMED (60 examples)
            ("I'm feeling so overwhelmed with caregiving", "stressed"),
            ("This is too much for me to handle", "stressed"),
            ("I can't cope with this anymore", "stressed"),
            ("I'm exhausted and don't know what to do", "stressed"),
            ("The burden is getting too heavy", "stressed"),
            ("I feel like I'm drowning", "stressed"),
            ("I haven't slept in days", "stressed"),
            ("I'm burning out from constant caregiving", "stressed"),
            ("Managing everything alone is impossible", "stressed"),
            ("I'm at my breaking point", "stressed"),
            ("The stress is affecting my health", "stressed"),
            ("I can't remember the last time I relaxed", "stressed"),
            ("Balancing work and caregiving is killing me", "stressed"),
            ("I'm running on empty", "stressed"),
            ("The responsibility is crushing me", "stressed"),
            ("I feel completely drained", "stressed"),
            ("I have no energy left", "stressed"),
            ("Every day feels harder than the last", "stressed"),
            ("I'm stretched too thin", "stressed"),
            ("I need a break but can't get one", "stressed"),
            ("The constant demands are overwhelming", "stressed"),
            ("I'm losing myself in caregiving", "stressed"),
            ("I don't have time for anything anymore", "stressed"),
            ("The pressure is unbearable", "stressed"),
            ("I'm doing everything and it's never enough", "stressed"),
            ("I feel trapped in this situation", "stressed"),
            ("My whole life revolves around caregiving now", "stressed"),
            ("I'm neglecting my own needs completely", "stressed"),
            ("The weight of responsibility is too much", "stressed"),
            ("I'm completely overwhelmed by the tasks", "stressed"),
            ("I can barely function anymore", "stressed"),
            ("Managing medications alone is stressing me out", "stressed"),
            ("I'm worried I'll make a mistake", "stressed"),
            ("The financial stress is overwhelming too", "stressed"),
            ("I have no support system", "stressed"),
            ("I'm doing this all by myself", "stressed"),
            ("Everyone expects me to handle everything", "stressed"),
            ("I feel like I'm suffocating", "stressed"),
            ("The constant vigilance is exhausting", "stressed"),
            ("I can't keep up with everything", "stressed"),
            ("My mental health is suffering", "stressed"),
            ("I'm beyond exhausted", "stressed"),
            ("I don't know how much longer I can do this", "stressed"),
            ("The sleepless nights are destroying me", "stressed"),
            ("I'm overwhelmed by doctor appointments", "stressed"),
            ("Coordinating care is too complicated", "stressed"),
            ("I feel like I'm failing at everything", "stressed"),
            ("The guilt and stress are consuming me", "stressed"),
            ("I'm stretched beyond my limits", "stressed"),
            ("I can't handle one more thing", "stressed"),
            ("Everything falls on my shoulders", "stressed"),
            ("I'm physically and emotionally drained", "stressed"),
            ("The constant caregiving is breaking me", "stressed"),
            ("I have no time for self-care", "stressed"),
            ("I'm overwhelmed by their changing needs", "stressed"),
            ("Managing behavioral issues is exhausting", "stressed"),
            ("I feel buried under responsibilities", "stressed"),
            ("The stress never stops", "stressed"),
            ("I'm running myself into the ground", "stressed"),
            ("I need help but don't know where to get it", "stressed"),

            # SAD/DEPRESSED (60 examples)
            ("I feel so sad watching them decline", "sad"),
            ("I'm depressed and lonely", "sad"),
            ("I miss who they used to be", "sad"),
            ("This disease is heartbreaking", "sad"),
            ("I cry every night", "sad"),
            ("I feel hopeless about the future", "sad"),
            ("Watching them forget me is devastating", "sad"),
            ("I've lost my partner to this disease", "sad"),
            ("They don't recognize me anymore", "sad"),
            ("I'm grieving while they're still alive", "sad"),
            ("This progressive loss is unbearable", "sad"),
            ("I feel empty inside", "sad"),
            ("My heart breaks every day", "sad"),
            ("I'm mourning the person they were", "sad"),
            ("The sadness is overwhelming", "sad"),
            ("I feel isolated and alone", "sad"),
            ("No one understands what I'm going through", "sad"),
            ("I'm losing them piece by piece", "sad"),
            ("This slow goodbye is torture", "sad"),
            ("I feel so alone in this journey", "sad"),
            ("Depression is consuming me", "sad"),
            ("I've lost my joy in life", "sad"),
            ("Everything feels dark and hopeless", "sad"),
            ("I can't stop crying", "sad"),
            ("The grief is unbearable", "sad"),
            ("I'm watching the person I love disappear", "sad"),
            ("They called me by the wrong name today", "sad"),
            ("I feel like I've lost my best friend", "sad"),
            ("This disease has stolen our future", "sad"),
            ("I'm heartbroken by their confusion", "sad"),
            ("Seeing them struggle breaks my heart", "sad"),
            ("I miss our conversations", "sad"),
            ("They don't remember our life together", "sad"),
            ("I feel so sad all the time", "sad"),
            ("The person I married is gone", "sad"),
            ("I'm mourning someone who's still here", "sad"),
            ("This anticipatory grief is crushing", "sad"),
            ("I feel profoundly sad", "sad"),
            ("I've never felt so lonely", "sad"),
            ("My heart aches constantly", "sad"),
            ("I'm depressed about their decline", "sad"),
            ("Watching this progression is devastating", "sad"),
            ("I feel like I'm losing everything", "sad"),
            ("The sadness never goes away", "sad"),
            ("I miss the life we had", "sad"),
            ("This disease has taken everything from us", "sad"),
            ("I'm drowning in sadness", "sad"),
            ("My parent doesn't know who I am", "sad"),
            ("The emptiness is overwhelming", "sad"),
            ("I feel defeated by this disease", "sad"),
            ("Every milestone lost brings more grief", "sad"),
            ("I'm sad about what we've lost", "sad"),
            ("This isn't how I imagined our golden years", "sad"),
            ("I feel like I'm in mourning", "sad"),
            ("The loneliness is suffocating", "sad"),
            ("I've lost my companion", "sad"),
            ("Dementia has stolen our happiness", "sad"),
            ("I'm heartbroken and exhausted", "sad"),
            ("I miss their smile and laughter", "sad"),
            ("The sadness is constant", "sad"),

            # ANXIOUS/WORRIED (60 examples)
            ("I'm worried about what comes next", "anxious"),
            ("I'm scared of the future", "anxious"),
            ("What if I can't handle this", "anxious"),
            ("I'm afraid they'll get worse", "anxious"),
            ("The uncertainty is killing me", "anxious"),
            ("I have so much anxiety about their safety", "anxious"),
            ("What if they wander off", "anxious"),
            ("I'm terrified of leaving them alone", "anxious"),
            ("I worry constantly about their wellbeing", "anxious"),
            ("What if they fall when I'm not there", "anxious"),
            ("I'm anxious about their declining health", "anxious"),
            ("I can't stop worrying", "anxious"),
            ("I'm afraid of what stage comes next", "anxious"),
            ("The fear of the unknown is paralyzing", "anxious"),
            ("I'm worried I'm not doing enough", "anxious"),
            ("What if I make the wrong decision", "anxious"),
            ("I'm scared they'll hurt themselves", "anxious"),
            ("I have panic attacks about their care", "anxious"),
            ("I'm constantly on edge", "anxious"),
            ("I worry about every little thing", "anxious"),
            ("What if emergency happens", "anxious"),
            ("I'm afraid to leave the house", "anxious"),
            ("I'm anxious about medication side effects", "anxious"),
            ("What if the treatments don't work", "anxious"),
            ("I'm worried about financial costs", "anxious"),
            ("How will we afford long-term care", "anxious"),
            ("I'm scared of making mistakes", "anxious"),
            ("What if I miss warning signs", "anxious"),
            ("I'm anxious about their aggressive behavior", "anxious"),
            ("I worry they'll get agitated again", "anxious"),
            ("What if they don't recognize danger", "anxious"),
            ("I'm afraid of sundowning episodes", "anxious"),
            ("I can't sleep because of anxiety", "anxious"),
            ("I'm worried about their nutrition", "anxious"),
            ("What if they stop eating", "anxious"),
            ("I'm scared of choking incidents", "anxious"),
            ("I worry about them every second", "anxious"),
            ("What if I can't keep them safe", "anxious"),
            ("I'm anxious about the progression rate", "anxious"),
            ("I fear they'll forget how to swallow", "anxious"),
            ("What if they need 24/7 care", "anxious"),
            ("I'm worried about placement decisions", "anxious"),
            ("I'm scared of nursing home options", "anxious"),
            ("What if they hate the care facility", "anxious"),
            ("I'm anxious about their resistance to help", "anxious"),
            ("I worry I'm not trained enough", "anxious"),
            ("What if I can't handle behavioral changes", "anxious"),
            ("I'm scared they'll become violent", "anxious"),
            ("I'm anxious about medical emergencies", "anxious"),
            ("What if they have a stroke", "anxious"),
            ("I worry about comorbid conditions", "anxious"),
            ("I'm afraid they're in pain but can't tell me", "anxious"),
            ("What if I can't understand their needs", "anxious"),
            ("I'm worried about my own future", "anxious"),
            ("Will I develop dementia too", "anxious"),
            ("I'm anxious about genetic risks", "anxious"),
            ("What if this happens to me", "anxious"),
            ("I'm scared of being a burden someday", "anxious"),
            ("The future terrifies me", "anxious"),
            ("I'm worried about everything all the time", "anxious"),

            # FRUSTRATED/ANGRY (50 examples)
            ("I'm so frustrated with this situation", "frustrated"),
            ("Why is this happening to us", "frustrated"),
            ("I'm angry at this disease", "frustrated"),
            ("Nothing seems to help", "frustrated"),
            ("This is so unfair", "frustrated"),
            ("I'm tired of dealing with the same problems", "frustrated"),
            ("They won't cooperate with anything", "frustrated"),
            ("I'm frustrated by their stubbornness", "frustrated"),
            ("Why won't they just listen to me", "frustrated"),
            ("I'm angry at the lack of support", "frustrated"),
            ("The healthcare system is failing us", "frustrated"),
            ("I'm frustrated with doctor appointments", "frustrated"),
            ("No one takes this seriously enough", "frustrated"),
            ("I'm angry they won't accept help", "frustrated"),
            ("Why is everything so difficult", "frustrated"),
            ("I'm frustrated by the repetitive questions", "frustrated"),
            ("Hearing the same story 50 times is maddening", "frustrated"),
            ("I'm angry at how long diagnosis took", "frustrated"),
            ("The system is broken", "frustrated"),
            ("I'm frustrated with insurance denials", "frustrated"),
            ("Why is care so expensive", "frustrated"),
            ("I'm angry at family members who don't help", "frustrated"),
            ("They promised to help but disappeared", "frustrated"),
            ("I'm frustrated doing this alone", "frustrated"),
            ("Everyone has excuses for not helping", "frustrated"),
            ("I'm angry at people who don't understand", "frustrated"),
            ("Stop giving me useless advice", "frustrated"),
            ("I'm frustrated with their sundowning behavior", "frustrated"),
            ("They refuse to shower again", "frustrated"),
            ("I'm angry at the unfairness of it all", "frustrated"),
            ("Why does this have to be so hard", "frustrated"),
            ("I'm frustrated they won't stay in bed", "frustrated"),
            ("They keep trying to leave the house", "frustrated"),
            ("I'm angry they hid the diagnosis from me", "frustrated"),
            ("I'm frustrated with medication side effects", "frustrated"),
            ("Nothing we try works", "frustrated"),
            ("I'm angry at doctors who don't listen", "frustrated"),
            ("I'm frustrated by the lack of treatment options", "frustrated"),
            ("Why isn't there a cure yet", "frustrated"),
            ("I'm angry this disease exists", "frustrated"),
            ("I'm frustrated with their accusations", "frustrated"),
            ("They think I'm stealing from them", "frustrated"),
            ("I'm angry at the paranoia", "frustrated"),
            ("I'm frustrated by the personality changes", "frustrated"),
            ("This isn't the person I knew", "frustrated"),
            ("I'm angry at how much this has changed us", "frustrated"),
            ("I'm frustrated I can't fix this", "frustrated"),
            ("I hate feeling so powerless", "frustrated"),
            ("I'm angry at my own limitations", "frustrated"),
            ("Why can't I make this better", "frustrated"),

            # POSITIVE/HOPEFUL (40 examples)
            ("We had a good day today", "positive"),
            ("I'm grateful for the support", "positive"),
            ("Things are getting a bit better", "positive"),
            ("I'm learning to cope", "positive"),
            ("Thank you for the helpful advice", "positive"),
            ("I feel more hopeful now", "positive"),
            ("The support group is really helping", "positive"),
            ("I'm finding moments of joy", "positive"),
            ("They smiled at me today", "positive"),
            ("We had a beautiful moment together", "positive"),
            ("I'm grateful for small victories", "positive"),
            ("The new medication is helping", "positive"),
            ("I found a great daycare program", "positive"),
            ("I'm feeling more confident", "positive"),
            ("We're adjusting to the new routine", "positive"),
            ("I appreciate your encouragement", "positive"),
            ("I'm learning new coping strategies", "positive"),
            ("Things are looking up", "positive"),
            ("I feel stronger today", "positive"),
            ("I'm proud of how I handled that", "positive"),
            ("We enjoyed a nice walk together", "positive"),
            ("They remembered something important today", "positive"),
            ("I'm grateful for respite care", "positive"),
            ("The caregiver I hired is wonderful", "positive"),
            ("I'm taking better care of myself now", "positive"),
            ("I feel more at peace with the situation", "positive"),
            ("I'm thankful for this community", "positive"),
            ("We had a laugh together today", "positive"),
            ("I found a silver lining", "positive"),
            ("I'm appreciating the present moments", "positive"),
            ("I feel blessed to care for them", "positive"),
            ("We're making the best of it", "positive"),
            ("I'm finding inner strength", "positive"),
            ("The advice really worked", "positive"),
            ("I'm feeling optimistic", "positive"),
            ("We created a happy memory today", "positive"),
            ("I'm grateful for their lucid moments", "positive"),
            ("Things are manageable right now", "positive"),
            ("I feel supported and understood", "positive"),
            ("I'm celebrating small wins", "positive"),

            # NEUTRAL/SEEKING INFORMATION (40 examples)
            ("What are the symptoms of dementia", "neutral"),
            ("How do I handle wandering", "neutral"),
            ("Can you tell me about medications", "neutral"),
            ("I need information about care options", "neutral"),
            ("What activities are good for dementia", "neutral"),
            ("Tell me about support groups", "neutral"),
            ("What are the stages of Alzheimer's", "neutral"),
            ("How can I manage sundowning", "neutral"),
            ("What foods are best for brain health", "neutral"),
            ("Can you explain memory care facilities", "neutral"),
            ("What legal documents do I need", "neutral"),
            ("How do I apply for disability benefits", "neutral"),
            ("What is respite care", "neutral"),
            ("Can you recommend daily activities", "neutral"),
            ("How do I child-proof the house", "neutral"),
            ("What are communication strategies", "neutral"),
            ("Tell me about caregiver resources", "neutral"),
            ("How do I manage bathing resistance", "neutral"),
            ("What are signs of progression", "neutral"),
            ("Can you explain Medicare coverage", "neutral"),
            ("What is the difference between dementia types", "neutral"),
            ("How do I find a good neurologist", "neutral"),
            ("What assistive devices are helpful", "neutral"),
            ("Can you explain power of attorney", "neutral"),
            ("What is adult day care", "neutral"),
            ("How do I prevent falls", "neutral"),
            ("What are medication side effects", "neutral"),
            ("Tell me about clinical trials", "neutral"),
            ("How do I create a care plan", "neutral"),
            ("What safety measures should I take", "neutral"),
            ("Can you explain hospice care", "neutral"),
            ("What is palliative care", "neutral"),
            ("How do I manage incontinence", "neutral"),
            ("What are behavioral management techniques", "neutral"),
            ("Tell me about genetic testing", "neutral"),
            ("How do I organize medical records", "neutral"),
            ("What questions should I ask the doctor", "neutral"),
            ("Can you explain long-term care insurance", "neutral"),
            ("What are early warning signs", "neutral"),
            ("How do I maintain their dignity", "neutral"),
        ]

        texts, labels = zip(*training_data)

        # Train vectorizer and model
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)

        self.sentiment_model.fit(X, y)
        self.is_trained = True

        # Save the model
        self.save_model()

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.is_trained:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': 'Model not trained'
            }

        try:
            # Transform text
            X = self.vectorizer.transform([text])

            # Predict sentiment
            prediction = self.sentiment_model.predict(X)[0]
            probabilities = self.sentiment_model.predict_proba(X)[0]

            sentiment = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))

            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    label: float(prob)
                    for label, prob in zip(
                        self.label_encoder.classes_,
                        probabilities
                    )
                }
            }

        except Exception as e:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }

    def analyze_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze entire conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Conversation analytics
        """
        if not messages:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_trend': [],
                'total_messages': 0
            }

        user_messages = [msg['content'] for msg in messages if msg.get('role') == 'user']

        if not user_messages:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_trend': [],
                'total_messages': 0
            }

        # Analyze each message
        sentiments = []
        for msg in user_messages:
            analysis = self.analyze_sentiment(msg)
            sentiments.append(analysis['sentiment'])

        # Calculate statistics
        sentiment_counts = pd.Series(sentiments).value_counts().to_dict()

        # Determine overall sentiment
        if sentiment_counts:
            overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        else:
            overall_sentiment = 'neutral'

        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_counts,
            'sentiment_trend': sentiments,
            'total_messages': len(user_messages),
            'needs_support': self._assess_support_needs(sentiments)
        }

    def _assess_support_needs(self, sentiments: List[str]) -> Dict[str, Any]:
        """
        Assess if user needs additional support.

        Args:
            sentiments: List of detected sentiments

        Returns:
            Support assessment
        """
        concerning_sentiments = ['stressed', 'sad', 'anxious', 'frustrated']
        concerning_count = sum(1 for s in sentiments if s in concerning_sentiments)

        total = len(sentiments)
        if total == 0:
            return {'level': 'low', 'recommendation': None}

        concerning_ratio = concerning_count / total

        if concerning_ratio > 0.7:
            return {
                'level': 'high',
                'recommendation': 'Consider suggesting professional support or counseling'
            }
        elif concerning_ratio > 0.4:
            return {
                'level': 'moderate',
                'recommendation': 'Provide additional empathy and support resources'
            }
        else:
            return {
                'level': 'low',
                'recommendation': None
            }

    def get_insights(self, conversation_analytics: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable insights from analytics.

        Args:
            conversation_analytics: Results from analyze_conversation

        Returns:
            List of insight strings
        """
        insights = []

        overall = conversation_analytics.get('overall_sentiment', 'neutral')
        total = conversation_analytics.get('total_messages', 0)

        if total > 0:
            insights.append(f"Analyzed {total} user messages")
            insights.append(f"Overall sentiment: {overall.title()}")

        distribution = conversation_analytics.get('sentiment_distribution', {})
        if distribution:
            top_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append(
                f"Top emotions: {', '.join([f'{emotion} ({count})' for emotion, count in top_emotions])}"
            )

        support = conversation_analytics.get('needs_support', {})
        if support.get('level') == 'high':
            insights.append("⚠️ User may benefit from additional support")
        elif support.get('level') == 'moderate':
            insights.append("💛 User showing signs of stress - extra empathy recommended")

        return insights

    def save_model(self):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.sentiment_model,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """Load a trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.vectorizer = model_data['vectorizer']
            self.sentiment_model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
