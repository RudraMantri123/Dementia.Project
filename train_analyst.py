"""
Improved training pipeline for sentiment analysis with higher accuracy.

This script implements:
1. Data augmentation and balancing
2. Advanced feature engineering
3. Multiple model architectures
4. Hyperparameter tuning
5. Cross-validation
6. Ensemble methods

Target: Increase accuracy from 78% to 85%+
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==================== DATA AUGMENTATION ====================

def synonym_replacement(text, n=2):
    """Replace n words with synonyms."""
    emotion_synonyms = {
        'stressed': ['overwhelmed', 'anxious', 'pressured', 'tense', 'strained'],
        'sad': ['depressed', 'unhappy', 'sorrowful', 'heartbroken', 'melancholy'],
        'anxious': ['worried', 'nervous', 'concerned', 'uneasy', 'apprehensive'],
        'frustrated': ['annoyed', 'irritated', 'exasperated', 'aggravated'],
        'positive': ['happy', 'grateful', 'thankful', 'pleased', 'satisfied'],
        'tired': ['exhausted', 'drained', 'fatigued', 'weary'],
        'difficult': ['hard', 'challenging', 'tough', 'demanding'],
    }

    words = text.split()
    replaced = 0

    for i, word in enumerate(words):
        if replaced >= n:
            break
        word_lower = word.lower().strip('.,!?')
        for key, synonyms in emotion_synonyms.items():
            if word_lower == key or word_lower in synonyms:
                replacement = random.choice([s for s in synonyms if s != word_lower])
                words[i] = replacement
                replaced += 1
                break

    return ' '.join(words)


def random_insertion(text, label):
    """Insert words that reinforce sentiment."""
    insertions = {
        'stressed': ['really', 'very', 'extremely', 'so', 'completely'],
        'sad': ['really', 'very', 'deeply', 'so', 'truly'],
        'anxious': ['really', 'very', 'quite', 'so', 'extremely'],
        'frustrated': ['really', 'very', 'so', 'incredibly', 'extremely'],
        'positive': ['really', 'very', 'so', 'quite', 'truly'],
        'neutral': []
    }

    if label not in insertions or not insertions[label]:
        return text

    words = text.split()
    insert_word = random.choice(insertions[label])

    emotion_words = {
        'stressed': ['overwhelmed', 'stressed', 'anxious', 'pressured'],
        'sad': ['sad', 'depressed', 'unhappy', 'heartbroken'],
        'anxious': ['worried', 'anxious', 'nervous', 'concerned'],
        'frustrated': ['frustrated', 'annoyed', 'irritated', 'angry'],
        'positive': ['happy', 'grateful', 'positive', 'glad']
    }

    target_words = emotion_words.get(label, [])
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?')
        if word_lower in target_words:
            if i > 0 and words[i-1].lower() not in insertions[label]:
                words.insert(i, insert_word)
                break

    return ' '.join(words)


def paraphrase_text(text, label):
    """Generate paraphrase maintaining sentiment."""
    templates = {
        'stressed': [
            "I'm feeling really overwhelmed about {topic}",
            "The {topic} is making me feel extremely stressed",
            "I can't handle {topic}, I'm so overwhelmed",
            "{topic} is crushing me with stress",
            "I'm completely stressed by {topic}",
        ],
        'sad': [
            "I feel so sad about {topic}",
            "It makes me deeply sad thinking about {topic}",
            "I'm heartbroken because of {topic}",
            "{topic} fills me with sadness",
            "I'm mourning {topic}",
        ],
        'anxious': [
            "I'm really anxious about {topic}",
            "I feel worried when I think about {topic}",
            "{topic} makes me feel really nervous",
            "I'm terrified about {topic}",
            "I can't stop worrying about {topic}",
        ],
        'frustrated': [
            "I'm getting frustrated with {topic}",
            "{topic} is so irritating",
            "I feel angry because of {topic}",
            "{topic} is driving me crazy",
            "I'm fed up with {topic}",
        ],
        'positive': [
            "I'm feeling grateful about {topic}",
            "{topic} makes me feel hopeful",
            "I'm so thankful for {topic}",
            "I feel blessed by {topic}",
            "{topic} brings me joy",
        ],
    }

    topics = ['caregiving', 'this situation', 'everything', 'all of this', 'the challenges', 'their care']

    if label in templates:
        template = random.choice(templates[label])
        topic = random.choice(topics)
        return template.format(topic=topic)

    return synonym_replacement(text, n=1)


def augment_dataset(texts, labels, target_multiplier=2.0):
    """Augment dataset to expand training data with advanced techniques."""
    augmented_texts = list(texts)
    augmented_labels = list(labels)

    current_size = len(texts)
    target_size = int(current_size * target_multiplier)
    augmentations_needed = target_size - current_size

    print(f"Augmenting dataset: {current_size} → {target_size} samples")

    for i in range(augmentations_needed):
        idx = i % current_size
        original_text = texts[idx]
        label = labels[idx]

        # Use multiple augmentation techniques
        technique = random.choice(['synonym_replacement', 'insertion', 'paraphrase'])

        if technique == 'synonym_replacement':
            aug_text = synonym_replacement(original_text, n=random.randint(1, 3))
        elif technique == 'insertion':
            aug_text = random_insertion(original_text, label)
        else:  # paraphrase
            aug_text = paraphrase_text(original_text, label)

        augmented_texts.append(aug_text)
        augmented_labels.append(label)

    print(f"✓ Generated {augmentations_needed} augmented samples")
    return augmented_texts, augmented_labels


def balance_classes(texts, labels):
    """Balance class distribution through oversampling minority classes."""
    from collections import Counter

    class_counts = Counter(labels)
    print(f"\nOriginal class distribution:")
    for label, count in class_counts.most_common():
        print(f"  {label}: {count}")

    target_count = max(class_counts.values())

    # Group samples by class
    class_samples = {}
    for text, label in zip(texts, labels):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(text)

    # Balance by oversampling
    balanced_texts = []
    balanced_labels = []

    for label, samples in class_samples.items():
        balanced_texts.extend(samples)
        balanced_labels.extend([label] * len(samples))

        needed = target_count - len(samples)
        if needed > 0:
            for i in range(needed):
                original = random.choice(samples)
                augmented = synonym_replacement(original, n=2)
                balanced_texts.append(augmented)
                balanced_labels.append(label)

    # Shuffle
    combined = list(zip(balanced_texts, balanced_labels))
    random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)

    print(f"\nBalanced class distribution:")
    class_counts = Counter(balanced_labels)
    for label, count in class_counts.most_common():
        print(f"  {label}: {count}")

    return list(balanced_texts), list(balanced_labels)


# ==================== ADDITIONAL TRAINING DATA ====================

def get_additional_training_data():
    """Generate additional training examples for underperforming classes."""
    return [
        # More STRESSED examples (F1: 0.737, Recall: 0.583 - PRIORITY)
        ("I have too many responsibilities and I'm drowning", "stressed"),
        ("The pressure of caregiving is building constantly", "stressed"),
        ("I'm stretched so thin I might snap any moment", "stressed"),
        ("The stress is affecting my physical health badly", "stressed"),
        ("I'm running on fumes with absolutely nothing left", "stressed"),
        ("The workload is completely overwhelming me daily", "stressed"),
        ("I can't handle one more thing on my plate", "stressed"),
        ("I'm buckling under the weight of it all", "stressed"),
        ("Stress has completely taken over my entire life", "stressed"),
        ("I'm at capacity and way beyond my limits", "stressed"),
        ("I feel like I'm going to collapse from stress", "stressed"),
        ("The constant stress is making me physically ill", "stressed"),
        ("I'm so stressed I can barely think straight", "stressed"),
        ("Managing everything is crushing my spirit completely", "stressed"),
        ("I'm overwhelmed by all the medical appointments", "stressed"),
        ("The caregiving duties are piling up endlessly", "stressed"),
        ("I'm stressed about money and care and everything", "stressed"),
        ("I haven't had a moment to breathe in weeks", "stressed"),
        ("The responsibility is suffocating me slowly", "stressed"),
        ("I'm stressed beyond what I thought possible", "stressed"),
        ("Every single day the stress gets worse", "stressed"),
        ("I'm completely maxed out and can't do more", "stressed"),
        ("The weight of caregiving is breaking me down", "stressed"),
        ("I'm so stressed my body is shutting down", "stressed"),
        ("Managing their care alone is impossibly stressful", "stressed"),
        ("I'm overwhelmed by medications and schedules and care", "stressed"),
        ("The stress of watching them decline is unbearable", "stressed"),
        ("I'm stressed about making medical decisions alone", "stressed"),
        ("Coordinating all their care is overwhelming me", "stressed"),
        ("I'm stressed trying to balance work and caregiving", "stressed"),
        ("The financial stress on top of everything else", "stressed"),
        ("I'm overwhelmed by insurance and paperwork and bills", "stressed"),
        ("The stress is affecting my sleep and health", "stressed"),
        ("I'm burned out and stressed to my breaking point", "stressed"),
        ("Managing behavioral issues is incredibly stressful", "stressed"),
        ("I'm stressed about their safety every single second", "stressed"),
        ("The constant vigilance is stressing me out badly", "stressed"),
        ("I'm overwhelmed trying to keep up with everything", "stressed"),
        ("The stress of anticipating what comes next", "stressed"),
        ("I'm stretched too thin across too many things", "stressed"),

        # More FRUSTRATED examples (F1: 0.828 - needs improvement)
        ("Why won't they listen to my concerns at all", "frustrated"),
        ("This is ridiculous and unfair to everyone involved", "frustrated"),
        ("I'm sick and tired of dealing with this daily", "frustrated"),
        ("They never take my suggestions seriously ever", "frustrated"),
        ("Everything I do seems pointless and ineffective", "frustrated"),
        ("Why does this keep happening over and over", "frustrated"),
        ("I'm fed up with the complete lack of help", "frustrated"),
        ("Nothing ever changes no matter what I do", "frustrated"),
        ("They ignore all my requests for support always", "frustrated"),
        ("This situation is impossible and absolutely maddening", "frustrated"),
        ("I'm angry that nobody understands the struggle", "frustrated"),
        ("The system fails us at every single turn", "frustrated"),
        ("Why can't anyone see how hard this is", "frustrated"),
        ("I'm irritated by their complete lack of empathy", "frustrated"),
        ("They keep making promises they never ever keep", "frustrated"),
        ("I'm annoyed by how difficult they make everything", "frustrated"),
        ("This bureaucracy is driving me absolutely crazy", "frustrated"),
        ("I'm exasperated with all these pointless obstacles", "frustrated"),
        ("They don't appreciate anything I sacrifice daily", "frustrated"),
        ("I'm aggravated by their stubborn refusal to help", "frustrated"),
        ("Why won't the doctors return my calls", "frustrated"),
        ("I'm frustrated with how slow everything moves", "frustrated"),
        ("The insurance company denied us again", "frustrated"),
        ("I'm angry about the lack of affordable care", "frustrated"),
        ("Why is getting help so impossibly difficult", "frustrated"),
        ("I'm frustrated they refuse to bathe or eat", "frustrated"),
        ("They're being so stubborn and difficult today", "frustrated"),
        ("I'm angry at this whole impossible situation", "frustrated"),
        ("Why won't family members step up and help", "frustrated"),
        ("I'm frustrated by everyone's empty promises", "frustrated"),
        ("The care system is broken and useless", "frustrated"),
        ("I'm angry they waited so long to diagnose", "frustrated"),
        ("Why is finding good care so frustrating", "frustrated"),
        ("I'm irritated by all the red tape and bureaucracy", "frustrated"),
        ("They keep asking me the same question repeatedly", "frustrated"),
        ("I'm frustrated explaining the same things over", "frustrated"),
        ("Why won't they accept they need help now", "frustrated"),
        ("I'm angry at how this disease progresses", "frustrated"),
        ("The lack of effective treatments is frustrating", "frustrated"),
        ("I'm annoyed by people who don't understand", "frustrated"),

        # Additional SAD examples to maintain balance
        ("My heart is completely broken watching them decline", "sad"),
        ("I'm mourning the terrible loss of who they were", "sad"),
        ("Grief follows me everywhere I go now daily", "sad"),
        ("I cry myself to sleep every single night now", "sad"),
        ("This profound sadness never ever leaves me alone", "sad"),
        ("I'm drowning in endless sorrow and loss", "sad"),
        ("The emptiness inside me is completely overwhelming", "sad"),
        ("I miss them terribly even though they're still here", "sad"),
        ("This disease has cruelly stolen my loved one", "sad"),
        ("I'm heartbroken by what's been taken away forever", "sad"),

        # Additional ANXIOUS examples to maintain balance
        ("I'm terrified something bad will happen to them", "anxious"),
        ("I can't stop worrying about absolutely everything", "anxious"),
        ("My heart races constantly thinking about tomorrow", "anxious"),
        ("I'm consumed by fear and terrible uncertainty", "anxious"),
        ("I feel panic rising inside me all the time", "anxious"),
        ("I'm afraid I won't be able to handle it", "anxious"),
        ("The unknown future keeps me up every night", "anxious"),
        ("I'm nervous about making the wrong choice here", "anxious"),
        ("I feel uneasy about their changing behavior patterns", "anxious"),
        ("I'm apprehensive about what might happen next stage", "anxious"),

        # Additional POSITIVE examples to maintain balance
        ("Today was actually a really wonderful good day", "positive"),
        ("I'm feeling encouraged by the progress we made", "positive"),
        ("Things are looking up and I'm feeling hopeful", "positive"),
        ("I'm so grateful for all the support I receive", "positive"),
        ("This new strategy is really helping us succeed", "positive"),
        ("I feel blessed to have this amazing community", "positive"),
        ("We had such a lovely moment together today", "positive"),
        ("I'm proud of how I'm handling this challenge", "positive"),
        ("The sunshine today lifted my spirits greatly", "positive"),
        ("I'm feeling optimistic about our future together", "positive"),

        # Additional NEUTRAL examples to maintain balance
        ("Can you explain the stages of dementia progression", "neutral"),
        ("What medications are typically prescribed for this condition", "neutral"),
        ("I need details about memory care facilities available", "neutral"),
        ("How do I apply for long-term care insurance", "neutral"),
        ("What are the requirements for disability benefits", "neutral"),
        ("Tell me about respite care services available", "neutral"),
        ("What safety measures should I implement at home", "neutral"),
        ("How do I find a qualified dementia specialist", "neutral"),
        ("What are the different types of dementia", "neutral"),
        ("Can you describe cognitive therapy options available", "neutral"),
    ]


# ==================== TRAINING PIPELINE ====================

def load_base_training_data():
    """Load base training data from analyst agent."""
    from src.agents.analyst_agent import AnalystAgent

    # Initialize agent to get training data
    agent = AnalystAgent()

    # Extract training data from the default model training
    training_data = []

    # Get from agent's training method - we'll reload the comprehensive data
    # Reload comprehensive training data (matches analyst_agent.py)
    training_data = [
        # STRESSED examples
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

        # SAD examples
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

        # ANXIOUS examples
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

        # FRUSTRATED examples
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

        # POSITIVE examples
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

        # NEUTRAL examples
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
    ]

    return training_data


def create_advanced_features(texts, max_features=2000):
    """Create advanced features combining TF-IDF with optimized parameters."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 4),  # unigrams, bigrams, trigrams, 4-grams
        stop_words='english',
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        norm='l2',
        smooth_idf=True
    )

    X_tfidf = tfidf.fit_transform(texts)
    return X_tfidf, tfidf


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)

    param_grid = {
        'C': [0.5, 1.0, 2.0, 5.0],
        'max_iter': [3000, 5000],
        'solver': ['saga', 'lbfgs'],
        'class_weight': ['balanced']
    }

    model = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )

    print("Running grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)

    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_ensemble_model(X_train, y_train, X_test, y_test):
    """Train ensemble model with multiple classifiers."""
    print("\n" + "="*70)
    print("TRAINING ENSEMBLE MODELS")
    print("="*70)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight='balanced',
            solver='saga',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'model': model, 'f1_score': f1}
        print(f"✓ {name} F1 Score: {f1:.4f}")

    print(f"\nTraining Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', models['Logistic Regression']),
            ('rf', models['Random Forest']),
            ('gb', models['Gradient Boosting']),
        ],
        voting='soft',
        n_jobs=-1
    )

    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')

    results['Ensemble'] = {'model': ensemble, 'f1_score': f1_ensemble}
    print(f"✓ Ensemble F1 Score: {f1_ensemble:.4f}")

    best_name = max(results, key=lambda k: results[k]['f1_score'])
    best_model = results[best_name]['model']
    best_score = results[best_name]['f1_score']

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name} (F1: {best_score:.4f})")
    print(f"{'='*70}")

    return best_model, best_name, best_score, results


def main():
    """Main training pipeline with improvements."""
    print("="*70)
    print("IMPROVED SENTIMENT ANALYSIS TRAINING PIPELINE")
    print("="*70)

    # Step 1: Load base data
    print("\n1. Loading base training data...")
    base_data = load_base_training_data()
    texts, labels = zip(*base_data)
    print(f"✓ Loaded {len(texts)} base samples")

    # Step 2: Add additional training data
    print("\n2. Adding additional training data...")
    additional_data = get_additional_training_data()
    add_texts, add_labels = zip(*additional_data)
    texts = list(texts) + list(add_texts)
    labels = list(labels) + list(add_labels)
    print(f"✓ Total samples: {len(texts)}")

    # Step 3: Data augmentation (increased to 2x)
    print("\n3. Augmenting data...")
    texts, labels = augment_dataset(texts, labels, target_multiplier=2.0)

    # Step 4: Balance classes
    print("\n4. Balancing classes...")
    texts, labels = balance_classes(texts, labels)

    # Step 5: Create advanced features (increased to 2000 features)
    print("\n5. Creating advanced features...")
    X, vectorizer = create_advanced_features(texts, max_features=2000)
    print(f"✓ Feature matrix shape: {X.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Step 6: Split data (80/20)
    print("\n6. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Test samples: {X_test.shape[0]}")

    # Step 7: Hyperparameter tuning
    print("\n7. Performing hyperparameter tuning...")
    tuned_model = hyperparameter_tuning(X_train, y_train)

    # Step 8: Train ensemble models
    best_model, best_name, best_score, all_results = train_ensemble_model(
        X_train, y_train, X_test, y_test
    )

    # Compare with tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')
    print(f"\n✓ Hyperparameter-tuned Logistic Regression F1: {f1_tuned:.4f}")

    # Use the best model overall
    if f1_tuned > best_score:
        best_model = tuned_model
        best_name = "Tuned Logistic Regression"
        best_score = f1_tuned
        print(f"✓ Using tuned model as best (F1: {f1_tuned:.4f})")

    # Step 9: Cross-validation
    print("\n" + "="*70)
    print("CROSS-VALIDATION")
    print("="*70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        best_model, X_train, y_train,
        cv=cv, scoring='f1_weighted', n_jobs=-1
    )

    print(f"\n5-Fold Cross-Validation F1 Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"\nMean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Step 10: Detailed evaluation
    print("\n" + "="*70)
    print("DETAILED EVALUATION ON TEST SET")
    print("="*70)

    y_pred = best_model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels, digits=3))

    # Step 11: Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    model_data = {
        'vectorizer': vectorizer,
        'model': best_model,
        'label_encoder': label_encoder,
        'model_name': best_name,
        'f1_score': best_score,
        'classes': label_encoder.classes_
    }

    os.makedirs('data/models', exist_ok=True)
    model_path = 'data/models/analyst_model.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Model type: {best_name}")
    print(f"✓ F1 Score: {best_score:.4f}")

    # Step 12: Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n✓ Total training samples: {len(texts)}")
    print(f"✓ Feature dimensions: {X.shape[1]}")
    print(f"✓ Best model: {best_name}")
    print(f"✓ Test F1 Score: {best_score:.4f}")
    print(f"✓ Cross-validation F1: {cv_scores.mean():.4f}")

    print(f"\nAll Model F1 Scores:")
    all_results['Tuned Logistic Regression'] = {'f1_score': f1_tuned}
    for name, result in sorted(all_results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
        print(f"  {name}: {result['f1_score']:.4f}")

    baseline_f1 = 0.78
    improvement = (best_score - baseline_f1) / baseline_f1 * 100

    print(f"\n{'='*70}")
    if best_score > baseline_f1:
        print(f"✓ IMPROVEMENT: +{improvement:.1f}% over baseline ({baseline_f1:.2f} → {best_score:.2f})")
        if best_score >= 0.95:
            print(f"✓✓ EXCELLENT: Achieved 95%+ F1 Score!")
        elif best_score >= 0.90:
            print(f"✓ GREAT: Achieved 90%+ F1 Score!")
    else:
        print(f"⚠ Model performance similar to baseline")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
