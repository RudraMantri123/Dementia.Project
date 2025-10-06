"""Train and test the Analyst Agent."""

from src.agents.analyst_agent import AnalystAgent


def main():
    """Train and test the analyst agent."""
    print("="*80)
    print("ANALYST AGENT - SENTIMENT ANALYSIS TRAINING")
    print("="*80)

    print("\nInitializing Analyst Agent...")
    analyst = AnalystAgent()

    print("✓ Analyst Agent initialized and trained")
    print(f"✓ Model saved to: {analyst.model_path}")
    print(f"✓ Training status: {'Trained' if analyst.is_trained else 'Not trained'}")

    # Test sentiment analysis
    print("\n" + "="*80)
    print("TESTING SENTIMENT ANALYSIS")
    print("="*80)

    test_messages = [
        "I'm feeling so overwhelmed with taking care of my mother",
        "What are the early signs of dementia?",
        "I'm so grateful for this support",
        "I can't handle this anymore, I'm exhausted",
        "We had a really good day today",
        "I'm worried about what will happen next"
    ]

    print("\nAnalyzing individual messages:\n")

    for msg in test_messages:
        result = analyst.analyze_sentiment(msg)
        print(f"Message: \"{msg}\"")
        print(f"  Sentiment: {result['sentiment'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print()

    # Test conversation analysis
    print("="*80)
    print("TESTING CONVERSATION ANALYSIS")
    print("="*80)

    conversation = [
        {'role': 'user', 'content': "I'm feeling really stressed about caregiving"},
        {'role': 'assistant', 'content': "I understand..."},
        {'role': 'user', 'content': "I can't sleep at night worrying"},
        {'role': 'assistant', 'content': "That must be hard..."},
        {'role': 'user', 'content': "But today we had a nice moment together"},
        {'role': 'assistant', 'content': "That's wonderful..."},
        {'role': 'user', 'content': "Still, I feel overwhelmed most of the time"}
    ]

    print("\nAnalyzing full conversation...\n")
    analytics = analyst.analyze_conversation(conversation)

    print(f"Overall Sentiment: {analytics['overall_sentiment'].upper()}")
    print(f"Total User Messages: {analytics['total_messages']}")
    print(f"\nSentiment Distribution:")
    for sentiment, count in analytics.get('sentiment_distribution', {}).items():
        print(f"  - {sentiment}: {count}")

    print(f"\nSentiment Trend: {' → '.join(analytics['sentiment_trend'])}")

    support_needs = analytics.get('needs_support', {})
    print(f"\nSupport Assessment:")
    print(f"  Level: {support_needs.get('level', 'unknown').upper()}")
    if support_needs.get('recommendation'):
        print(f"  Recommendation: {support_needs['recommendation']}")

    # Get insights
    print("\n" + "="*80)
    print("CONVERSATION INSIGHTS")
    print("="*80 + "\n")

    insights = analyst.get_insights(analytics)
    for insight in insights:
        print(f"  • {insight}")

    print("\n" + "="*80)
    print("✓ Analyst Agent training and testing completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
