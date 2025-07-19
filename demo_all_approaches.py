#!/usr/bin/env python3
"""
Demo script showing all non-LLM sentiment analysis approaches
Run this to compare different methods on the same text
"""

from sentiment_lexicon import LexiconSentimentAnalyzer
from sentiment_traditional_ml import TraditionalSentimentAnalyzer
from sentiment_rule_based import RuleBasedSentimentAnalyzer
from sentiment_ensemble import EnsembleSentimentAnalyzer

def demo_all_approaches():
    """Demo all sentiment analysis approaches"""
    
    print("🔍 Non-LLM Sentiment Analysis Demo")
    print("=" * 80)
    
    # Initialize all analyzers
    print("Initializing analyzers...")
    lexicon_analyzer = LexiconSentimentAnalyzer()
    
    ml_analyzer = TraditionalSentimentAnalyzer()
    print("Training ML analyzer...")
    ml_analyzer.train_on_sample_data()
    
    rule_analyzer = RuleBasedSentimentAnalyzer()
    ensemble_analyzer = EnsembleSentimentAnalyzer()
    
    print("✅ All analyzers ready!\n")
    
    # Test sentences
    test_sentences = [
        "I absolutely love this product! The quality is amazing and the service was excellent.",
        "This was the worst experience I've ever had. Terrible quality and awful customer service.",
        "The product is okay but the delivery was really slow and expensive.",
        "Great service but not worth the price. Would not recommend.",
        "Mixed feelings about this purchase. Some good points, some bad.",
        "I really love the service but the delivery was awful.",
    ]
    
    for i, text in enumerate(test_sentences, 1):
        print(f"\n{'='*20} TEST {i} {'='*20}")
        print(f"Text: '{text}'")
        print("\n" + "🏷️ LEXICON-BASED APPROACH:")
        print("-" * 50)
        
        # Lexicon approach
        try:
            lexicon_score, lexicon_class, lexicon_contrib = lexicon_analyzer.analyze_sentiment(text)
            print(f"Prediction: {lexicon_class}")
            print(f"Score: {lexicon_score:.3f}")
            
            # Show top contributing words
            significant_words = [c for c in lexicon_contrib if abs(c['final_score']) > 0.5]
            if significant_words:
                print("Key words:", ", ".join([f"{c['word']}({c['final_score']:.1f})" for c in significant_words[:5]]))
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "🤖 TRADITIONAL ML APPROACH:")
        print("-" * 50)
        
        # ML approach
        try:
            ml_pred, ml_conf, ml_features = ml_analyzer.predict_with_reasoning(text)
            ml_sentiment = "POSITIVE" if ml_pred == 1 else "NEGATIVE"
            print(f"Prediction: {ml_sentiment}")
            print(f"Confidence: {ml_conf:.3f}")
            
            # Show top features
            if ml_features:
                top_features = ml_features[:3]
                print("Top features:", ", ".join([f"'{f['feature']}'" for f in top_features]))
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "📋 RULE-BASED APPROACH:")
        print("-" * 50)
        
        # Rule-based approach
        try:
            rule_sentiment, rule_score, rule_matches = rule_analyzer.analyze_sentiment(text)
            print(f"Prediction: {rule_sentiment.value}")
            print(f"Score: {rule_score:.3f}")
            print(f"Rules triggered: {len(rule_matches)}")
            
            # Show strongest rules
            if rule_matches:
                strong_rules = [m for m in rule_matches if abs(m.rule.weight) > 1.0]
                print(f"Strong rules: {len(strong_rules)}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "🎯 ENSEMBLE APPROACH:")
        print("-" * 50)
        
        # Ensemble approach
        try:
            ensemble_pred = ensemble_analyzer.analyze_sentiment(text)
            print(f"Prediction: {ensemble_pred.final_sentiment}")
            print(f"Confidence: {ensemble_pred.confidence:.3f}")
            print(f"Consensus: {ensemble_pred.consensus_score:.3f}")
            
            # Show method agreement
            individual_preds = [pred['sentiment'] for pred in ensemble_pred.individual_predictions.values()]
            all_agree = len(set(individual_preds)) == 1
            print(f"Methods agree: {'Yes' if all_agree else 'No'}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "📊 SUMMARY:")
        print("-" * 50)
        try:
            # Create a summary table
            methods = [
                ("Lexicon", lexicon_class, lexicon_score),
                ("ML", "POSITIVE" if ml_pred == 1 else "NEGATIVE", ml_conf if ml_pred == 1 else -ml_conf),
                ("Rule", rule_sentiment.value, rule_score),
                ("Ensemble", ensemble_pred.final_sentiment, ensemble_pred.confidence)
            ]
            
            print(f"{'Method':<10} {'Sentiment':<10} {'Score':<10}")
            print("-" * 30)
            for method, sentiment, score in methods:
                print(f"{method:<10} {sentiment:<10} {score:<10.3f}")
        except Exception as e:
            print(f"Error creating summary: {e}")
    
    print("\n" + "="*80)
    print("🎉 Demo complete! Each approach offers different insights:")
    print("• Lexicon: Fast, interpretable, good for domain-specific vocabulary")
    print("• ML: Good performance, statistical confidence, learns from data")
    print("• Rule: Transparent, handles complex patterns, domain expertise")
    print("• Ensemble: Robust, comprehensive, combines all approaches")
    print("\nTry running individual analyzers for more detailed explanations!")

def interactive_demo():
    """Interactive demo where user can input their own text"""
    
    print("\n🎮 Interactive Demo")
    print("=" * 50)
    
    # Initialize ensemble analyzer (includes all methods)
    ensemble_analyzer = EnsembleSentimentAnalyzer()
    
    print("Enter text to analyze (or 'quit' to exit):")
    
    while True:
        text = input("\n> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not text:
            continue
        
        try:
            # Use ensemble for comprehensive analysis
            ensemble_analyzer.explain_prediction(text)
        except Exception as e:
            print(f"Error: {e}")
        
        print("\nEnter another text (or 'quit' to exit):")

if __name__ == "__main__":
    # Run the demo
    demo_all_approaches()
    
    # Ask if user wants interactive demo
    choice = input("\nWould you like to try the interactive demo? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        interactive_demo() 