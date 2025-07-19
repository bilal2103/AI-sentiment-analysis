import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict

# Import our custom analyzers
from sentiment_lexicon import LexiconSentimentAnalyzer
from sentiment_traditional_ml import TraditionalSentimentAnalyzer
from sentiment_rule_based import RuleBasedSentimentAnalyzer, SentimentLabel

@dataclass
class EnsemblePrediction:
    """Represents a prediction from the ensemble"""
    final_sentiment: str
    confidence: float
    individual_predictions: Dict[str, Dict]
    reasoning: str
    consensus_score: float

class EnsembleSentimentAnalyzer:
    """
    Ensemble sentiment analyzer that combines multiple approaches.
    Provides comprehensive reasoning by aggregating different methodologies.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        # Initialize individual analyzers
        self.lexicon_analyzer = LexiconSentimentAnalyzer()
        self.ml_analyzer = TraditionalSentimentAnalyzer()
        self.rule_analyzer = RuleBasedSentimentAnalyzer()
        
        # Train the ML analyzer
        self.ml_analyzer.train_on_sample_data()
        
        # Default weights for ensemble voting
        self.weights = weights or {
            'lexicon': 0.3,
            'ml': 0.4,
            'rule': 0.3
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.values()}
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize different scoring systems to [-1, 1] range"""
        normalized = {}
        
        # Lexicon scores are already in reasonable range
        normalized['lexicon'] = max(-1.0, min(1.0, scores['lexicon'] / 5.0))
        
        # ML scores (logistic regression outputs) - convert to [-1, 1]
        ml_score = scores['ml']
        normalized['ml'] = (ml_score - 0.5) * 2  # Convert [0,1] to [-1,1]
        
        # Rule scores - similar to lexicon
        normalized['rule'] = max(-1.0, min(1.0, scores['rule'] / 5.0))
        
        return normalized
    
    def calculate_consensus(self, predictions: Dict[str, str]) -> float:
        """Calculate consensus among different methods"""
        sentiment_counts = defaultdict(int)
        for pred in predictions.values():
            sentiment_counts[pred] += 1
        
        max_count = max(sentiment_counts.values())
        total_predictions = len(predictions)
        
        return max_count / total_predictions
    
    def analyze_sentiment(self, text: str) -> EnsemblePrediction:
        """Analyze sentiment using ensemble of methods"""
        
        # Get predictions from all methods
        individual_predictions = {}
        
        # 1. Lexicon-based analysis
        lexicon_score, lexicon_class, lexicon_contributions = self.lexicon_analyzer.analyze_sentiment(text)
        individual_predictions['lexicon'] = {
            'sentiment': lexicon_class,
            'score': lexicon_score,
            'reasoning': lexicon_contributions
        }
        
        # 2. Traditional ML analysis
        ml_pred, ml_conf, ml_features = self.ml_analyzer.predict_with_reasoning(text)
        ml_sentiment = "POSITIVE" if ml_pred == 1 else "NEGATIVE"
        individual_predictions['ml'] = {
            'sentiment': ml_sentiment,
            'score': ml_conf if ml_pred == 1 else -ml_conf,
            'reasoning': ml_features
        }
        
        # 3. Rule-based analysis
        rule_sentiment, rule_score, rule_matches = self.rule_analyzer.analyze_sentiment(text)
        individual_predictions['rule'] = {
            'sentiment': rule_sentiment.value,
            'score': rule_score,
            'reasoning': rule_matches
        }
        
        # Calculate weighted ensemble score
        scores = {
            'lexicon': individual_predictions['lexicon']['score'],
            'ml': individual_predictions['ml']['score'],
            'rule': individual_predictions['rule']['score']
        }
        
        normalized_scores = self.normalize_scores(scores)
        
        weighted_score = sum(
            self.weights[method] * normalized_scores[method] 
            for method in self.weights.keys()
        )
        
        # Determine final sentiment
        if weighted_score > 0.1:
            final_sentiment = "POSITIVE"
        elif weighted_score < -0.1:
            final_sentiment = "NEGATIVE"
        else:
            final_sentiment = "NEUTRAL"
        
        # Calculate confidence based on agreement and score magnitude
        predictions_only = {k: v['sentiment'] for k, v in individual_predictions.items()}
        consensus_score = self.calculate_consensus(predictions_only)
        confidence = min(0.95, abs(weighted_score) * consensus_score)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(individual_predictions, weighted_score, consensus_score)
        
        return EnsemblePrediction(
            final_sentiment=final_sentiment,
            confidence=confidence,
            individual_predictions=individual_predictions,
            reasoning=reasoning,
            consensus_score=consensus_score
        )
    
    def generate_reasoning(self, predictions: Dict, weighted_score: float, consensus: float) -> str:
        """Generate human-readable reasoning for the ensemble decision"""
        reasoning_parts = []
        
        # Analyze individual method agreements
        sentiments = [pred['sentiment'] for pred in predictions.values()]
        
        if len(set(sentiments)) == 1:
            reasoning_parts.append(f"All methods agree: {sentiments[0]}")
        else:
            sentiment_counts = defaultdict(int)
            for sent in sentiments:
                sentiment_counts[sent] += 1
            
            majority = max(sentiment_counts.items(), key=lambda x: x[1])
            reasoning_parts.append(f"Majority consensus: {majority[0]} ({majority[1]}/3 methods)")
        
        # Analyze score contributions
        reasoning_parts.append(f"Weighted ensemble score: {weighted_score:.3f}")
        reasoning_parts.append(f"Consensus strength: {consensus:.2f}")
        
        # Highlight key insights from each method
        method_insights = []
        
        # Lexicon insights
        lexicon_reasoning = predictions['lexicon']['reasoning']
        positive_words = [c for c in lexicon_reasoning if c['final_score'] > 0]
        negative_words = [c for c in lexicon_reasoning if c['final_score'] < 0]
        
        if positive_words:
            method_insights.append(f"Lexicon found {len(positive_words)} positive indicators")
        if negative_words:
            method_insights.append(f"Lexicon found {len(negative_words)} negative indicators")
        
        # ML insights
        ml_features = predictions['ml']['reasoning'][:3]  # Top 3 features
        if ml_features:
            top_feature = ml_features[0]
            method_insights.append(f"ML top feature: '{top_feature['feature']}' ({top_feature['sentiment_direction']})")
        
        # Rule insights
        rule_matches = predictions['rule']['reasoning']
        if rule_matches:
            strongest_rule = max(rule_matches, key=lambda x: abs(x.rule.weight))
            method_insights.append(f"Strongest rule: {strongest_rule.rule.description}")
        
        if method_insights:
            reasoning_parts.append("Key insights: " + "; ".join(method_insights))
        
        return " | ".join(reasoning_parts)
    
    def explain_prediction(self, text: str) -> None:
        """Provide detailed explanation of the ensemble prediction"""
        prediction = self.analyze_sentiment(text)
        
        print(f"Text: '{text}'")
        print(f"Final Prediction: {prediction.final_sentiment}")
        print(f"Confidence: {prediction.confidence:.3f}")
        print(f"Consensus Score: {prediction.consensus_score:.3f}")
        print("\n" + "="*80)
        
        # Show individual method results
        print("INDIVIDUAL METHOD RESULTS:")
        print("-" * 40)
        
        for method, result in prediction.individual_predictions.items():
            print(f"\n{method.upper()} METHOD:")
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Score: {result['score']:.3f}")
            
            # Show method-specific reasoning
            if method == 'lexicon':
                contributions = result['reasoning']
                significant = [c for c in contributions if abs(c['final_score']) > 0.5]
                if significant:
                    print(f"  Key words: {', '.join([c['word'] for c in significant[:5]])}")
            
            elif method == 'ml':
                features = result['reasoning'][:3]
                if features:
                    print(f"  Top features: {', '.join(f['feature'] for f in features)}")
            
            elif method == 'rule':
                matches = result['reasoning']
                if matches:
                    print(f"  Rules triggered: {len(matches)}")
                    strong_rules = [m for m in matches if abs(m.rule.weight) > 1.0]
                    if strong_rules:
                        print(f"  Strong rules: {len(strong_rules)}")
        
        print("\n" + "="*80)
        print("ENSEMBLE REASONING:")
        print(prediction.reasoning)
        
        # Show method weights
        print(f"\nMethod weights: {self.weights}")
    
    def compare_methods(self, text: str) -> None:
        """Compare all methods side by side"""
        prediction = self.analyze_sentiment(text)
        
        print(f"Text: '{text}'")
        print("\nMETHOD COMPARISON:")
        print("-" * 60)
        print(f"{'Method':<12} {'Sentiment':<10} {'Score':<10} {'Reasoning'}")
        print("-" * 60)
        
        for method, result in prediction.individual_predictions.items():
            reasoning_summary = ""
            if method == 'lexicon':
                contrib = result['reasoning']
                pos_words = len([c for c in contrib if c['final_score'] > 0])
                neg_words = len([c for c in contrib if c['final_score'] < 0])
                reasoning_summary = f"pos:{pos_words}, neg:{neg_words}"
            
            elif method == 'ml':
                features = result['reasoning'][:2]
                if features:
                    reasoning_summary = f"top: {features[0]['feature'][:15]}"
            
            elif method == 'rule':
                matches = result['reasoning']
                reasoning_summary = f"{len(matches)} rules"
            
            print(f"{method:<12} {result['sentiment']:<10} {result['score']:<10.3f} {reasoning_summary}")
        
        print("-" * 60)
        print(f"{'ENSEMBLE':<12} {prediction.final_sentiment:<10} {prediction.confidence:<10.3f} consensus:{prediction.consensus_score:.2f}")


# Example usage
if __name__ == "__main__":
    # Create ensemble analyzer
    analyzer = EnsembleSentimentAnalyzer()
    
    print("Ensemble Sentiment Analyzer")
    print("Combines: Lexicon-based, Traditional ML, and Rule-based approaches")
    print("=" * 80)
    
    # Test sentences
    test_sentences = [
        "I really love the service but the delivery was awful.",
        "The food was not good at all.",
        "Extremely disappointed with the poor quality.",
        "Very happy with the excellent service!",
        "The product was quite nice but expensive.",
        "This is the worst experience I've ever had!",
        "Would definitely recommend this to everyone.",
        "The quality is better than expected."
    ]
    
    for sentence in test_sentences:
        analyzer.explain_prediction(sentence)
        print("\n" + "="*80 + "\n")
    
    # Show a detailed comparison for one sentence
    print("DETAILED METHOD COMPARISON:")
    print("="*80)
    analyzer.compare_methods("I really love the service but the delivery was awful.") 