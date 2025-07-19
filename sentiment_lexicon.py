from collections import defaultdict
import re
import numpy as np
from typing import Dict, List, Tuple

class LexiconSentimentAnalyzer:
    """
    Lexicon-based sentiment analyzer with word-level reasoning.
    Uses VADER-style lexicon with intensifiers and negation handling.
    """
    
    def __init__(self):
        # Basic sentiment lexicon (simplified version)
        self.lexicon = {
            # Positive words
            'love': 2.5, 'amazing': 2.0, 'excellent': 2.5, 'great': 2.0,
            'good': 1.5, 'wonderful': 2.0, 'fantastic': 2.5, 'awesome': 2.0,
            'perfect': 2.5, 'beautiful': 2.0, 'nice': 1.5, 'happy': 2.0,
            'pleased': 1.5, 'satisfied': 1.5, 'delighted': 2.0,
            
            # Negative words
            'hate': -2.5, 'terrible': -2.5, 'awful': -2.5, 'horrible': -2.5,
            'bad': -1.5, 'worst': -2.5, 'disgusting': -2.0, 'annoying': -1.5,
            'disappointed': -2.0, 'frustrated': -2.0, 'angry': -2.0,
            'sad': -1.5, 'upset': -1.5, 'unhappy': -2.0, 'poor': -1.5,
            
            # Neutral/context words
            'service': 0.0, 'delivery': 0.0, 'product': 0.0, 'experience': 0.0,
            'staff': 0.0, 'quality': 0.0, 'price': 0.0, 'food': 0.0
        }
        
        # Intensifiers that amplify sentiment
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
            'totally': 1.5, 'completely': 1.8, 'really': 1.3, 'quite': 1.2,
            'rather': 1.1, 'somewhat': 0.8, 'slightly': 0.6, 'a bit': 0.7
        }
        
        # Negation words
        self.negations = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither',
            'nobody', 'nor', 'cannot', 'can\'t', 'won\'t', 'wouldn\'t',
            'shouldn\'t', 'couldn\'t', 'doesn\'t', 'don\'t', 'didn\'t',
            'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t'
        }
        
        # Contrasting conjunctions that can flip sentiment
        self.contrasts = {'but', 'however', 'although', 'though', 'yet', 'nevertheless'}
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text"""
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Simple tokenization (split by space and punctuation)
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str, List[Dict]]:
        """
        Analyze sentiment with detailed reasoning.
        Returns: (score, classification, word_contributions)
        """
        tokens = self.preprocess_text(text)
        word_contributions = []
        total_score = 0.0
        
        negation_active = False
        intensifier_multiplier = 1.0
        
        for i, token in enumerate(tokens):
            contribution = {
                'word': token,
                'base_score': 0.0,
                'final_score': 0.0,
                'modifiers': []
            }
            
            # Check for negation
            if token in self.negations:
                negation_active = True
                contribution['modifiers'].append('negation_trigger')
                word_contributions.append(contribution)
                continue
            
            # Check for intensifiers
            if token in self.intensifiers:
                intensifier_multiplier = self.intensifiers[token]
                contribution['modifiers'].append(f'intensifier_{intensifier_multiplier}x')
                word_contributions.append(contribution)
                continue
            
            # Check for contrasting conjunctions
            if token in self.contrasts:
                # Reset negation and reduce previous sentiment by half
                negation_active = False
                total_score *= 0.5
                contribution['modifiers'].append('contrast_reduces_previous')
                word_contributions.append(contribution)
                continue
            
            # Get sentiment score from lexicon
            if token in self.lexicon:
                base_score = self.lexicon[token]
                contribution['base_score'] = base_score
                
                # Apply intensifier
                final_score = base_score * intensifier_multiplier
                
                # Apply negation
                if negation_active:
                    final_score *= -0.8  # Negation reduces and flips sentiment
                    contribution['modifiers'].append('negated')
                
                contribution['final_score'] = final_score
                total_score += final_score
                
                # Reset modifiers after applying
                if negation_active:
                    negation_active = False
                intensifier_multiplier = 1.0
            
            word_contributions.append(contribution)
        
        # Classify sentiment
        if total_score >= 0.5:
            classification = "POSITIVE"
        elif total_score <= -0.5:
            classification = "NEGATIVE"
        else:
            classification = "NEUTRAL"
        
        return total_score, classification, word_contributions
    
    def explain_prediction(self, text: str) -> None:
        """Print detailed explanation of sentiment analysis"""
        score, classification, contributions = self.analyze_sentiment(text)
        
        print(f"Text: '{text}'")
        print(f"Overall Score: {score:.3f}")
        print(f"Classification: {classification}")
        print("\nWord-by-word analysis:")
        print("-" * 50)
        
        for contrib in contributions:
            if contrib['final_score'] != 0 or contrib['modifiers']:
                modifiers_str = ', '.join(contrib['modifiers']) if contrib['modifiers'] else 'none'
                print(f"'{contrib['word']}': base={contrib['base_score']:.2f}, "
                      f"final={contrib['final_score']:.2f}, modifiers=[{modifiers_str}]")
        
        print("\nReasoning:")
        positive_words = [c for c in contributions if c['final_score'] > 0]
        negative_words = [c for c in contributions if c['final_score'] < 0]
        
        if positive_words:
            pos_sum = sum(c['final_score'] for c in positive_words)
            print(f"Positive contribution: {pos_sum:.2f} from {len(positive_words)} words")
        
        if negative_words:
            neg_sum = sum(c['final_score'] for c in negative_words)
            print(f"Negative contribution: {neg_sum:.2f} from {len(negative_words)} words")


# Example usage
if __name__ == "__main__":
    analyzer = LexiconSentimentAnalyzer()
    
    # Test sentences
    test_sentences = [
        "I really love the service but the delivery was awful.",
        "The food was not good at all.",
        "Extremely disappointed with the poor quality.",
        "Very happy with the excellent service!",
        "The product was quite nice but expensive."
    ]
    
    for sentence in test_sentences:
        analyzer.explain_prediction(sentence)
        print("=" * 70) 