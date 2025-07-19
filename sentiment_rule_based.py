import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class SentimentLabel(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

@dataclass
class Rule:
    """Represents a sentiment rule with pattern and reasoning"""
    pattern: str
    sentiment: SentimentLabel
    weight: float
    description: str
    rule_type: str

@dataclass
class RuleMatch:
    """Represents a matched rule with context"""
    rule: Rule
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float

class RuleBasedSentimentAnalyzer:
    """
    Rule-based sentiment analyzer with explicit reasoning patterns.
    Uses regex patterns and linguistic rules to classify sentiment.
    """
    
    def __init__(self):
        self.rules = []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize sentiment rules"""
        
        # Strong positive patterns
        self.rules.extend([
            Rule(r'\b(love|adore|amazing|excellent|fantastic|wonderful|perfect|outstanding|brilliant|superb)\b', 
                 SentimentLabel.POSITIVE, 2.5, "Strong positive words", "lexical"),
            Rule(r'\b(great|good|nice|pleased|satisfied|happy|delighted|enjoyed)\b', 
                 SentimentLabel.POSITIVE, 2.0, "Positive sentiment words", "lexical"),
            Rule(r'\b(recommend|definitely|absolutely|totally)\b', 
                 SentimentLabel.POSITIVE, 1.8, "Positive intensifiers", "intensifier"),
            Rule(r'\b(best|top|first-class|five-star|premium)\b', 
                 SentimentLabel.POSITIVE, 2.2, "Positive quality indicators", "quality"),
        ])
        
        # Strong negative patterns
        self.rules.extend([
            Rule(r'\b(hate|awful|terrible|horrible|disgusting|worst|pathetic|appalling)\b', 
                 SentimentLabel.NEGATIVE, -2.5, "Strong negative words", "lexical"),
            Rule(r'\b(bad|poor|disappointing|frustrated|angry|upset|annoyed|disappointed)\b', 
                 SentimentLabel.NEGATIVE, -2.0, "Negative sentiment words", "lexical"),
            Rule(r'\b(never|avoid|waste|useless|pointless|regret)\b', 
                 SentimentLabel.NEGATIVE, -1.8, "Negative action words", "action"),
            Rule(r'\b(expensive|overpriced|costly|pricey)\b', 
                 SentimentLabel.NEGATIVE, -1.2, "Price complaints", "price"),
        ])
        
        # Intensification patterns
        self.rules.extend([
            Rule(r'\b(very|extremely|incredibly|absolutely|totally|completely|really|quite|rather)\s+(\w+)', 
                 SentimentLabel.NEUTRAL, 0.0, "Intensifiers", "intensifier"),
            Rule(r'(\w+)\s+(but|however|although|though|yet|nevertheless)\s+(\w+)', 
                 SentimentLabel.NEUTRAL, 0.0, "Contrast patterns", "contrast"),
        ])
        
        # Negation patterns
        self.rules.extend([
            Rule(r'\b(not|no|never|none|nothing|nowhere|neither|nobody|nor|cannot|can\'t|won\'t|wouldn\'t|shouldn\'t|couldn\'t|doesn\'t|don\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t)\s+(\w+)', 
                 SentimentLabel.NEUTRAL, 0.0, "Negation patterns", "negation"),
        ])
        
        # Conditional patterns
        self.rules.extend([
            Rule(r'\bif\s+only\b', SentimentLabel.NEGATIVE, -1.5, "Wishful thinking (negative)", "conditional"),
            Rule(r'\bwould\s+be\s+better\b', SentimentLabel.NEGATIVE, -1.0, "Improvement suggestions", "conditional"),
            Rule(r'\bshould\s+have\b', SentimentLabel.NEGATIVE, -1.2, "Expectation not met", "conditional"),
        ])
        
        # Comparative patterns
        self.rules.extend([
            Rule(r'\bbetter\s+than\b', SentimentLabel.POSITIVE, 1.5, "Positive comparison", "comparative"),
            Rule(r'\bworse\s+than\b', SentimentLabel.NEGATIVE, -1.5, "Negative comparison", "comparative"),
            Rule(r'\bbest\s+(\w+)\s+ever\b', SentimentLabel.POSITIVE, 2.8, "Superlative positive", "comparative"),
            Rule(r'\bworst\s+(\w+)\s+ever\b', SentimentLabel.NEGATIVE, -2.8, "Superlative negative", "comparative"),
        ])
        
        # Emotional expressions
        self.rules.extend([
            Rule(r'[!]{2,}', SentimentLabel.POSITIVE, 1.0, "Excitement indicators", "emotion"),
            Rule(r'[?]{2,}', SentimentLabel.NEGATIVE, -0.8, "Confusion/frustration", "emotion"),
            Rule(r'[.]{3,}', SentimentLabel.NEGATIVE, -0.5, "Trailing off (uncertainty)", "emotion"),
        ])
        
        # Quality and service patterns
        self.rules.extend([
            Rule(r'\b(quality|service|staff|experience|product|delivery)\s+is\s+(excellent|great|good|amazing|wonderful|perfect)\b', 
                 SentimentLabel.POSITIVE, 2.0, "Positive quality assessment", "quality"),
            Rule(r'\b(quality|service|staff|experience|product|delivery)\s+is\s+(poor|bad|awful|terrible|horrible)\b', 
                 SentimentLabel.NEGATIVE, -2.0, "Negative quality assessment", "quality"),
        ])
    
    def find_rule_matches(self, text: str) -> List[RuleMatch]:
        """Find all rule matches in the text"""
        matches = []
        text_lower = text.lower()
        
        for rule in self.rules:
            pattern_matches = list(re.finditer(rule.pattern, text_lower))
            
            for match in pattern_matches:
                rule_match = RuleMatch(
                    rule=rule,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=abs(rule.weight)
                )
                matches.append(rule_match)
        
        return matches
    
    def apply_linguistic_rules(self, text: str, matches: List[RuleMatch]) -> List[RuleMatch]:
        """Apply linguistic rules to modify sentiment based on context"""
        modified_matches = []
        text_lower = text.lower()
        
        for match in matches:
            modified_match = match
            
            # Check for negation before the match
            negation_window = text_lower[max(0, match.start_pos-20):match.start_pos]
            negation_pattern = r'\b(not|no|never|none|nothing|nowhere|neither|nobody|nor|cannot|can\'t|won\'t|wouldn\'t|shouldn\'t|couldn\'t|doesn\'t|don\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t)\s*$'
            
            if re.search(negation_pattern, negation_window):
                # Flip and reduce sentiment
                new_weight = -match.rule.weight * 0.8
                modified_rule = Rule(
                    pattern=match.rule.pattern,
                    sentiment=SentimentLabel.NEGATIVE if match.rule.sentiment == SentimentLabel.POSITIVE else SentimentLabel.POSITIVE,
                    weight=new_weight,
                    description=f"Negated: {match.rule.description}",
                    rule_type=match.rule.rule_type
                )
                modified_match = RuleMatch(
                    rule=modified_rule,
                    matched_text=match.matched_text,
                    start_pos=match.start_pos,
                    end_pos=match.end_pos,
                    confidence=match.confidence
                )
            
            # Check for intensifiers before the match
            intensifier_window = text_lower[max(0, match.start_pos-15):match.start_pos]
            intensifier_pattern = r'\b(very|extremely|incredibly|absolutely|totally|completely|really|quite|rather)\s*$'
            intensifier_match = re.search(intensifier_pattern, intensifier_window)
            
            if intensifier_match:
                intensifier_word = intensifier_match.group().strip()
                multiplier = {
                    'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
                    'totally': 1.5, 'completely': 1.8, 'really': 1.3, 'quite': 1.2, 'rather': 1.1
                }.get(intensifier_word, 1.3)
                
                new_weight = modified_match.rule.weight * multiplier
                modified_rule = Rule(
                    pattern=modified_match.rule.pattern,
                    sentiment=modified_match.rule.sentiment,
                    weight=new_weight,
                    description=f"Intensified ({intensifier_word}): {modified_match.rule.description}",
                    rule_type=modified_match.rule.rule_type
                )
                modified_match = RuleMatch(
                    rule=modified_rule,
                    matched_text=modified_match.matched_text,
                    start_pos=modified_match.start_pos,
                    end_pos=modified_match.end_pos,
                    confidence=modified_match.confidence * multiplier
                )
            
            modified_matches.append(modified_match)
        
        return modified_matches
    
    def analyze_sentiment(self, text: str) -> Tuple[SentimentLabel, float, List[RuleMatch]]:
        """Analyze sentiment using rule-based approach"""
        
        # Find all rule matches
        matches = self.find_rule_matches(text)
        
        # Apply linguistic rules
        modified_matches = self.apply_linguistic_rules(text, matches)
        
        # Calculate overall sentiment score
        total_score = 0.0
        for match in modified_matches:
            total_score += match.rule.weight
        
        # Determine sentiment label
        if total_score > 0.5:
            sentiment = SentimentLabel.POSITIVE
        elif total_score < -0.5:
            sentiment = SentimentLabel.NEGATIVE
        else:
            sentiment = SentimentLabel.NEUTRAL
        
        return sentiment, total_score, modified_matches
    
    def explain_prediction(self, text: str) -> None:
        """Explain the sentiment prediction with detailed reasoning"""
        sentiment, score, matches = self.analyze_sentiment(text)
        
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment.value}")
        print(f"Score: {score:.3f}")
        print(f"\nRules applied ({len(matches)} matches):")
        print("-" * 80)
        
        # Group matches by rule type
        rule_types = {}
        for match in matches:
            rule_type = match.rule.rule_type
            if rule_type not in rule_types:
                rule_types[rule_type] = []
            rule_types[rule_type].append(match)
        
        # Display matches by type
        for rule_type, type_matches in rule_types.items():
            print(f"\n{rule_type.upper()} RULES:")
            for match in type_matches:
                sentiment_symbol = "+" if match.rule.weight > 0 else "-"
                print(f"  {sentiment_symbol} '{match.matched_text}' -> {match.rule.description} "
                      f"(weight: {match.rule.weight:.2f})")
        
        print(f"\nFinal Reasoning:")
        positive_matches = [m for m in matches if m.rule.weight > 0]
        negative_matches = [m for m in matches if m.rule.weight < 0]
        
        if positive_matches:
            pos_score = sum(m.rule.weight for m in positive_matches)
            print(f"Positive contribution: {pos_score:.3f} from {len(positive_matches)} rules")
        
        if negative_matches:
            neg_score = sum(m.rule.weight for m in negative_matches)
            print(f"Negative contribution: {neg_score:.3f} from {len(negative_matches)} rules")
        
        print(f"Net sentiment score: {score:.3f} -> {sentiment.value}")
    
    def get_rule_statistics(self) -> Dict:
        """Get statistics about the rules"""
        stats = {
            'total_rules': len(self.rules),
            'positive_rules': len([r for r in self.rules if r.weight > 0]),
            'negative_rules': len([r for r in self.rules if r.weight < 0]),
            'neutral_rules': len([r for r in self.rules if r.weight == 0]),
            'rule_types': {}
        }
        
        for rule in self.rules:
            if rule.rule_type not in stats['rule_types']:
                stats['rule_types'][rule.rule_type] = 0
            stats['rule_types'][rule.rule_type] += 1
        
        return stats


# Example usage
if __name__ == "__main__":
    analyzer = RuleBasedSentimentAnalyzer()
    
    # Print rule statistics
    stats = analyzer.get_rule_statistics()
    print("Rule-Based Sentiment Analyzer Statistics:")
    print(f"Total rules: {stats['total_rules']}")
    print(f"Positive rules: {stats['positive_rules']}")
    print(f"Negative rules: {stats['negative_rules']}")
    print(f"Neutral rules: {stats['neutral_rules']}")
    print(f"Rule types: {dict(stats['rule_types'])}")
    
    print("\n" + "=" * 80)
    print("TESTING PREDICTIONS")
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
        "If only the staff was more helpful.",
        "The quality is better than expected."
    ]
    
    for sentence in test_sentences:
        analyzer.explain_prediction(sentence)
        print("=" * 80) 