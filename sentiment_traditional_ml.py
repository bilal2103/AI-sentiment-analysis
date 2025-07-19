import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
from typing import List, Tuple, Dict

class TraditionalSentimentAnalyzer:
    """
    Traditional ML approach for sentiment analysis with feature importance.
    Uses TF-IDF vectorization and Logistic Regression with interpretability.
    """
    
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0  # Regularization parameter
        )
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.classifier)
        ])
        self.is_trained = False
        self.feature_names = None
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def train_on_sample_data(self):
        """Train on sample data (in real scenario, use your own dataset)"""
        # Sample training data
        sample_data = [
            ("I love this product, it's amazing!", 1),
            ("This is the worst thing I've ever bought", 0),
            ("Great quality and excellent service", 1),
            ("Terrible experience, very disappointed", 0),
            ("The product is okay, nothing special", 0),
            ("Outstanding performance, highly recommend", 1),
            ("Poor quality, waste of money", 0),
            ("Fantastic value for money", 1),
            ("Not worth the price, disappointed", 0),
            ("Excellent customer service, very happy", 1),
            ("The delivery was slow but product is good", 1),
            ("Service was awful, never again", 0),
            ("Really satisfied with the purchase", 1),
            ("Quality is poor, expected better", 0),
            ("Amazing product, exceeded expectations", 1),
            ("Very bad experience overall", 0),
            ("Good product, would buy again", 1),
            ("Disappointed with the service", 0),
            ("Perfect product, great value", 1),
            ("Waste of time and money", 0),
            ("The food was delicious and fresh", 1),
            ("Staff was rude and unhelpful", 0),
            ("Beautiful design and good quality", 1),
            ("Horrible experience, avoid this", 0),
            ("Satisfied with the overall experience", 1),
            ("Not recommended, poor quality", 0),
            ("Excellent service, will come back", 1),
            ("Very disappointed with the quality", 0),
            ("Great product, fast delivery", 1),
            ("Poor customer service experience", 0),
        ]
        
        texts = [self.preprocess_text(text) for text, _ in sample_data]
        labels = [label for _, label in sample_data]
        
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print("Model trained on sample data!")
        print(f"Features: {len(self.feature_names)}")
        print(f"Training samples: {len(texts)}")
    
    def predict_with_reasoning(self, text: str) -> Tuple[int, float, List[Dict]]:
        """
        Predict sentiment with detailed reasoning based on feature importance.
        Returns: (prediction, confidence, top_features)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_on_sample_data() first.")
        
        processed_text = self.preprocess_text(text)
        
        # Get prediction and probabilities
        prediction = self.pipeline.predict([processed_text])[0]
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        confidence = max(probabilities)
        
        # Get feature importance
        feature_importance = self.get_feature_importance_for_text(processed_text)
        
        return prediction, confidence, feature_importance
    
    def get_feature_importance_for_text(self, text: str) -> List[Dict]:
        """Get feature importance for a specific text"""
        # Transform text to TF-IDF features
        tfidf_features = self.vectorizer.transform([text])
        
        # Get coefficients from logistic regression
        coefficients = self.classifier.coef_[0]
        
        # Calculate feature importance (TF-IDF score * coefficient)
        feature_importance = []
        
        # Get non-zero features (features that appear in the text)
        feature_indices = tfidf_features.nonzero()[1]
        
        for idx in feature_indices:
            feature_name = self.feature_names[idx]
            tfidf_score = tfidf_features[0, idx]
            coefficient = coefficients[idx]
            importance = tfidf_score * coefficient
            
            feature_importance.append({
                'feature': feature_name,
                'tfidf_score': tfidf_score,
                'coefficient': coefficient,
                'importance': importance,
                'sentiment_direction': 'positive' if coefficient > 0 else 'negative'
            })
        
        # Sort by absolute importance
        feature_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return feature_importance
    
    def explain_prediction(self, text: str, top_n: int = 10) -> None:
        """Print detailed explanation of sentiment prediction"""
        prediction, confidence, features = self.predict_with_reasoning(text)
        
        sentiment_label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        
        print(f"Text: '{text}'")
        print(f"Prediction: {sentiment_label}")
        print(f"Confidence: {confidence:.3f}")
        print(f"\nTop {top_n} most important features:")
        print("-" * 80)
        
        for i, feature in enumerate(features[:top_n]):
            print(f"{i+1:2d}. '{feature['feature']}' "
                  f"(TF-IDF: {feature['tfidf_score']:.3f}, "
                  f"Coeff: {feature['coefficient']:.3f}, "
                  f"Importance: {feature['importance']:.3f}) "
                  f"-> {feature['sentiment_direction']}")
        
        print("\nReasoning:")
        positive_features = [f for f in features if f['importance'] > 0]
        negative_features = [f for f in features if f['importance'] < 0]
        
        if positive_features:
            pos_sum = sum(f['importance'] for f in positive_features)
            print(f"Positive contribution: {pos_sum:.3f} from {len(positive_features)} features")
        
        if negative_features:
            neg_sum = sum(f['importance'] for f in negative_features)
            print(f"Negative contribution: {neg_sum:.3f} from {len(negative_features)} features")
    
    def get_top_features_by_class(self, top_n: int = 20) -> Dict:
        """Get top features for each sentiment class"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        coefficients = self.classifier.coef_[0]
        
        # Get top positive and negative features
        top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
        top_negative_idx = np.argsort(coefficients)[:top_n]
        
        top_features = {
            'positive': [
                {
                    'feature': self.feature_names[idx],
                    'coefficient': coefficients[idx]
                }
                for idx in top_positive_idx
            ],
            'negative': [
                {
                    'feature': self.feature_names[idx],
                    'coefficient': coefficients[idx]
                }
                for idx in top_negative_idx
            ]
        }
        
        return top_features
    
    def print_model_insights(self, top_n: int = 10) -> None:
        """Print insights about what the model learned"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        top_features = self.get_top_features_by_class(top_n)
        
        print("Model Insights - Top Features by Sentiment:")
        print("=" * 60)
        
        print(f"\nTop {top_n} POSITIVE indicators:")
        for i, feature in enumerate(top_features['positive']):
            print(f"{i+1:2d}. '{feature['feature']}' (coeff: {feature['coefficient']:.3f})")
        
        print(f"\nTop {top_n} NEGATIVE indicators:")
        for i, feature in enumerate(top_features['negative']):
            print(f"{i+1:2d}. '{feature['feature']}' (coeff: {feature['coefficient']:.3f})")


# Example usage
if __name__ == "__main__":
    analyzer = TraditionalSentimentAnalyzer()
    
    # Train the model
    analyzer.train_on_sample_data()
    
    # Print model insights
    analyzer.print_model_insights()
    
    print("\n" + "=" * 80)
    print("TESTING PREDICTIONS")
    print("=" * 80)
    
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
        print("=" * 80) 