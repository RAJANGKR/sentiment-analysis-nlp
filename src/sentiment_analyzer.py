"""
Sentiment Analyzer module for SentimentScope.
Loads model and vectorizer, performs single and batch inference,
calculates confidence and polarity, and provides Explainable AI (XAI) token weights.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
import numpy as np
import joblib

from src.config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    ORIGINAL_MODEL_PATH,
    ORIGINAL_VECTORIZER_PATH,
    LABEL_MAP
)
from src.preprocessing import clean_text, validate_text


class ModelLoadError(Exception):
    """Raised when sentiment model or vectorizer fails to load."""
    pass


class SentimentAnalyzer:
    """
    Sentiment Analyzer using pre-trained Scikit-learn Logistic Regression
    and TF-IDF Vectorizer trained on IMDB 50k movie reviews.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None
    ):
        self.custom_paths = bool(model_path or vectorizer_path)
        self.model_path = model_path or MODEL_PATH
        self.vectorizer_path = vectorizer_path or VECTORIZER_PATH
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.coefficients = None
        self.intercept = None
        self.is_loaded = False
        
        self.load_models()

    def load_models(self) -> None:
        """
        Load serialized model and vectorizer with fallback support for defaults.
        """
        # If explicit custom paths were given, only attempt those paths
        if self.custom_paths:
            targets = [(self.model_path, self.vectorizer_path)]
        else:
            targets = [
                (self.model_path, self.vectorizer_path),
                (ORIGINAL_MODEL_PATH, ORIGINAL_VECTORIZER_PATH)
            ]
        
        last_error = None
        for m_path, v_path in targets:
            if m_path.exists() and v_path.exists():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.model = joblib.load(m_path)
                        self.vectorizer = joblib.load(v_path)
                    
                    self.feature_names = np.array(self.vectorizer.get_feature_names_out())
                    self.coefficients = self.model.coef_[0]
                    self.intercept = float(self.model.intercept_[0]) if hasattr(self.model, "intercept_") else 0.0
                    self.is_loaded = True
                    return
                except Exception as e:
                    last_error = e
                    continue

        raise ModelLoadError(
            f"Failed to load sentiment analysis model or vectorizer. "
            f"Searched: {self.model_path} and {ORIGINAL_MODEL_PATH}. Error: {last_error}"
        )

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text string.
        
        Returns:
            Dict containing:
            - sentiment: 'Positive' or 'Negative'
            - predicted_class: 1 or 0
            - confidence: float (0.50 to 1.00)
            - sentiment_score: float (-1.00 to +1.00, polarity margin)
            - probabilities: dict of {'Negative': p0, 'Positive': p1}
            - cleaned_text: preprocessed text string
            - top_features: list of dicts with word, impact, and polarity direction
        """
        if not self.is_loaded:
            raise ModelLoadError("Model is not loaded. Call load_models() first.")
        
        is_valid, error = validate_text(text)
        if not is_valid:
            raise ValueError(error)

        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        
        # Probabilities and class
        probs = self.model.predict_proba(vec)[0]  # [P(Negative), P(Positive)]
        neg_prob = float(probs[0])
        pos_prob = float(probs[1])
        
        predicted_class = int(np.argmax(probs))
        sentiment = LABEL_MAP.get(predicted_class, "Unknown")
        confidence = float(max(neg_prob, pos_prob))
        
        # Polarity score defined as (P(Positive) - P(Negative)) in range [-1.0, 1.0]
        sentiment_score = round(pos_prob - neg_prob, 4)

        # Explainable AI: Feature contributions
        top_features = self._explain_prediction(vec)

        return {
            "sentiment": sentiment,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "sentiment_score": sentiment_score,
            "probabilities": {
                "Negative": round(neg_prob, 4),
                "Positive": round(pos_prob, 4)
            },
            "cleaned_text": cleaned,
            "top_features": top_features,
            "model_name": "Logistic Regression (TF-IDF 5K, IMDB Trained)"
        }

    def _explain_prediction(self, vec, top_n: int = 6) -> List[Dict[str, Any]]:
        """
        Extract the most influential TF-IDF terms for the current prediction.
        """
        if vec.nnz == 0:
            return []
        
        # Non-zero indices in sparse vector
        _, col_indices = vec.nonzero()
        contributions = []
        
        for idx in col_indices:
            tfidf_val = vec[0, idx]
            weight = self.coefficients[idx]
            impact = tfidf_val * weight
            contributions.append({
                "word": str(self.feature_names[idx]),
                "tfidf": round(float(tfidf_val), 4),
                "weight": round(float(weight), 4),
                "impact": round(float(impact), 4),
                "direction": "Positive" if impact > 0 else "Negative"
            })
            
        # Sort by absolute impact descending
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:top_n]

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of texts efficiently using vectorized operations.
        """
        if not self.is_loaded:
            raise ModelLoadError("Model is not loaded. Call load_models() first.")
        
        cleaned_texts = [clean_text(t) if isinstance(t, str) else "" for t in texts]
        vectors = self.vectorizer.transform(cleaned_texts)
        
        probs = self.model.predict_proba(vectors)
        predictions = self.model.predict(vectors)
        
        results = []
        for i in range(len(texts)):
            # If the text was blank or invalid
            orig = texts[i]
            if not isinstance(orig, str) or not orig.strip():
                results.append({
                    "predicted_sentiment": "Skipped",
                    "predicted_class": -1,
                    "confidence": 0.0,
                    "sentiment_score": 0.0,
                    "prob_negative": 0.0,
                    "prob_positive": 0.0,
                    "status": "Invalid/Empty text"
                })
                continue
            
            p_neg = float(probs[i][0])
            p_pos = float(probs[i][1])
            cls_idx = int(predictions[i])
            sentiment = LABEL_MAP.get(cls_idx, "Unknown")
            confidence = float(max(p_neg, p_pos))
            score = round(p_pos - p_neg, 4)
            
            results.append({
                "predicted_sentiment": sentiment,
                "predicted_class": cls_idx,
                "confidence": round(confidence, 4),
                "sentiment_score": score,
                "prob_negative": round(p_neg, 4),
                "prob_positive": round(p_pos, 4),
                "status": "Success"
            })
            
        return results
