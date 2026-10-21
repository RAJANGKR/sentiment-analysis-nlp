"""
Configuration settings for SentimentScope.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model and Vectorizer filepaths
# We default to the re-serialized joblib models for scikit-learn compatibility,
# with fallback to original pickle files if needed.
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
ORIGINAL_MODEL_PATH = MODELS_DIR / "original" / "model.pkl"
ORIGINAL_VECTORIZER_PATH = MODELS_DIR / "original" / "vectorizer.pkl"

# Constraints and Limits
MAX_TEXT_LENGTH = 5000          # Maximum characters for single review input
MAX_UPLOAD_SIZE_MB = 15         # Maximum CSV file upload size in MB
MAX_BATCH_ROWS = 5000           # Maximum CSV rows to process in one batch
DEFAULT_BATCH_SAMPLE = 500      # Recommended preview batch size

# Sentiment labels mapping
LABEL_MAP = {
    0: "Negative",
    1: "Positive"
}

SENTIMENT_COLORS = {
    "Positive": "#10B981",  # Vibrant Emerald
    "Negative": "#EF4444",  # Crimson Coral
    "Neutral": "#6B7280"    # Slate Gray
}

# Pre-defined example sentences for quick demonstration across linguistic categories
EXAMPLE_SENTENCES = {
    "🌟 Clearly Positive": "An absolute cinematic triumph with mesmerizing cinematography and brilliant performances throughout!",
    "💔 Clearly Negative": "A painfully dull and incoherent mess. Complete waste of two hours and awful acting.",
    "⚖️ Neutral / Factual": "The film was released in theaters in November 2021 and has a running time of 115 minutes.",
    "🎭 Mixed / Ambiguous": "The visual effects and musical score were stunning, but the plot was predictable and characters felt shallow.",
    "⚠️ Negation Challenge": "The acting was not good and I certainly cannot recommend this film to anyone.",
    "😏 Sarcastic / Context": "Oh fantastic, another two hours of my life spent watching CGI explosions with zero storyline."
}
