"""
Automated unit and integration tests for SentimentAnalyzer.
"""
from pathlib import Path
import pytest
from src.sentiment_analyzer import SentimentAnalyzer, ModelLoadError


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_analyzer_initialization(analyzer):
    assert analyzer.is_loaded is True
    assert analyzer.model is not None
    assert analyzer.vectorizer is not None
    assert len(analyzer.feature_names) > 0


def test_predict_positive_sentiment(analyzer):
    text = "An absolute masterpiece of modern cinema! Brilliant performances and direction."
    result = analyzer.predict(text)
    assert result["sentiment"] == "Positive"
    assert result["predicted_class"] == 1
    assert result["confidence"] > 0.50
    assert result["sentiment_score"] > 0.0
    assert result["probabilities"]["Positive"] > result["probabilities"]["Negative"]


def test_predict_negative_sentiment(analyzer):
    text = "Terrible script, agonizing pacing, and completely unconvincing performances. Waste of time."
    result = analyzer.predict(text)
    assert result["sentiment"] == "Negative"
    assert result["predicted_class"] == 0
    assert result["confidence"] > 0.50
    assert result["sentiment_score"] < 0.0
    assert result["probabilities"]["Negative"] > result["probabilities"]["Positive"]


def test_predict_output_structure(analyzer):
    text = "A decent film with enjoyable moments."
    result = analyzer.predict(text)
    expected_keys = {
        "sentiment",
        "predicted_class",
        "confidence",
        "sentiment_score",
        "probabilities",
        "cleaned_text",
        "top_features",
        "model_name"
    }
    assert expected_keys.issubset(set(result.keys()))
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["confidence"], float)
    assert 0.50 <= result["confidence"] <= 1.00
    assert -1.00 <= result["sentiment_score"] <= 1.00


def test_predict_empty_text_raises_error(analyzer):
    with pytest.raises(ValueError) as exc:
        analyzer.predict("   ")
    assert "Empty or whitespace-only" in str(exc.value)


def test_explainable_ai_features(analyzer):
    text = "Astonishing cinematography and superb acting in an excellent film."
    result = analyzer.predict(text)
    features = result["top_features"]
    assert isinstance(features, list)
    assert len(features) > 0
    # Each feature explanation should have word, weight, impact, direction
    first = features[0]
    assert "word" in first
    assert "weight" in first
    assert "impact" in first
    assert first["direction"] in ["Positive", "Negative"]


def test_batch_predict(analyzer):
    texts = [
        "A truly wonderful and inspiring cinematic experience.",
        "Dreadful and boring movie with terrible acting.",
        ""  # Blank to test edge case
    ]
    batch_res = analyzer.batch_predict(texts)
    assert len(batch_res) == 3
    
    assert batch_res[0]["predicted_sentiment"] == "Positive"
    assert batch_res[0]["status"] == "Success"
    
    assert batch_res[1]["predicted_sentiment"] == "Negative"
    assert batch_res[1]["status"] == "Success"
    
    # Check skipped row
    assert batch_res[2]["predicted_sentiment"] == "Skipped"
    assert batch_res[2]["status"] == "Invalid/Empty text"


def test_missing_model_files_error():
    fake_path = Path("/non_existent_dir/fake_model.joblib")
    with pytest.raises(ModelLoadError):
        SentimentAnalyzer(model_path=fake_path, vectorizer_path=fake_path)
