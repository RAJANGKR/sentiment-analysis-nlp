"""
End-to-end integration tests using Streamlit AppTest framework.
Verifies interactive single-text inference, validation, batch CSV loading, and dashboard flows.
"""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app():
    """Initialize a fresh AppTest instance of app.py."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    return at


def test_app_initial_render(app):
    """Verify application renders without uncaught exceptions and shows navigation."""
    assert len(app.exception) == 0
    assert len(app.sidebar.radio) > 0
    assert app.sidebar.radio[0].value == "📝 Single Text Analysis"


def test_single_text_analysis_flow(app):
    """Test entering text and running sentiment prediction."""
    # Input positive text
    app.text_area[0].input(
        "An astonishing and sublime cinematic triumph with brilliant performances!"
    ).run()
    
    # Click Analyze Sentiment button
    btn_analyze = [b for b in app.button if "Analyze Sentiment" in b.label][0]
    btn_analyze.click().run()
    
    assert len(app.exception) == 0
    # Check metrics rendered
    labels = [m.label for m in app.metric]
    assert "Primary Sentiment" in labels
    assert "Positive Probability" in labels
    assert "Negative Probability" in labels
    assert "Polarity Index" in labels
    
    # Check that sentiment was predicted as Positive
    sentiment_metric = [m for m in app.metric if m.label == "Primary Sentiment"][0]
    assert sentiment_metric.value == "Positive"


def test_single_text_empty_validation(app):
    """Test empty input validation warning."""
    app.text_area[0].input("").run()
    btn_analyze = [b for b in app.button if "Analyze Sentiment" in b.label][0]
    btn_analyze.click().run()
    
    assert len(app.warning) > 0
    assert "Empty or whitespace-only" in app.warning[0].value


def test_batch_csv_full_pipeline(app):
    """Test loading sample reviews, running batch analysis, and validating metrics."""
    # Switch to Batch CSV tab
    app.sidebar.radio[0].set_value("📊 Batch CSV Analysis").run()
    assert len(app.exception) == 0
    
    # Click Load Sample Reviews
    btn_load = [b for b in app.button if "Load Sample Reviews" in b.label][0]
    btn_load.click().run()
    
    assert len(app.exception) == 0
    assert len(app.success) > 0
    assert "Successfully loaded" in app.success[0].value
    
    # Check default selected column is review_text
    assert app.selectbox[0].value == "review_text"
    
    # Click Run Batch Sentiment Analysis
    btn_run = [b for b in app.button if "Run Batch Sentiment Analysis" in b.label][0]
    btn_run.click().run()
    
    assert len(app.exception) == 0
    
    # Verify summary metrics
    metric_dict = {m.label: m.value for m in app.metric}
    assert metric_dict.get("Total Records") == "20"
    assert metric_dict.get("Successfully Analyzed") == "19"
    assert metric_dict.get("Skipped / Blank") == "1"
    assert metric_dict.get("Positive Sentiment") == "11"
    assert metric_dict.get("Negative Sentiment") == "8"


def test_analytics_and_methodology_navigation(app):
    """Verify Analytics and Methodology tabs render cleanly without errors."""
    # Switch to Analytics
    app.sidebar.radio[0].set_value("📈 Analytics & Insights").run()
    assert len(app.exception) == 0
    
    # Switch to Methodology
    app.sidebar.radio[0].set_value("🧠 Methodology & Limitations").run()
    assert len(app.exception) == 0
