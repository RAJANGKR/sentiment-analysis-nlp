"""
Automated unit tests for CSV data processing and batch aggregation.
"""
from pathlib import Path
import pytest
import pandas as pd
from src.data_processing import (
    load_csv_file,
    detect_text_columns,
    process_batch_dataframe,
    export_dataframe_to_csv,
    CSVProcessingError
)
from src.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_load_csv_valid():
    csv_content = b"id,review\n1,Great movie\n2,Awful plot\n"
    df = load_csv_file(csv_content)
    assert len(df) == 2
    assert "review" in df.columns


def test_load_csv_empty_raises_error():
    empty_csv = b""
    with pytest.raises(CSVProcessingError) as exc:
        load_csv_file(empty_csv)
    assert "no data rows" in str(exc.value) or "Malformed" in str(exc.value)


def test_load_csv_oversized_raises_error():
    huge_bytes = b"a" * (20 * 1024 * 1024)
    with pytest.raises(CSVProcessingError) as exc:
        load_csv_file(huge_bytes, max_mb=10)
    assert "exceeds maximum allowed limit" in str(exc.value)


def test_detect_text_columns():
    df = pd.DataFrame({
        "id": [1, 2],
        "rating": [5, 1],
        "comment": ["Loved this film", "Hated every minute"],
        "category": ["Drama", "Action"]
    })
    text_cols = detect_text_columns(df)
    assert "comment" in text_cols
    assert "category" in text_cols


def test_process_batch_dataframe(analyzer):
    df = pd.DataFrame({
        "user": ["Alice", "Bob", "Charlie"],
        "feedback": [
            "Outstanding performance and wonderful music!",
            "Terrible script and dreadful acting.",
            ""  # Blank entry to test skipping
        ]
    })
    result_df, metrics = process_batch_dataframe(df, "feedback", analyzer)
    
    # Check DataFrame augmented columns
    assert "predicted_sentiment" in result_df.columns
    assert "sentiment_score" in result_df.columns
    assert "confidence" in result_df.columns
    assert "analysis_status" in result_df.columns
    
    # Check metrics
    assert metrics["total_records"] == 3
    assert metrics["analyzed_count"] == 2
    assert metrics["skipped_count"] == 1
    assert metrics["positive_count"] == 1
    assert metrics["negative_count"] == 1
    assert metrics["positive_pct"] == 50.0
    assert metrics["negative_pct"] == 50.0
    assert len(metrics["most_positive"]) == 1
    assert len(metrics["most_negative"]) == 1


def test_process_batch_missing_column(analyzer):
    df = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(CSVProcessingError) as exc:
        process_batch_dataframe(df, "non_existent_col", analyzer)
    assert "not found in DataFrame" in str(exc.value)


def test_export_dataframe_to_csv():
    df = pd.DataFrame({"col1": ["val1", "val2"], "col2": [10, 20]})
    exported_bytes = export_dataframe_to_csv(df)
    assert isinstance(exported_bytes, bytes)
    assert b"col1,col2" in exported_bytes
    assert b"val1,10" in exported_bytes
