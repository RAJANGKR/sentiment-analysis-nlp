"""
Data processing module for CSV validation, batch sentiment analysis,
summary metric computation, and CSV export.
"""
from typing import Tuple, Dict, Any, Optional, List
import io
import pandas as pd
from src.config import MAX_UPLOAD_SIZE_MB, MAX_BATCH_ROWS
from src.sentiment_analyzer import SentimentAnalyzer


class CSVProcessingError(Exception):
    """Raised when CSV loading or processing encounters a data issue."""
    pass


def load_csv_file(
    file_bytes: bytes,
    filename: str = "upload.csv",
    max_mb: int = MAX_UPLOAD_SIZE_MB
) -> pd.DataFrame:
    """
    Safely load an uploaded CSV file with encoding fallbacks and size checks.
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise CSVProcessingError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_mb} MB."
        )

    # Attempt common text encodings
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    df = None
    last_err = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            raise CSVProcessingError(f"Malformed or unreadable CSV file: {str(e)}")

    if df is None:
        raise CSVProcessingError(f"Could not decode file with common encodings: {last_err}")

    if df.empty:
        raise CSVProcessingError("The uploaded CSV file contains no data rows.")

    return df


def detect_text_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify columns likely containing text reviews or feedback.
    """
    text_cols = []
    for col in df.columns:
        # Check if column is object or string type
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            text_cols.append(col)
            
    # If no object columns found, return all columns
    return text_cols if text_cols else list(df.columns)


def process_batch_dataframe(
    df: pd.DataFrame,
    text_column: str,
    analyzer: SentimentAnalyzer,
    row_limit: int = MAX_BATCH_ROWS
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute batch sentiment analysis on a selected text column in a DataFrame.
    
    Returns:
        (result_df, metrics_summary): The augmented DataFrame and computed metrics.
    """
    if text_column not in df.columns:
        raise CSVProcessingError(f"Column '{text_column}' not found in DataFrame.")

    total_uploaded = len(df)
    
    # Apply row limit to ensure responsive UI performance
    if total_uploaded > row_limit:
        work_df = df.iloc[:row_limit].copy()
        truncated = True
    else:
        work_df = df.copy()
        truncated = False

    # Extract text values, treating NaNs and non-strings safely
    raw_texts = [
        str(val).strip() if pd.notna(val) and str(val).strip() != "" else ""
        for val in work_df[text_column]
    ]

    # Run batch inference
    batch_results = analyzer.batch_predict(raw_texts)

    # Augment DataFrame with predictions
    work_df["predicted_sentiment"] = [r["predicted_sentiment"] for r in batch_results]
    work_df["sentiment_score"] = [r["sentiment_score"] for r in batch_results]
    work_df["confidence"] = [r["confidence"] for r in batch_results]
    work_df["analysis_status"] = [r["status"] for r in batch_results]

    # Calculate Summary Metrics
    total_records = len(work_df)
    valid_mask = work_df["analysis_status"] == "Success"
    analyzed_count = int(valid_mask.sum())
    skipped_count = total_records - analyzed_count

    pos_mask = (work_df["predicted_sentiment"] == "Positive") & valid_mask
    neg_mask = (work_df["predicted_sentiment"] == "Negative") & valid_mask

    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())

    pos_pct = round((pos_count / analyzed_count * 100), 1) if analyzed_count > 0 else 0.0
    neg_pct = round((neg_count / analyzed_count * 100), 1) if analyzed_count > 0 else 0.0

    avg_conf = (
        round(float(work_df.loc[valid_mask, "confidence"].mean()), 4)
        if analyzed_count > 0 else 0.0
    )
    avg_score = (
        round(float(work_df.loc[valid_mask, "sentiment_score"].mean()), 4)
        if analyzed_count > 0 else 0.0
    )

    # Top positive & negative records
    most_pos = []
    most_neg = []
    if analyzed_count > 0:
        valid_df = work_df[valid_mask]
        pos_df = valid_df[valid_df["sentiment_score"] > 0]
        sorted_pos = pos_df.sort_values(by="sentiment_score", ascending=False).head(3)
        for _, row in sorted_pos.iterrows():
            most_pos.append({
                "text": str(row[text_column])[:250],
                "score": float(row["sentiment_score"]),
                "confidence": float(row["confidence"])
            })

        neg_df = valid_df[valid_df["sentiment_score"] < 0]
        sorted_neg = neg_df.sort_values(by="sentiment_score", ascending=True).head(3)
        for _, row in sorted_neg.iterrows():
            most_neg.append({
                "text": str(row[text_column])[:250],
                "score": float(row["sentiment_score"]),
                "confidence": float(row["confidence"])
            })

    metrics = {
        "total_records": total_records,
        "total_uploaded": total_uploaded,
        "is_truncated": truncated,
        "row_limit": row_limit,
        "analyzed_count": analyzed_count,
        "skipped_count": skipped_count,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "average_confidence": avg_conf,
        "average_score": avg_score,
        "most_positive": most_pos,
        "most_negative": most_neg
    }

    return work_df, metrics


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to UTF-8 encoded CSV bytes for user download.
    """
    return df.to_csv(index=False).encode("utf-8")
