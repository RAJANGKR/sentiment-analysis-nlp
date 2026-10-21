"""
Text preprocessing and validation utilities for SentimentScope.
"""
import re
import html
from typing import Tuple, Optional
from src.config import MAX_TEXT_LENGTH


def validate_text(text: Optional[str], max_length: int = MAX_TEXT_LENGTH) -> Tuple[bool, Optional[str]]:
    """
    Validate raw input text before sentiment analysis.
    
    Returns:
        (is_valid, error_message): Tuple indicating whether input is valid,
        and an explanatory error message if invalid.
    """
    if text is None:
        return False, "Input text cannot be None."
    
    stripped = text.strip()
    if not stripped:
        return False, "Please enter some text to analyze. Empty or whitespace-only inputs cannot be processed."
    
    if len(stripped) > max_length:
        return False, f"Input is too long ({len(stripped):,} characters). Please limit your review to {max_length:,} characters."
    
    return True, None


def clean_text(text: str) -> str:
    """
    Clean and normalize text for model inference.
    
    Preserves model compatibility with the training preprocessing pipeline
    from the original project while resolving raw HTML and formatting artifacts:
    1. Unescapes HTML entities (e.g. &amp; -> &)
    2. Strips HTML tags (e.g. <br />, <div>)
    3. Removes HTTP/HTTPS web links
    4. Normalizes multiple whitespaces into single spaces
    5. Converts to lowercase for vocabulary matching
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # 4. Standardize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # 5. Lowercase and strip leading/trailing spaces
    return text.strip().lower()


def get_text_statistics(text: str) -> dict:
    """
    Calculate basic linguistic statistics for display in UI.
    """
    if not text:
        return {"char_count": 0, "word_count": 0, "sentence_count": 0}
    
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": max(1, len(sentences))
    }
