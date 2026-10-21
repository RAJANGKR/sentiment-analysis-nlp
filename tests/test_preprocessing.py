"""
Automated unit tests for preprocessing and validation.
"""
import pytest
from src.preprocessing import clean_text, validate_text, get_text_statistics


def test_validate_text_valid():
    is_valid, err = validate_text("A great movie with fantastic actors!")
    assert is_valid is True
    assert err is None


def test_validate_text_empty():
    is_valid, err = validate_text("")
    assert is_valid is False
    assert "Empty or whitespace-only" in err


def test_validate_text_whitespace():
    is_valid, err = validate_text("    \n\t   ")
    assert is_valid is False
    assert "Empty or whitespace-only" in err


def test_validate_text_none():
    is_valid, err = validate_text(None)
    assert is_valid is False
    assert "cannot be None" in err


def test_validate_text_too_long():
    long_text = "word " * 1500
    is_valid, err = validate_text(long_text, max_length=500)
    assert is_valid is False
    assert "too long" in err


def test_clean_text_html_tags():
    raw = "<p>This is a <b>wonderful</b> film!<br />Loved it.</p>"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "wonderful" in cleaned
    assert "loved it" in cleaned


def test_clean_text_urls():
    raw = "Check the review at https://imdb.com/title/123 or www.movies.com! Great film."
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "www." not in cleaned
    assert "great film" in cleaned


def test_clean_text_entities():
    raw = "Rock &amp; Roll cinema &lt;br&gt; thrills!"
    cleaned = clean_text(raw)
    assert "&amp;" not in cleaned
    assert "&" in cleaned or "rock" in cleaned


def test_clean_text_whitespace():
    raw = "  Lots    of   extra \n\n  spaces   and tabs \t\t here.  "
    cleaned = clean_text(raw)
    assert "  " not in cleaned
    assert cleaned.startswith("lots")
    assert cleaned.endswith("here.")


def test_get_text_statistics():
    sample = "The direction was superb. Acting was top-notch!"
    stats = get_text_statistics(sample)
    assert stats["word_count"] > 0
    assert stats["sentence_count"] == 2
    assert stats["char_count"] == len(sample)
