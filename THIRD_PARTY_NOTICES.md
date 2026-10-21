# Third-Party Notices & Attribution

This project, **SentimentScope**, incorporates and enhances code, serialized models, and concepts from the following open-source resources:

---

## 1. Primary Base Project: sentiment-analysis-reviews

* **Original Author:** Sonal Shinde
* **Source Repository:** [https://github.com/sonalshinde24/sentiment-analysis-reviews](https://github.com/sonalshinde24/sentiment-analysis-reviews)
* **Year:** 2025
* **Original Description:** An NLP web application built with Streamlit that predicts whether movie reviews are Positive or Negative using Logistic Regression and TF-IDF.
* **Original Assets Utilized:**
  * Base Logistic Regression classifier architecture trained on the IMDB 50k Dataset.
  * TF-IDF Vectorizer configuration (5,000 unigram features).
  * Original model serialization weights (preserved in `models/original/`).

### Enhancements & Modifications in SentimentScope:
* **Architecture Modernization:** Refactored from a monolithic 178-line script into modular software engineering layers (`src/config.py`, `src/preprocessing.py`, `src/sentiment_analyzer.py`, `src/data_processing.py`).
* **Resolved Preprocessing Inconsistency:** Fixed an inference flaw in the original project where text cleaning used during notebook training was bypassed in `app.py`. Added comprehensive HTML unescaping, tag stripping, and URL sanitization.
* **Explainable AI (XAI) Attribution:** Added dynamic token weight extraction calculating the mathematical impact of individual vocabulary terms ($x_i \cdot \beta_i$) on positive vs. negative predictions.
* **Batch CSV Processing Engine:** Added secure CSV file uploading, automatic text-column detection, batch vectorization, metrics calculation (percentages, confidence, polarity indices), and downloadable annotated CSV exports.
* **Comprehensive Automated Testing:** Developed a full test suite with 25 unit and integration tests covering text validation, model inference, error handling, and CSV data workflows using `pytest`.
* **State-of-the-Art UI/UX:** Built a multi-page dashboard with custom analytics, responsive layouts, interactive Plotly visualizations, confidence gauges, and spotlight cards.

---

## 2. IMDB Dataset of 50K Movie Reviews

* **Original Creators:** Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts (2011).
* **Reference:** "Learning Word Vectors for Sentiment Analysis", The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
* **Dataset Scope:** 50,000 highly polar movie reviews for binary sentiment classification (25,000 train, 25,000 test).

---

## 3. Python Open-Source Libraries

* **Scikit-Learn:** BSD 3-Clause License (Pedregosa et al., 2011)
* **Streamlit:** Apache License 2.0 (Streamlit Inc.)
* **Pandas:** BSD 3-Clause License (AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team)
* **NumPy:** BSD 3-Clause License (NumPy Developers)
* **Plotly:** MIT License (Plotly, Inc.)
* **Pytest:** MIT License (Holger Krekel and pytest-dev team)
