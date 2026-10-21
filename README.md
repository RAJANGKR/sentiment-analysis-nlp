# 🎯 SentimentScope — An Interactive Sentiment Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-25%20Passing-brightgreen?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**SentimentScope** is an end-to-end Natural Language Processing (NLP) and Machine Learning mini-project designed for college academic viva demonstrations, classroom peer review, and real-world sentiment exploration.

The project features real-time single review classification, Explainable AI (XAI) feature attribution, high-throughput batch CSV processing with automated analytics dashboards, and interactive visual reporting.

---

## 📌 Table of Contents
1. [Project Overview & Motivation](#-project-overview--motivation)
2. [Key Features](#-key-features)
3. [Technologies & Tools](#-technologies--tools)
4. [Original Repository & Attribution](#-original-repository--attribution)
5. [Model Architecture & NLP Pipeline](#-model-architecture--nlp-pipeline)
6. [Project Directory Structure](#-project-directory-structure)
7. [Prerequisites & System Requirements](#-prerequisites--system-requirements)
8. [Local Installation & Setup](#-local-installation--setup)
9. [Launching the Application](#-launching-the-application)
10. [Running Automated Tests](#-running-automated-tests)
11. [CSV Batch Processing Guide](#-csv-batch-processing-guide)
12. [Linguistic Edge Cases & Known Limitations (Viva Guide)](#-linguistic-edge-cases--known-limitations-viva-guide)
13. [Future Enhancements](#-future-enhancements)

---

## 💡 Project Overview & Motivation

Sentiment analysis is a fundamental problem in Natural Language Processing with widespread application in customer feedback analysis, brand monitoring, and recommendation engines. 

While basic sentiment models often exist as isolated Jupyter notebooks or toy command-line scripts, real-world deployment demands:
- **Consistent text preprocessing** that mirrors the training distribution.
- **Explainability:** Understanding *which* words led to a positive or negative decision.
- **Batch Processing:** Handling bulk CSV uploads safely without memory crashes or silent failures.
- **Interactive Dashboards:** Presenting statistical breakdowns, confidence intervals, and distributions visually.

**SentimentScope** bridges this gap by packaging a trained Machine Learning model into a production-grade, modular, and fully tested interactive dashboard.

---

## ✨ Key Features

### 1. ✍️ Single-Text Sentiment Analysis
- Instant binary sentiment classification (**Positive 😀** vs. **Negative 😞**).
- **Exact Posterior Probabilities:** Displays both $P(\text{Positive})$ and $P(\text{Negative})$.
- **Continuous Polarity Score:** Scaled from $-1.0$ (strongly negative) to $+1.0$ (strongly positive).
- **Real-time Gauge & Bar Visualizations:** Dynamic Plotly gauge chart and probability breakdown.
- **Explainable AI (XAI) Feature Chips:** Highlights specific vocabulary terms in the input text that contributed positively or negatively.

### 2. 💡 Curated Linguistic Examples
- 6 one-click demo presets illustrating diverse linguistic phenomena:
  - Clearly Positive
  - Clearly Negative
  - Neutral / Factual
  - Mixed / Ambiguous
  - Negation Challenge (e.g. *"not good"*)
  - Sarcasm & Contextual Irony

### 3. 📂 Batch CSV Analysis Engine
- Drag-and-drop CSV upload (supporting UTF-8, Latin-1, and Windows CP1252 encodings).
- Smart column detection for text reviews.
- Configurable row limits to maintain responsive UI performance.
- Resilient error handling: gracefully detects and skips blank rows without dropping data.
- **1-Click Download:** Export the annotated dataset with predicted sentiments, confidence values, and polarity scores.

### 4. 📊 Batch Results Analytics Dashboard
- Summary KPIs: Total Records, Successfully Analyzed, Skipped/Empty, Positive %, Negative %.
- **Interactive Donut Chart:** Visual breakdown of sentiment proportions.
- **Polarity Histogram:** Distribution of sentiment scores across the entire uploaded corpus.
- **Spotlight Cards:** Automatic extraction of the Top 3 most positive and Top 3 most negative reviews.
- **Filterable & Searchable Table:** Inspect records by sentiment filter or full-text keyword search.

### 5. 🧠 Methodology & Viva Evaluation Module
- Mathematical formulations of TF-IDF and Logistic Regression.
- Comparison matrix: Rule-Based (VADER) vs. Machine Learning (Logistic Regression).
- Detailed breakdown of linguistic failure modes (negations, sarcasm, domain shift).

---

## 🛠️ Technologies & Tools

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language execution |
| **Web Framework** | Streamlit | Reactive frontend and interactive widgets |
| **Machine Learning** | Scikit-learn 1.7+ | Logistic Regression & TF-IDF Vectorization |
| **Data Manipulation** | Pandas & NumPy | High-performance CSV parsing and matrix operations |
| **Interactive Visuals** | Plotly Express & Graph Objects | Dynamic gauge meters, histograms, and donut charts |
| **Model Serialization** | Joblib | High-efficiency persistence of estimators |
| **Automated Testing** | Pytest | Unit and integration test suite |

---

## 🏛️ Original Repository & Attribution

SentimentScope is built upon and significantly enhances an open-source sentiment analysis baseline:

* **Original Repository:** [sonalshinde24/sentiment-analysis-reviews](https://github.com/sonalshinde24/sentiment-analysis-reviews)
* **Original Developer:** Sonal Shinde (2025)
* **License & Notice:** Public open-source repository. See [`THIRD_PARTY_NOTICES.md`](file:///Users/rjn_g/Documents/%20Tech/Akash%20NLP/THIRD_PARTY_NOTICES.md) for full attribution.

### What Was Enhanced:
1. **Resolved Inference Inconsistency:** Fixed an issue in the original code where text cleaning applied during training (`clean_text`) was omitted during Streamlit inference.
2. **Added Batch CSV Processing:** The original project only supported entering one review at a time. SentimentScope adds a complete CSV upload, validation, and export pipeline.
3. **Engineered Explainable AI (XAI):** Built feature extraction calculating vocabulary weight contributions ($w_i \cdot x_i$) for full model interpretability.
4. **Refactored Architecture:** Replaced a monolithic script with a modular structure (`src/preprocessing.py`, `src/sentiment_analyzer.py`, `src/data_processing.py`).
5. **Created Automated Test Suite:** Added 25 automated tests covering validation, model inference, and CSV data integrity.

---

## 🤖 Model Architecture & NLP Pipeline

```
Raw Review Text
      │
      ▼
Text Preprocessing
(HTML Stripping, Unescaping, URL Removal, Lowercasing, Whitespace Normalization)
      │
      ▼
TF-IDF Feature Extraction
(5,000 Unigram Vocabulary trained on IMDB 50k Reviews)
      │
      ▼
Logistic Regression Classifier
(Sigmoid activation: z = β₀ + Σ wᵢ xᵢ)
      │
      ▼
Inference & Interpretation
├── Predicted Class (1: Positive, 0: Negative)
├── Confidence (max(P₀, P₁))
├── Polarity Score (P₁ - P₀ ∈ [-1.0, +1.0])
└── Explainable AI Feature Attribution
```

---

## 📁 Project Directory Structure

```text
Akash NLP/
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Comprehensive documentation
├── THIRD_PARTY_NOTICES.md         # Open-source attribution and license notices
├── app.py                         # Streamlit multi-page dashboard UI
├── src/                           # Core business and ML logic
│   ├── __init__.py
│   ├── config.py                  # Global settings, paths, limits, and color themes
│   ├── preprocessing.py           # Text cleaning, normalization, and validation
│   ├── sentiment_analyzer.py      # Model loading, single & batch inference, XAI
│   └── data_processing.py         # CSV loading, column validation, metrics & export
├── models/                        # Serialized ML model and vectorizer
│   ├── sentiment_model.joblib     # Logistic Regression classifier
│   ├── tfidf_vectorizer.joblib    # TF-IDF vectorizer (5k features)
│   └── original/                  # Original model weights preserved for audit
│       ├── model.pkl
│       └── vectorizer.pkl
├── data/                          # Datasets and sample files
│   └── sample_reviews.csv         # 20 diverse curated reviews for 1-click batch demo
└── tests/                         # Automated test suite (25 tests)
    ├── __init__.py
    ├── test_preprocessing.py      # Unit tests for text cleaning and validation
    ├── test_sentiment_analyzer.py # Unit tests for model inference and XAI
    └── test_data_processing.py    # Unit tests for CSV ingestion and batch metrics
```

---

## 💻 Prerequisites & System Requirements

- **Operating System:** macOS (Apple Silicon / Intel), Linux, or Windows 10/11.
- **Python Version:** Python 3.10 or higher.
- **RAM:** Minimum 2 GB RAM (lightweight footprint).
- **Disk Space:** ~200 MB for virtual environment and model files.

---

## 🚀 Local Installation & Setup

### Step 1: Clone or Navigate to the Workspace
```bash
git clone <your-repository-url>
cd "Akash NLP"
```

### Step 2: Create a Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Launching the Application

Activate your virtual environment and run:

```bash
streamlit run app.py
```

The application will launch and automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 Running Automated Tests

SentimentScope includes a comprehensive automated test suite built with `pytest`:

```bash
pytest -v
```

### Expected Output:
```text
============================= test session starts ==============================
platform darwin -- Python 3.10.x, pytest-9.x.x
rootdir: /path/to/Akash NLP
collected 25 items

tests/test_data_processing.py::test_load_csv_valid PASSED                [  4%]
tests/test_data_processing.py::test_load_csv_empty_raises_error PASSED   [  8%]
tests/test_data_processing.py::test_load_csv_oversized_raises_error PASSED [ 12%]
tests/test_data_processing.py::test_detect_text_columns PASSED           [ 16%]
tests/test_data_processing.py::test_process_batch_dataframe PASSED       [ 20%]
tests/test_data_processing.py::test_process_batch_missing_column PASSED  [ 24%]
tests/test_data_processing.py::test_export_dataframe_to_csv PASSED       [ 28%]
tests/test_preprocessing.py::test_validate_text_valid PASSED             [ 32%]
tests/test_preprocessing.py::test_validate_text_empty PASSED             [ 36%]
tests/test_preprocessing.py::test_validate_text_whitespace PASSED        [ 40%]
tests/test_preprocessing.py::test_validate_text_none PASSED              [ 44%]
tests/test_preprocessing.py::test_validate_text_too_long PASSED          [ 48%]
tests/test_preprocessing.py::test_clean_text_html_tags PASSED            [ 52%]
tests/test_preprocessing.py::test_clean_text_urls PASSED                 [ 56%]
tests/test_preprocessing.py::test_clean_text_entities PASSED             [ 60%]
tests/test_preprocessing.py::test_clean_text_whitespace PASSED           [ 64%]
tests/test_preprocessing.py::test_get_text_statistics PASSED             [ 68%]
tests/test_sentiment_analyzer.py::test_analyzer_initialization PASSED    [ 72%]
tests/test_sentiment_analyzer.py::test_predict_positive_sentiment PASSED [ 76%]
tests/test_sentiment_analyzer.py::test_predict_negative_sentiment PASSED [ 80%]
tests/test_sentiment_analyzer.py::test_predict_output_structure PASSED   [ 84%]
tests/test_sentiment_analyzer.py::test_predict_empty_text_raises_error PASSED [ 88%]
tests/test_sentiment_analyzer.py::test_explainable_ai_features PASSED    [ 92%]
tests/test_sentiment_analyzer.py::test_batch_predict PASSED              [ 96%]
tests/test_sentiment_analyzer.py::test_missing_model_files_error PASSED  [100%]

============================== 25 passed in 1.56s ==============================
```

---

## 📊 CSV Batch Processing Guide

To test batch analysis:
1. Navigate to the **📊 Batch CSV Analysis** tab in the sidebar.
2. Click **🧪 Load Sample Reviews** to immediately test with the built-in 20-review test dataset (`data/sample_reviews.csv`), or upload your own CSV file.
3. Select which column in your CSV contains the review text.
4. Set an optional row processing limit (default 1,000, max 5,000).
5. Click **🚀 Run Batch Sentiment Analysis**.
6. Inspect the KPI cards, donut chart, polarity histogram, and spotlight records.
7. Click **💾 Download Annotated CSV Results** to export your analyzed dataset.

---

## 🎓 Linguistic Edge Cases & Known Limitations (Viva Guide)

During a project viva or oral presentation, examiners frequently ask about the model's limitations:

### 1. The Negation Blind Spot
* **Question:** *"What does the model predict for 'The movie was not good' and why?"*
* **Explanation:** The model predicts **Positive** (~86% confidence). This occurs because the baseline TF-IDF vectorizer was trained with Scikit-learn's default `"english"` stopword list, which filters out the word `"not"`! The unigram model only sees the remaining token `"good"` and assigns it positive weight.
* **Viva Solution:** Explain that this limitation can be resolved using **bi-grams** (`ngram_range=(1, 2)`), custom stopword lists that retain negation markers, or context-aware architectures like BERT / Transformers.

### 2. Sarcasm & Ironic Expression
* **Question:** *"How does the model handle sarcasm like 'Oh brilliant, another two hours wasted'?"*
* **Explanation:** Unigram models perform bag-of-words accumulation. Highly weighted positive tokens like `"brilliant"` can outweigh negative terms if contextual polarity inversion is not modeled.

### 3. Domain Shift
* **Question:** *"Can this model be applied to financial news or healthcare reviews?"*
* **Explanation:** The model was trained specifically on movie reviews (IMDB). Vocabulary associations in movies (e.g. `"kills"`, `"unbelievable"`, `"wicked"`) have different contextual meanings in financial or medical settings.

---

## 🔮 Future Enhancements

- [ ] Support Bi-Gram and Tri-Gram TF-IDF feature extraction (`ngram_range=(1, 3)`).
- [ ] Add comparative side-by-side inference with NLTK VADER or RoBERTa.
- [ ] Add word cloud visualization for batch dataset clusters.
- [ ] Deploy as a live web application on Streamlit Community Cloud.

---

## 📄 License

This enhanced educational project is distributed under the **MIT License**. Third-party models and components retain their original copyright and licensing as detailed in [`THIRD_PARTY_NOTICES.md`](file:///Users/rjn_g/Documents/%20Tech/Akash%20NLP/THIRD_PARTY_NOTICES.md).
