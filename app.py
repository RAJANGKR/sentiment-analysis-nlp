"""
SentimentScope - An Interactive Sentiment Analysis Dashboard
Built with Python, Scikit-learn, and Streamlit.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    EXAMPLE_SENTENCES,
    SENTIMENT_COLORS,
    MAX_TEXT_LENGTH,
    MAX_BATCH_ROWS,
    DATA_DIR
)
from src.preprocessing import clean_text, validate_text, get_text_statistics
from src.sentiment_analyzer import SentimentAnalyzer, ModelLoadError
from src.data_processing import (
    load_csv_file,
    detect_text_columns,
    process_batch_dataframe,
    export_dataframe_to_csv,
    CSVProcessingError
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SentimentScope | Sentiment Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Theme & Component Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Typography & Font Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: #F8FAFC;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 8px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #CBD5E1;
        max-width: 850px;
        line-height: 1.5;
        margin: 0;
    }
    .hero-badges {
        margin-top: 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        color: #F1F5F9;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Result Card Styling */
    .sentiment-card {
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.14) 100%);
        border-left: 6px solid #10B981;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(220, 38, 38, 0.14) 100%);
        border-left: 6px solid #EF4444;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-score {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 12px;
    }

    /* Metric Box */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Chip Styling for Contributing Words */
    .token-chip-pos {
        display: inline-block;
        background: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        border: 1px solid #A7F3D0;
    }
    .token-chip-neg {
        display: inline-block;
        background: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        border: 1px solid #FECACA;
    }

    /* Footer */
    .footer-container {
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 0.86rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Model Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing SentimentScope Engine...")
def load_sentiment_engine() -> SentimentAnalyzer:
    """Load and cache the SentimentAnalyzer instance."""
    return SentimentAnalyzer()


try:
    analyzer = load_sentiment_engine()
except ModelLoadError as e:
    st.error(f"⚠️ Model Loading Error: {e}")
    st.info("Please verify that models/sentiment_model.joblib exists in the project.")
    st.stop()


# ---------------------------------------------------------
# Sidebar Navigation & Context
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 SentimentScope")
    st.caption("Natural Language Processing Mini-Project")
    
    selected_page = st.radio(
        "Navigation",
        [
            "📝 Single Text Analysis",
            "📊 Batch CSV Analysis",
            "📈 Analytics & Insights",
            "🧠 Methodology & Limitations"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Model Pipeline Specs")
    st.markdown(
        """
        - **Algorithm:** Logistic Regression
        - **Features:** TF-IDF (5,000 unigrams)
        - **Dataset:** IMDB 50k Movie Reviews
        - **Inference Time:** ~1.2ms / review
        - **Classes:** Positive (1) vs Negative (0)
        """
    )
    
    st.markdown("---")
    st.markdown("#### 📁 Quick Demo Dataset")
    sample_csv_path = DATA_DIR / "sample_reviews.csv"
    if sample_csv_path.exists():
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample CSV",
                data=f.read(),
                file_name="sample_reviews.csv",
                mime="text/csv",
                help="Click to download a curated sample CSV with diverse test reviews."
            )
    
    st.markdown("---")
    st.caption("Based on open-source project by [Sonal Shinde](https://github.com/sonalshinde24/sentiment-analysis-reviews)")


# ---------------------------------------------------------
# Hero Banner (Rendered on all pages)
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span>🎯 SentimentScope</span>
    </div>
    <p class="hero-subtitle">
        An interactive Natural Language Processing dashboard for real-time text sentiment classification,
        explainable AI feature attribution, and high-throughput batch CSV analytics.
    </p>
    <div class="hero-badges">
        <span class="badge">🤖 Scikit-Learn 1.7</span>
        <span class="badge">📊 TF-IDF Vectorizer</span>
        <span class="badge">🎬 IMDB Trained</span>
        <span class="badge">⚡ Streamlit</span>
        <span class="badge">🔍 Explainable AI (XAI)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: Single Text Sentiment Analysis
# =========================================================
if selected_page == "📝 Single Text Analysis":
    st.markdown("### ✍️ Single-Text Sentiment Analysis")
    st.write(
        "Enter any English sentence, product review, or movie feedback below to determine "
        "its emotional polarity and inspect which vocabulary terms influenced the prediction."
    )

    # Example sentence selection
    col_ex, col_clear = st.columns([4, 1])
    with col_ex:
        selected_example_label = st.selectbox(
            "💡 Load an illustrative example sentence:",
            ["(Choose an example to load...)"] + list(EXAMPLE_SENTENCES.keys()),
            index=0,
            help="Select an example demonstrating strengths and edge cases like negations or sarcasm."
        )

    # Manage text session state
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    if selected_example_label != "(Choose an example to load...)":
        st.session_state.user_text = EXAMPLE_SENTENCES[selected_example_label]

    # Input text area
    user_input = st.text_area(
        "Enter text review here:",
        value=st.session_state.user_text,
        height=140,
        placeholder="Type or paste your text here (e.g. 'An outstanding performance with incredible cinematography!')...",
        max_chars=MAX_TEXT_LENGTH
    )

    # Action buttons
    col_btn1, col_btn2, col_info = st.columns([1.5, 1, 4])
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state.user_text = ""
            st.rerun()

    # Text statistics bar
    stats = get_text_statistics(user_input)
    with col_info:
        st.caption(
            f"Characters: **{stats['char_count']:,}** / {MAX_TEXT_LENGTH:,} | "
            f"Words: **{stats['word_count']}** | "
            f"Sentences: **{stats['sentence_count']}**"
        )

    # Inference Execution
    if analyze_clicked:
        is_valid, validation_msg = validate_text(user_input)
        
        if not is_valid:
            st.warning(f"⚠️ {validation_msg}")
        else:
            with st.spinner("Analyzing text polarity..."):
                result = analyzer.predict(user_input)

            sentiment = result["sentiment"]
            conf = result["confidence"]
            score = result["sentiment_score"]
            neg_p = result["probabilities"]["Negative"]
            pos_p = result["probabilities"]["Positive"]
            features = result["top_features"]

            # Visual Result Card
            card_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
            icon = "😀" if sentiment == "Positive" else "😞"
            
            st.markdown(f"""
            <div class="sentiment-card {card_class}">
                <div class="card-title">
                    <span>{icon} Predicted Sentiment: <b>{sentiment}</b></span>
                </div>
                <div class="card-score">
                    Model Confidence: <b>{conf:.1%}</b> &nbsp;|&nbsp; 
                    Polarity Score: <b>{score:+.4f}</b> (Scale: -1.0 to +1.0) &nbsp;|&nbsp;
                    Cleaned Input: <i>"{result['cleaned_text']}"</i>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics Breakdown Columns
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Primary Sentiment", sentiment, f"{'+' if sentiment == 'Positive' else '-'}{conf:.1%}")
            with col_m2:
                st.metric("Positive Probability", f"{pos_p:.1%}")
            with col_m3:
                st.metric("Negative Probability", f"{neg_p:.1%}")
            with col_m4:
                st.metric("Polarity Index", f"{score:+.3f}", delta=f"{sentiment}")

            # Visual Charts Section: Gauge & Probability Bar
            st.markdown("#### 📊 Probability & Score Distribution")
            chart_col1, chart_col2 = st.columns([1.2, 1.8])

            with chart_col1:
                # Gauge Meter for Polarity
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Sentiment Polarity Index", 'font': {'size': 16, 'family': 'Plus Jakarta Sans'}},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': SENTIMENT_COLORS[sentiment], 'thickness': 0.28},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#CBD5E1",
                        'steps': [
                            {'range': [-1, -0.1], 'color': 'rgba(239, 68, 68, 0.15)'},
                            {'range': [-0.1, 0.1], 'color': 'rgba(107, 114, 128, 0.1)'},
                            {'range': [0.1, 1], 'color': 'rgba(16, 185, 129, 0.15)'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.8,
                            'value': score
                        }
                    }
                ))
                gauge_fig.update_layout(height=240, margin=dict(l=25, r=25, t=40, b=15))
                st.plotly_chart(gauge_fig, use_container_width=True)

            with chart_col2:
                # Horizontal Probability Comparison Bar
                prob_df = pd.DataFrame({
                    "Sentiment": ["Negative", "Positive"],
                    "Probability": [neg_p, pos_p],
                    "Color": [SENTIMENT_COLORS["Negative"], SENTIMENT_COLORS["Positive"]]
                })
                bar_fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Sentiment",
                    orientation="h",
                    text=[f"{neg_p:.1%}", f"{pos_p:.1%}"],
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    title="Class Probability Distribution"
                )
                bar_fig.update_layout(
                    height=240,
                    margin=dict(l=10, r=20, t=40, b=15),
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    showlegend=False
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            # Explainable AI: Feature Contribution Tokens
            st.markdown("#### 🔍 Explainable AI (XAI): Vocabulary Feature Attribution")
            st.caption(
                "These are the specific words identified in the review that influenced the Logistic Regression model's decision. "
                "Green tokens pushed towards **Positive**, while red tokens pushed towards **Negative**."
            )
            
            if features:
                chips_html = ""
                for f in features:
                    cls = "token-chip-pos" if f["direction"] == "Positive" else "token-chip-neg"
                    arrow = "▲" if f["direction"] == "Positive" else "▼"
                    chips_html += f'<span class="{cls}">{f["word"]} ({arrow} {abs(f["impact"]):.2f})</span>'
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.info("ℹ️ None of the words in this review matched the 5,000 unigram vocabulary features of the trained IMDB model.")

            # Interpretation Guide
            with st.expander("📖 How to Interpret this Output"):
                st.markdown(r"""
                - **Predicted Sentiment:** The binary class with highest posterior probability ($P > 0.50$).
                - **Confidence:** Represents $\max(P(\text{Negative}), P(\text{Positive}))$. A confidence of 98% means the model's hyperplane strongly separates this review from the opposing class.
                - **Polarity Score:** Calculated as $P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$. Values near $+1.0$ indicate strong positive sentiment; values near $-1.0$ indicate strong negative sentiment; values near $0.0$ represent borderline/uncertain sentiment.
                - **Important Limitation:** Note how unigram models treat negations like *"not good"*. Because standard stopword filters remove *"not"*, the model only vectorizes *"good"*, demonstrating why n-gram or contextual embeddings are necessary for complete semantic understanding.
                """)


# =========================================================
# PAGE 2: Batch CSV Analysis & Dashboard
# =========================================================
elif selected_page == "📊 Batch CSV Analysis":
    st.markdown("### 📂 Batch Sentiment Analysis via CSV")
    st.write(
        "Upload a CSV file containing reviews or customer feedback to process hundreds of records simultaneously, "
        "visualize distribution metrics, and export the annotated dataset."
    )

    # Quick demo loader or file uploader
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file (Max 15 MB)",
            type=["csv"],
            help="Upload any CSV file containing a column of English text reviews."
        )
    with col_up2:
        st.write("")
        st.write("")
        load_demo = st.button("🧪 Load Sample Reviews", help="Load the built-in sample dataset containing 20 varied reviews.")

    # Determine and persist file bytes source in session_state
    if uploaded_file is not None:
        st.session_state["active_file_bytes"] = uploaded_file.getvalue()
        st.session_state["active_file_name"] = uploaded_file.name
    elif load_demo:
        sample_path = DATA_DIR / "sample_reviews.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.session_state["active_file_bytes"] = f.read()
            st.session_state["active_file_name"] = "sample_reviews.csv"

    file_bytes = st.session_state.get("active_file_bytes", None)
    file_name = st.session_state.get("active_file_name", "sample_reviews.csv")

    if file_bytes is not None:
        try:
            raw_df = load_csv_file(file_bytes, filename=file_name)
            st.success(f"✅ Successfully loaded **{file_name}** ({len(raw_df):,} total rows, {len(raw_df.columns)} columns).")

            # Preview raw data
            with st.expander("👁️ Preview Uploaded CSV (First 5 Rows)", expanded=False):
                st.dataframe(raw_df.head(5), use_container_width=True)

            # Column Selection and Configuration
            col_sel1, col_sel2 = st.columns([2, 1])
            text_candidates = detect_text_columns(raw_df)
            
            # Smart default column detection prioritizing content over author/user
            default_col_idx = 0
            # First pass: check for columns explicitly representing review content
            content_keywords = ["review_text", "text", "review", "comment", "feedback", "body", "content"]
            found = False
            for target_k in content_keywords:
                for idx, col in enumerate(text_candidates):
                    col_l = col.lower()
                    if col_l == target_k or (target_k in col_l and not any(avoid in col_l for avoid in ["name", "user", "id", "author", "date"])):
                        default_col_idx = idx
                        found = True
                        break
                if found:
                    break

            with col_sel1:
                selected_col = st.selectbox(
                    "Select the column containing review text:",
                    text_candidates,
                    index=default_col_idx
                )

            with col_sel2:
                row_cap = st.number_input(
                    "Process limit (rows):",
                    min_value=10,
                    max_value=MAX_BATCH_ROWS,
                    value=min(len(raw_df), 1000),
                    step=100,
                    help="Caps processing to preserve fast interactive rendering."
                )

            # Run Batch Processing
            if st.button("🚀 Run Batch Sentiment Analysis", type="primary"):
                with st.spinner(f"Analyzing up to {row_cap:,} records in '{selected_col}'..."):
                    result_df, metrics = process_batch_dataframe(
                        raw_df,
                        selected_col,
                        analyzer,
                        row_limit=int(row_cap)
                    )

                st.session_state["batch_result_df"] = result_df
                st.session_state["batch_metrics"] = metrics
                st.session_state["batch_text_col"] = selected_col

        except CSVProcessingError as e:
            st.error(f"❌ CSV Processing Error: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred while parsing the CSV: {e}")

    # Render Batch Results Dashboard if available in session
    if "batch_result_df" in st.session_state:
        res_df = st.session_state["batch_result_df"]
        m = st.session_state["batch_metrics"]
        text_col = st.session_state["batch_text_col"]

        st.markdown("---")
        st.markdown("### 📊 Batch Results Dashboard")

        # Metric summary cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Records", f"{m['total_records']:,}")
        with c2:
            st.metric("Successfully Analyzed", f"{m['analyzed_count']:,}")
        with c3:
            st.metric("Skipped / Blank", f"{m['skipped_count']:,}")
        with c4:
            st.metric("Positive Sentiment", f"{m['positive_count']:,}", f"{m['positive_pct']}%")
        with c5:
            st.metric("Negative Sentiment", f"{m['negative_count']:,}", f"{m['negative_pct']}%")

        if m["is_truncated"]:
            st.info(f"ℹ️ Note: Processing was capped at the configured limit of {m['row_limit']:,} rows.")

        # Interactive Visualizations
        st.markdown("#### 📈 Visual Distribution")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            # Donut Chart for Sentiment Breakdown
            dist_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [m["positive_count"], m["negative_count"]]
            })
            pie_fig = px.pie(
                dist_data,
                names="Sentiment",
                values="Count",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.45,
                title="Sentiment Class Proportion"
            )
            pie_fig.update_traces(textinfo="percent+label", pull=[0.05, 0.05])
            pie_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(pie_fig, use_container_width=True)

        with vcol2:
            # Histogram of Polarity Scores
            valid_rows = res_df[res_df["analysis_status"] == "Success"]
            hist_fig = px.histogram(
                valid_rows,
                x="sentiment_score",
                nbins=25,
                color="predicted_sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Polarity Score Distribution (-1.0 to +1.0)",
                labels={"sentiment_score": "Sentiment Polarity Score"}
            )
            hist_fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(range=[-1.05, 1.05])
            )
            st.plotly_chart(hist_fig, use_container_width=True)

        # Spotlight: Top Positive and Top Negative Records
        st.markdown("#### 🌟 Spotlight: Strongest Sentiment Polarities")
        spot_col1, spot_col2 = st.columns(2)

        with spot_col1:
            st.markdown("##### 🟢 Most Positive Reviews")
            if m["most_positive"]:
                for idx, r in enumerate(m["most_positive"], 1):
                    st.markdown(
                        f"""
                        <div style="background:#F0FDF4;border-left:4px solid #10B981;padding:10px 14px;border-radius:8px;margin-bottom:8px;font-size:0.88rem;">
                            <b>#{idx}</b>: <i>"{r['text']}..."</i><br>
                            <span style="color:#065F46;font-weight:600;">Score: {r['score']:+.4f} | Conf: {r['confidence']:.1%}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No positive records found.")

        with spot_col2:
            st.markdown("##### 🔴 Most Negative Reviews")
            if m["most_negative"]:
                for idx, r in enumerate(m["most_negative"], 1):
                    st.markdown(
                        f"""
                        <div style="background:#FEF2F2;border-left:4px solid #EF4444;padding:10px 14px;border-radius:8px;margin-bottom:8px;font-size:0.88rem;">
                            <b>#{idx}</b>: <i>"{r['text']}..."</i><br>
                            <span style="color:#991B1B;font-weight:600;">Score: {r['score']:+.4f} | Conf: {r['confidence']:.1%}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No negative records found.")

        # Interactive Filterable Data Table
        st.markdown("#### 📋 Processed Records Table")
        
        filter_col1, filter_col2 = st.columns([2, 2])
        with filter_col1:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment:",
                ["Positive", "Negative", "Skipped"],
                default=["Positive", "Negative", "Skipped"]
            )
        with filter_col2:
            search_query = st.text_input("🔍 Search within text column:", "")

        filtered_df = res_df[res_df["predicted_sentiment"].isin(sentiment_filter)]
        if search_query.strip():
            filtered_df = filtered_df[
                filtered_df[text_col].astype(str).str.contains(search_query, case=False, na=False)
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "sentiment_score": st.column_config.ProgressColumn(
                    "Polarity Score",
                    min_value=-1.0,
                    max_value=1.0,
                    format="%+.2f"
                ),
                "confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f"
                )
            }
        )
        st.caption(f"Showing **{len(filtered_df):,}** of **{len(res_df):,}** records.")

        # Export CSV Button
        csv_download_bytes = export_dataframe_to_csv(res_df)
        st.download_button(
            label="💾 Download Annotated CSV Results",
            data=csv_download_bytes,
            file_name=f"analyzed_{file_name}",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("👆 Upload a CSV file or click 'Load Sample Reviews' above to begin batch analysis.")


# =========================================================
# PAGE 3: Analytics & Insights
# =========================================================
elif selected_page == "📈 Analytics & Insights":
    st.markdown("### 📈 Sentiment Analytics & Model Behavior")
    st.write(
        "Explore how review length, confidence thresholds, and model properties interact across datasets."
    )

    if "batch_result_df" in st.session_state:
        df = st.session_state["batch_result_df"].copy()
        text_col = st.session_state["batch_text_col"]
        valid_df = df[df["analysis_status"] == "Success"].copy()

        # Add review length in characters and words
        valid_df["char_length"] = valid_df[text_col].astype(str).str.len()
        valid_df["word_count"] = valid_df[text_col].astype(str).str.split().str.len()

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            # Scatter: Review length vs Sentiment Score
            scatter_fig = px.scatter(
                valid_df,
                x="word_count",
                y="sentiment_score",
                color="predicted_sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                hover_data=[text_col],
                title="Review Word Count vs. Polarity Score",
                labels={"word_count": "Word Count", "sentiment_score": "Polarity Score"}
            )
            scatter_fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col_a2:
            # Boxplot of Confidence by Sentiment
            box_fig = px.box(
                valid_df,
                x="predicted_sentiment",
                y="confidence",
                color="predicted_sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Model Confidence Distribution by Class",
                labels={"predicted_sentiment": "Sentiment", "confidence": "Confidence"}
            )
            box_fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(box_fig, use_container_width=True)

        # High confidence vs Low confidence reviews
        st.markdown("#### 🎯 Confidence Distribution Insights")
        conf_bands = pd.cut(
            valid_df["confidence"],
            bins=[0.5, 0.7, 0.85, 1.0],
            labels=["Moderate (50-70%)", "High (70-85%)", "Very High (85-100%)"]
        ).value_counts().reset_index()
        conf_bands.columns = ["Confidence Band", "Record Count"]

        band_fig = px.bar(
            conf_bands,
            x="Confidence Band",
            y="Record Count",
            color="Confidence Band",
            title="Records by Confidence Tier",
            text="Record Count"
        )
        band_fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(band_fig, use_container_width=True)

    else:
        st.info("💡 To view active dataset analytics, run a batch analysis in the **📊 Batch CSV Analysis** tab first.")
        
        # Display general model vocabulary metrics
        st.markdown("#### 🧠 Model Architectural Attributes")
        st.write("Even without an uploaded CSV, you can examine the underlying model attributes below:")
        
        coefs = analyzer.coefficients
        names = analyzer.feature_names
        
        top_pos_idx = np.argsort(coefs)[-10:]
        top_neg_idx = np.argsort(coefs)[:10]

        top_pos_words = [(names[i], coefs[i]) for i in reversed(top_pos_idx)]
        top_neg_words = [(names[i], coefs[i]) for i in top_neg_idx]

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("##### 🟢 Top 10 Positive Model Weights")
            pos_weight_df = pd.DataFrame(top_pos_words, columns=["Vocabulary Word", "Weight"])
            fig_p = px.bar(
                pos_weight_df,
                x="Weight",
                y="Vocabulary Word",
                orientation="h",
                color_discrete_sequence=["#10B981"]
            )
            fig_p.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_p, use_container_width=True)

        with gcol2:
            st.markdown("##### 🔴 Top 10 Negative Model Weights")
            neg_weight_df = pd.DataFrame(top_neg_words, columns=["Vocabulary Word", "Weight"])
            fig_n = px.bar(
                neg_weight_df,
                x="Weight",
                y="Vocabulary Word",
                orientation="h",
                color_discrete_sequence=["#EF4444"]
            )
            fig_n.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_n, use_container_width=True)


# =========================================================
# PAGE 4: Methodology & Limitations
# =========================================================
elif selected_page == "🧠 Methodology & Limitations":
    st.markdown("### 🧠 Technical Methodology & Model Limitations")
    st.write(
        "Comprehensive documentation of the NLP pipeline, mathematical formulation, "
        "and linguistic failure modes for viva evaluation and peer review."
    )

    with st.expander("🔬 1. NLP Pipeline & Transformation Flow", expanded=True):
        st.markdown(r"""
        The SentimentScope architecture follows a standard supervised classification pipeline:
        
        1. **Text Normalization:**
           - HTML Entity Unescaping (`&amp;` $\rightarrow$ `&`)
           - Tag Removal (`<br />` stripped via regex)
           - URL Stripping (`https://...` removed)
           - Whitespace compression and case-folding to lowercase.
        2. **Feature Extraction (TF-IDF):**
           - **Term Frequency-Inverse Document Frequency** measures how important a word is to a document relative to the corpus.
           $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left( \ln \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} + 1 \right)$$
           - Vocabulary size: 5,000 unigrams.
        3. **Classification (Logistic Regression):**
           - Computes linear log-odds: $z = \beta_0 + \sum_{i=1}^M w_i x_i$
           - Sigmoidal transformation maps $z$ to a valid probability:
             $$P(Y = 1 | x) = \frac{1}{1 + e^{-z}}$$
        """)

    with st.expander("⚖️ 2. Polarity Score vs. Confidence vs. Probability", expanded=True):
        st.markdown(r"""
        It is vital to distinguish between classification probabilities and continuous polarity scores:
        - **Class Probability $P(\text{Positive})$:** Direct sigmoidal output between $0.0$ and $1.0$.
        - **Confidence $\max(P_0, P_1)$:** Measures how certain the classifier is in the winning class (always $\ge 0.50$).
        - **Polarity Score $(P_1 - P_0)$:** Rescales probabilities to the $[-1.0, +1.0]$ interval where $-1.0$ is maximum negative, $0.0$ is ambiguous/neutral, and $+1.0$ is maximum positive.
        """)

    with st.expander("⚠️ 3. Known Limitations & Edge Cases (Viva Talking Points)", expanded=True):
        st.markdown("""
        Sentiment analysis using bag-of-words or unigram TF-IDF models exhibits predictable linguistic blind spots:
        
        1. **Negation Blind Spot:**
           - Example: *"The movie was not good."*
           - *Why it fails:* The word `"not"` was included in Scikit-learn's default English stopword list during training, so it was filtered out! The remaining token `"good"` triggered a positive prediction.
           - *Solution:* Bigrams (`ngram_range=(1, 2)`) or retaining negation words in custom stopword lists.
        2. **Sarcasm and Irony:**
           - Example: *"Oh brilliant, another two hours wasted."*
           - *Why it fails:* The word `"brilliant"` carries high positive weight, and unigram models lack semantic context to detect ironic intent.
        3. **Context and Domain Transfer:**
           - A model trained on movie reviews (IMDB) may misinterpret financial, medical, or conversational slang.
        4. **Mixed Emotions:**
           - Sentences containing both praise and criticism (*"Great visuals, horrible script"*) will cancel out or lean toward whichever vocabulary frequency dominates.
        """)

    with st.expander("🎓 4. Rule-Based (VADER) vs. Machine Learning (Logistic Regression)", expanded=True):
        st.markdown("""
        | Dimension | Rule-Based / Lexicon (e.g. VADER) | Machine Learning (SentimentScope) |
        | :--- | :--- | :--- |
        | **Underlying Mechanism** | Manually curated sentiment dictionary with heuristic rules (punctuation, capitalization, negation flip). | Statistical weights learned from 50,000 labeled reviews via gradient optimization. |
        | **Training Requirement** | No training dataset needed. | Requires annotated training corpus (IMDB 50k). |
        | **Negation Handling** | Handles negation through look-back rules. | Requires n-grams or sub-word embeddings. |
        | **Domain Nuance** | Limited to lexicon terms. | Captures domain-specific vocabulary associations in movie reviews. |
        """)


# ---------------------------------------------------------
# Footer & Attribution (Requirement from Master Prompt)
# ---------------------------------------------------------
st.markdown("""
<div class="footer-container">
    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
        <div>
            <b>SentimentScope — An Interactive Sentiment Analysis Dashboard</b><br>
            Designed for College Viva Demonstration, NLP Education, and GitHub Publication.
        </div>
        <div style="text-align: right;">
            <b>Tech Stack:</b> Python 3.10 | Scikit-Learn | Streamlit | Pandas | Plotly | Pytest
        </div>
    </div>
    <div style="font-size: 0.8rem; color: #94A3B8;">
        <b>Attribution & Provenance:</b> Enhanced from original open-source base repository 
        <a href="https://github.com/sonalshinde24/sentiment-analysis-reviews" target="_blank" style="color: #0EA5E9; text-decoration: none;">
            sonalshinde24/sentiment-analysis-reviews
        </a> by Sonal Shinde (2025). 
        Upgraded with full batch CSV processing, Explainable AI (XAI) feature attribution, 
        robust data sanitization, comprehensive automated test suites, and modern UI analytics.
    </div>
</div>
""", unsafe_allow_html=True)
