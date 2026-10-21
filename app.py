"""
SentimentScope - An Interactive Sentiment Analysis Dashboard
Streamlined, focused, and clean UI for sentiment evaluation.
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
from src.preprocessing import clean_text, validate_text
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
    page_title="SentimentScope",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Modern, Restrained Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* App Header */
    .app-header {
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E2E8F0;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Result Card */
    .sentiment-card {
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .sentiment-positive {
        background: #F0FDF4;
        border-left: 5px solid #10B981;
    }
    .sentiment-negative {
        background: #FEF2F2;
        border-left: 5px solid #EF4444;
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-score {
        font-size: 0.92rem;
        color: #475569;
    }

    /* Token Chips */
    .token-chip-pos {
        display: inline-block;
        background: #D1FAE5;
        color: #065F46;
        padding: 3px 9px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
        border: 1px solid #A7F3D0;
    }
    .token-chip-neg {
        display: inline-block;
        background: #FEE2E2;
        color: #991B1B;
        padding: 3px 9px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
        border: 1px solid #FECACA;
    }

    /* Clean Footer */
    .clean-footer {
        margin-top: 40px;
        padding-top: 14px;
        border-top: 1px solid #E2E8F0;
        color: #94A3B8;
        font-size: 0.82rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Model Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_engine() -> SentimentAnalyzer:
    """Load and cache SentimentAnalyzer instance."""
    return SentimentAnalyzer()


try:
    analyzer = load_sentiment_engine()
except ModelLoadError as e:
    st.error(f"⚠️ Model Loading Error: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 SentimentScope")
    
    selected_page = st.radio(
        "Menu",
        [
            "📝 Single Text Analysis",
            "📊 Batch CSV Analysis",
            "📈 Analytics & Insights",
            "🧠 Methodology & Limitations"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    sample_csv_path = DATA_DIR / "sample_reviews.csv"
    if sample_csv_path.exists():
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample CSV",
                data=f.read(),
                file_name="sample_reviews.csv",
                mime="text/csv",
                use_container_width=True
            )


# ---------------------------------------------------------
# Minimal App Header
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🎯 SentimentScope</div>
    <p class="app-subtitle">Interactive Sentiment Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: Single Text Sentiment Analysis
# =========================================================
if selected_page == "📝 Single Text Analysis":
    # Example presets selector
    selected_example_label = st.selectbox(
        "Load Example Review:",
        ["(Select an example...)"] + list(EXAMPLE_SENTENCES.keys()),
        index=0
    )

    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    if selected_example_label != "(Select an example...)":
        st.session_state.user_text = EXAMPLE_SENTENCES[selected_example_label]

    # Review text input
    user_input = st.text_area(
        "Review Text:",
        value=st.session_state.user_text,
        height=130,
        placeholder="Enter text to analyze...",
        max_chars=MAX_TEXT_LENGTH
    )

    col_btn1, col_btn2, _ = st.columns([1.5, 1, 5])
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state.user_text = ""
            st.rerun()

    # Inference Execution
    if analyze_clicked:
        is_valid, validation_msg = validate_text(user_input)
        
        if not is_valid:
            st.warning(f"⚠️ {validation_msg}")
        else:
            with st.spinner("Analyzing..."):
                result = analyzer.predict(user_input)

            sentiment = result["sentiment"]
            conf = result["confidence"]
            score = result["sentiment_score"]
            neg_p = result["probabilities"]["Negative"]
            pos_p = result["probabilities"]["Positive"]
            features = result["top_features"]

            card_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
            icon = "😀" if sentiment == "Positive" else "😞"
            
            st.markdown(f"""
            <div class="sentiment-card {card_class}">
                <div class="card-title">{icon} {sentiment} Sentiment</div>
                <div class="card-score">
                    Confidence: <b>{conf:.1%}</b> &nbsp;|&nbsp; 
                    Polarity Score: <b>{score:+.4f}</b> (Scale: -1.0 to +1.0)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Core Metric Pillars
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Primary Sentiment", sentiment, f"{'+' if sentiment == 'Positive' else '-'}{conf:.1%}")
            with col_m2:
                st.metric("Positive Probability", f"{pos_p:.1%}")
            with col_m3:
                st.metric("Negative Probability", f"{neg_p:.1%}")
            with col_m4:
                st.metric("Polarity Index", f"{score:+.3f}")

            # Polarity Gauge & Probability Breakdown
            chart_col1, chart_col2 = st.columns([1.2, 1.8])

            with chart_col1:
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Polarity Score", 'font': {'size': 15}},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': SENTIMENT_COLORS[sentiment], 'thickness': 0.3},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#CBD5E1",
                        'steps': [
                            {'range': [-1, -0.1], 'color': 'rgba(239, 68, 68, 0.12)'},
                            {'range': [-0.1, 0.1], 'color': 'rgba(107, 114, 128, 0.08)'},
                            {'range': [0.1, 1], 'color': 'rgba(16, 185, 129, 0.12)'}
                        ],
                        'threshold': {'line': {'color': "#0F172A", 'width': 3}, 'value': score}
                    }
                ))
                gauge_fig.update_layout(height=210, margin=dict(l=20, r=20, t=35, b=10))
                st.plotly_chart(gauge_fig, use_container_width=True)

            with chart_col2:
                prob_df = pd.DataFrame({
                    "Sentiment": ["Negative", "Positive"],
                    "Probability": [neg_p, pos_p]
                })
                bar_fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Sentiment",
                    orientation="h",
                    text=[f"{neg_p:.1%}", f"{pos_p:.1%}"],
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    title="Class Probabilities"
                )
                bar_fig.update_layout(
                    height=210,
                    margin=dict(l=10, r=20, t=35, b=10),
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    showlegend=False
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            # Influential Terms
            if features:
                st.markdown("##### Contributing Keywords")
                chips_html = ""
                for f in features:
                    cls = "token-chip-pos" if f["direction"] == "Positive" else "token-chip-neg"
                    arrow = "▲" if f["direction"] == "Positive" else "▼"
                    chips_html += f'<span class="{cls}">{f["word"]} ({arrow} {abs(f["impact"]):.2f})</span>'
                st.markdown(chips_html, unsafe_allow_html=True)


# =========================================================
# PAGE 2: Batch CSV Analysis & Dashboard
# =========================================================
elif selected_page == "📊 Batch CSV Analysis":
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])
    with col_up2:
        st.write("")
        st.write("")
        load_demo = st.button("🧪 Load Sample Reviews", use_container_width=True)

    # Persist file bytes across reruns
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
            st.success(f"Successfully loaded **{file_name}** ({len(raw_df):,} rows).")

            text_candidates = detect_text_columns(raw_df)
            default_col_idx = 0
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

            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                selected_col = st.selectbox("Text Column:", text_candidates, index=default_col_idx)
            with col_sel2:
                row_cap = st.number_input("Max Rows to Process:", min_value=10, max_value=MAX_BATCH_ROWS, value=min(len(raw_df), 1000), step=100)

            if st.button("🚀 Run Batch Sentiment Analysis", type="primary"):
                with st.spinner("Processing batch..."):
                    result_df, metrics = process_batch_dataframe(raw_df, selected_col, analyzer, row_limit=int(row_cap))

                st.session_state["batch_result_df"] = result_df
                st.session_state["batch_metrics"] = metrics
                st.session_state["batch_text_col"] = selected_col

        except CSVProcessingError as e:
            st.error(f"CSV Error: {e}")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

    # Display Dashboard Results
    if "batch_result_df" in st.session_state:
        res_df = st.session_state["batch_result_df"]
        m = st.session_state["batch_metrics"]
        text_col = st.session_state["batch_text_col"]

        st.markdown("---")
        
        # Summary KPI Cards
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

        # Visual Breakdown Charts
        vcol1, vcol2 = st.columns(2)
        with vcol1:
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
            pie_fig.update_traces(textinfo="percent+label")
            pie_fig.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(pie_fig, use_container_width=True)

        with vcol2:
            valid_rows = res_df[res_df["analysis_status"] == "Success"]
            hist_fig = px.histogram(
                valid_rows,
                x="sentiment_score",
                nbins=20,
                color="predicted_sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Polarity Score Distribution",
                labels={"sentiment_score": "Score"}
            )
            hist_fig.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10), xaxis=dict(range=[-1.05, 1.05]))
            st.plotly_chart(hist_fig, use_container_width=True)

        # Spotlight Records
        spot_col1, spot_col2 = st.columns(2)
        with spot_col1:
            st.markdown("##### 🟢 Most Positive")
            for idx, r in enumerate(m["most_positive"][:2], 1):
                st.markdown(
                    f"""<div style="background:#F0FDF4;border-left:3px solid #10B981;padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:0.85rem;">
                    <b>#{idx}</b>: <i>"{r['text']}..."</i> (Score: <b>{r['score']:+.2f}</b>, Conf: <b>{r['confidence']:.1%}</b>)
                    </div>""",
                    unsafe_allow_html=True
                )

        with spot_col2:
            st.markdown("##### 🔴 Most Negative")
            for idx, r in enumerate(m["most_negative"][:2], 1):
                st.markdown(
                    f"""<div style="background:#FEF2F2;border-left:3px solid #EF4444;padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:0.85rem;">
                    <b>#{idx}</b>: <i>"{r['text']}..."</i> (Score: <b>{r['score']:+.2f}</b>, Conf: <b>{r['confidence']:.1%}</b>)
                    </div>""",
                    unsafe_allow_html=True
                )

        # Filterable Results Table
        st.markdown("##### Analyzed Records")
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            sentiment_filter = st.multiselect("Filter:", ["Positive", "Negative", "Skipped"], default=["Positive", "Negative", "Skipped"], label_visibility="collapsed")
        with filter_col2:
            search_query = st.text_input("Search:", placeholder="Filter by keyword...", label_visibility="collapsed")

        filtered_df = res_df[res_df["predicted_sentiment"].isin(sentiment_filter)]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df[text_col].astype(str).str.contains(search_query, case=False, na=False)]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "sentiment_score": st.column_config.ProgressColumn("Polarity Score", min_value=-1.0, max_value=1.0, format="%+.2f"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.2f")
            }
        )

        csv_download_bytes = export_dataframe_to_csv(res_df)
        st.download_button(
            label="💾 Download Results (CSV)",
            data=csv_download_bytes,
            file_name=f"analyzed_{file_name}",
            mime="text/csv",
            type="primary"
        )


# =========================================================
# PAGE 3: Analytics & Insights
# =========================================================
elif selected_page == "📈 Analytics & Insights":
    if "batch_result_df" in st.session_state:
        df = st.session_state["batch_result_df"].copy()
        text_col = st.session_state["batch_text_col"]
        valid_df = df[df["analysis_status"] == "Success"].copy()

        valid_df["word_count"] = valid_df[text_col].astype(str).str.split().str.len()

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            scatter_fig = px.scatter(
                valid_df,
                x="word_count",
                y="sentiment_score",
                color="predicted_sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                hover_data=[text_col],
                title="Word Count vs. Polarity Score"
            )
            scatter_fig.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col_a2:
            box_fig = px.box(
                valid_df,
                x="predicted_sentiment",
                y="confidence",
                color="predicted_sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Confidence by Class"
            )
            box_fig.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(box_fig, use_container_width=True)
    else:
        # Show top model weights if no batch run yet
        coefs = analyzer.coefficients
        names = analyzer.feature_names
        top_pos_idx = np.argsort(coefs)[-8:]
        top_neg_idx = np.argsort(coefs)[:8]

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            pos_weight_df = pd.DataFrame([(names[i], coefs[i]) for i in reversed(top_pos_idx)], columns=["Word", "Weight"])
            fig_p = px.bar(pos_weight_df, x="Weight", y="Word", orientation="h", color_discrete_sequence=["#10B981"], title="Top Positive Weights")
            fig_p.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_p, use_container_width=True)

        with gcol2:
            neg_weight_df = pd.DataFrame([(names[i], coefs[i]) for i in top_neg_idx], columns=["Word", "Weight"])
            fig_n = px.bar(neg_weight_df, x="Weight", y="Word", orientation="h", color_discrete_sequence=["#EF4444"], title="Top Negative Weights")
            fig_n.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_n, use_container_width=True)


# =========================================================
# PAGE 4: Methodology & Limitations
# =========================================================
elif selected_page == "🧠 Methodology & Limitations":
    with st.expander("1. NLP Pipeline Flow", expanded=True):
        st.markdown(r"""
        - **Preprocessing:** HTML stripping, URL removal, lowercasing, whitespace trimming.
        - **Vectorization:** TF-IDF with 5,000 unigram vocabulary.
        - **Classification:** Logistic Regression with Sigmoid posterior probability $P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-z}}$.
        """)

    with st.expander("2. Known Limitations & Viva Notes", expanded=True):
        st.markdown("""
        - **Negation Limitation:** *"not good"* predicts Positive because default English stopwords filter out *"not"*, leaving only *"good"*.
        - **Sarcasm:** Bag-of-words lacks context to detect irony or sarcasm.
        - **Domain Context:** Trained on movie reviews; domain-specific slang in other fields may vary.
        """)

    with st.expander("3. Rule-Based (VADER) vs. ML (SentimentScope)"):
        st.markdown("""
        - **VADER:** Lexicon and heuristic rules; no training dataset required.
        - **Logistic Regression:** Statistical weights trained on 50,000 labeled IMDB reviews.
        """)


# ---------------------------------------------------------
# Clean, Minimal Footer
# ---------------------------------------------------------
st.markdown('<div class="clean-footer">SentimentScope • NLP Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
