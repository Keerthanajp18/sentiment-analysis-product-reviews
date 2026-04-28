import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# Load models
model = pickle.load(open("svm_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# ─── Header ────────────────────────────────────────────────────
st.title("💬 Product Review Sentiment Analyzer")
st.markdown("Classify customer reviews as **Positive**, **Negative**, or **Neutral** using a trained SVM model.")
st.divider()

# ─── Single Review Tab ─────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Review", "📋 Batch Reviews"])

with tab1:
    st.subheader("Analyze a Single Review")
    review_input = st.text_area(
        "Enter your product review below:",
        placeholder="e.g. This product is amazing! Works perfectly and arrived on time.",
        height=150
    )

    if st.button("Predict Sentiment", type="primary", use_container_width=True):
        if review_input.strip() == "":
            st.warning("⚠️ Please enter a review before predicting.")
        else:
            sentiment = predict_sentiment(review_input)
            sentiment_lower = str(sentiment).lower()

            if "positive" in sentiment_lower or sentiment == 1 or sentiment == "1":
                st.success("✅ Sentiment: **POSITIVE**")
                st.balloons()
            elif "negative" in sentiment_lower or sentiment == 0 or sentiment == "0":
                st.error("❌ Sentiment: **NEGATIVE**")
            else:
                st.info(f"💡 Sentiment: **{str(sentiment).upper()}**")

            with st.expander("🔎 See cleaned text"):
                st.code(clean_text(review_input))

# ─── Batch Review Tab ──────────────────────────────────────────
with tab2:
    st.subheader("Analyze Multiple Reviews")
    st.markdown("Enter one review per line:")

    batch_input = st.text_area(
        "Paste multiple reviews (one per line):",
        placeholder="Great product!\nTerrible quality, waste of money.\nIt's okay, nothing special.",
        height=200
    )

    if st.button("Predict All", type="primary", use_container_width=True):
        if batch_input.strip() == "":
            st.warning("⚠️ Please enter at least one review.")
        else:
            reviews = [r.strip() for r in batch_input.strip().split("\n") if r.strip()]
            results = []
            for r in reviews:
                pred = predict_sentiment(r)
                pred_str = str(pred).lower()
                if "positive" in pred_str or pred in [1, "1"]:
                    label = "✅ Positive"
                elif "negative" in pred_str or pred in [0, "0"]:
                    label = "❌ Negative"
                else:
                    label = f"💡 {str(pred).capitalize()}"
                results.append({"Review": r, "Sentiment": label})

            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            pos = sum(1 for r in results if "Positive" in r["Sentiment"])
            neg = sum(1 for r in results if "Negative" in r["Sentiment"])
            other = len(results) - pos - neg

            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Positive", pos)
            col2.metric("❌ Negative", neg)
            col3.metric("💡 Other", other)

# ─── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ using Streamlit | SVM + TF-IDF Model | Deployment by Fidal")
