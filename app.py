import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠")
st.title("🧠 Sentiment Analysis App")
st.markdown("Enter a sentence below to find out if the sentiment is **Positive** or **Negative**.")

# Text input
text_input = st.text_area("💬 Your Text Here:", height=100)

if st.button("🔍 Predict Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = text_input.lower()
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success("✅ Positive Sentiment Detected! 😊")
        else:
            st.error("❌ Negative Sentiment Detected! 😞")
