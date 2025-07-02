# 🧠 Real-Time Sentiment Analysis (NB vs LR vs SVM)

![Made with Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-red)

A mini NLP pipeline that:

| Layer | What it Does |
|-------|--------------|
| **EDA** | WordClouds, sentiment distribution, top-10 words |
| **Model Benchmarking** | Naive Bayes · Logistic Regression · SVM on TF-IDF |
| **Web UI** | Streamlit app for real-time sentiment prediction |

---

## 🚀 Quick Start

```bash
# 1) clone & install
git clone https://github.com/<your-handle>/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt

# 2) run the app
streamlit run app.py
