# eda_visuals.py  –  FINAL PATCHED VERSION
# ---------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re, nltk
from nltk.corpus import stopwords
from collections import Counter

# ---------- 0. Setup ----------
nltk.download('stopwords')
custom_stopwords = set(stopwords.words('english'))

# ---------- 1. Load *full* file then RANDOM-sample 5 000 rows ----------
df_full = pd.read_csv(
            "training.1600000.processed.noemoticon.csv",
            encoding='latin-1',
            header=None,
            usecols=[0, 5]          # load only label + tweet text columns
)
df_full = df_full[df_full[0] != 2]               # drop neutral
df_full[0] = df_full[0].apply(lambda x: 1 if x == 4 else 0)

# random sample for balanced visuals
df = df_full.sample(n=5000, random_state=42).reset_index(drop=True)
df.columns = ['target', 'text']

print("Rows sampled:", len(df))
print("Positive tweets:", (df['target'] == 1).sum())
print("Negative tweets:", (df['target'] == 0).sum())

# ---------- 2. Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in custom_stopwords)

df['cleaned'] = df['text'].apply(clean_text)

# ---------- 3. Sentiment distribution bar chart ----------
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='target', hue='target', legend=False, palette='Set2')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.title('Sentiment Distribution')
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# ---------- 4. WordCloud helper ----------
def draw_wordcloud(text_str, title, filename, stop_words=None):
    if text_str.strip():
        wc = WordCloud(width=800, height=400,
                       background_color='white',
                       stopwords=stop_words).generate(text_str)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    else:
        print(f"⚠️  Skipped {title}: no words available.")

# Positive WordCloud (raw tweets, default stop-words only)
raw_pos_text = " ".join(df[df['target'] == 1]['text'].str.lower())
draw_wordcloud(raw_pos_text,
               "Positive Sentiment Words",
               "wordcloud_pos.png",
               stop_words=STOPWORDS)

# Negative WordCloud
raw_neg_text = " ".join(df[df['target'] == 0]['text'].str.lower())
draw_wordcloud(raw_neg_text,
               "Negative Sentiment Words",
               "wordcloud_neg.png",
               stop_words=STOPWORDS)

# ---------- 5. Top-10 frequent words (from cleaned text) ----------
all_words = " ".join(df['cleaned']).split()
word_freq = Counter(all_words)
top10 = word_freq.most_common(10)
words, counts = zip(*top10)

plt.figure(figsize=(8, 4))
sns.barplot(x=list(words), y=list(counts),
            color=sns.color_palette("Blues_r")[2])   # single colour → no warning
plt.title("Top 10 Frequent Words")
plt.ylabel("Count")
plt.xlabel("Word")
plt.tight_layout()
plt.savefig("top10_words.png")
plt.show()
