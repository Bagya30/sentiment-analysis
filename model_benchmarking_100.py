import pandas as pd
import re, nltk, joblib, seaborn as sns, matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# -------- 1. LOAD a 100-row balanced sample --------
CSV = 'training.1600000.processed.noemoticon.csv'   # make sure path is correct
df_full = pd.read_csv(CSV, encoding='latin-1', header=None,
                      names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# keep only pos(4) & neg(0)
df_full = df_full[df_full['target'] != 2]
df_full['target'] = df_full['target'].apply(lambda x: 1 if x == 4 else 0)

# pick 50 pos + 50 neg
pos = df_full[df_full['target'] == 1].head(50)
neg = df_full[df_full['target'] == 0].head(50)
df = pd.concat([pos, neg]).sample(frac=1, random_state=42)   # shuffle

print(f"Loaded {len(df)} rows (Balanced 50/50)")

# -------- 2. CLEAN TEXT --------
nltk.download('stopwords')
sw = set(stopwords.words('english'))

def clean(t):
    t = t.lower()
    t = re.sub(r'http\S+|www\.\S+', '', t)
    t = re.sub(r'@\w+|#', '', t)
    t = re.sub(r'[^a-z\s]', '', t)
    return ' '.join(w for w in t.split() if w not in sw)

df['clean'] = df['text'].apply(clean)

# -------- 3. VECTORISE --------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean'])
y = df['target']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- 4. TRAIN & EVALUATE --------
models = {
    "Naive Bayes": MultinomialNB(),
    "Log Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

best_name, best_f1 = None, 0

for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)
    print(f"\n--- {name} ---")
    print(classification_report(y_te, y_pred, digits=3))
    
    # Confusion matrix plot
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} – Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.show()
    
    f1 = cm.trace()/cm.sum()          # simple accuracy as proxy
    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, mdl

print(f"\nBest model on 100-row sample: {best_name} (accuracy ≈ {best_f1:.2%})")

# -------- 5. SAVE BEST MODEL & VECTORIZER --------
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(tfidf,       'tfidf.pkl')
print("✔ Saved best_model.pkl and tfidf.pkl")
