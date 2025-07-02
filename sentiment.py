import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords (only needed once)
nltk.download('stopwords')

# Load dataset (make sure this CSV is in the same folder)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, nrows=100)

# Rename columns
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]

# Keep only Positive (4) and Negative (0) labels
df = df[df['target'] != 2]
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#", "", text)           # Remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", "", text)         # Keep only letters and spaces
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply cleaning to the text column
df['text'] = df['text'].apply(clean_text)

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Test the model with your own input
while True:
    user_input = input("\nEnter a sentence to analyze (or type 'exit'): ")
    if user_input.lower() == 'exit':
        print("Exiting Sentiment Analysis Tool.")
        break
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    print("Predicted Sentiment:", sentiment)
