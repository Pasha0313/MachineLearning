# ============================================
# NLP Sentiment Analysis Pipeline (Complete)
# Steps 1 - 7
# ============================================

# 1. Import Libraries & Set Seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

import joblib
from collections import Counter


# 2. Load Dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

print("Dataset preview:")
print(dataset.head())
print("\nClass distribution:")
print(dataset.iloc[:, -1].value_counts(normalize=True))


# 3. Text Cleaning
corpus = []
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
stop_words.remove("not")

for i in range(len(dataset)):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    corpus.append(" ".join(review))

print("\nSample cleaned review:", corpus[0])

# Top word frequencies
all_words = " ".join(corpus).split()
word_freq = Counter(all_words)
print("\nMost common words:", word_freq.most_common(10))


# 4. Feature Extraction
cv = CountVectorizer(max_features=1500)
X_bow = cv.fit_transform(corpus).toarray()

tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(corpus).toarray()

y = dataset.iloc[:, -1].values


# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=SEED
)


# 6. Train Models & Evaluate
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    results[name] = acc


# Cross-validation with Logistic Regression
clf = LogisticRegression(max_iter=1000, random_state=SEED)
scores = cross_val_score(clf, X_tfidf, y, cv=5, scoring="f1_macro")
print("\nCross-validated F1 (Logistic Regression):", scores.mean())


# 7. Save Best Model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump((best_model, tfidf), "sentiment_model.pkl")
print(f"\nBest model saved: {best_model_name}")
