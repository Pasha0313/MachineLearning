# ============================================
# NLP Sentiment Analysis with sklearn Pipeline
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
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

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
from sklearn.pipeline import Pipeline

import joblib


# 2. Load Dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)
y = dataset.iloc[:, -1].values

print("Dataset preview:")
print(dataset.head())
print("\nClass distribution (overall):")
print(pd.Series(y).value_counts(normalize=True))


# 3. Text Cleaning (Stemming & Lemmatization)
def clean_text(text, mode="lemma"):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    stop_words = set(stopwords.words("english"))
    stop_words.remove("not")  # keep "not" for sentiment

    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    if mode == "stem":
        words = [ps.stem(w) for w in text if w not in stop_words]
    else:  # default to lemmatization
        words = [lemmatizer.lemmatize(w) for w in text if w not in stop_words]

    return " ".join(words)


# Choose stemming or lemmatization
dataset["Cleaned"] = dataset["Review"].apply(lambda x: clean_text(x, mode="lemma"))

print("\nSample cleaned review:", dataset["Cleaned"].iloc[0])


# 4. Train/Test Split (with stratify)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    dataset["Cleaned"],
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

print("\nClass distribution (train):")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nClass distribution (test):")
print(pd.Series(y_test).value_counts(normalize=True))


# 5. Build Pipelines (BoW or TF-IDF)
pipelines = {
    "Naive Bayes (BoW)": Pipeline([
        ("vectorizer", CountVectorizer(max_features=1500)),
        ("clf", MultinomialNB())
    ]),
    "Logistic Regression (TF-IDF)": Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
    ]),
    "Random Forest (TF-IDF)": Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED))
    ]),
}


# 6. Train & Evaluate
results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    results[name] = acc


# 7. Cross-validation (Logistic Regression + TF-IDF)
clf = Pipeline([
    ("vectorizer", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
])
scores = cross_val_score(clf, dataset["Cleaned"], y, cv=5, scoring="f1_macro")
print("\nCross-validated F1 (Logistic Regression + TF-IDF):", scores.mean())


# 8. Save Best Model
best_model_name = max(results, key=results.get)
best_model = pipelines[best_model_name]
joblib.dump(best_model, "sentiment_model.pkl")
print(f"\nBest model saved: {best_model_name}")
