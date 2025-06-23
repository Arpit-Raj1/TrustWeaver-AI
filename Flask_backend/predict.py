import pandas as pd
import numpy as np
import os
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# from sklearn.base import accuracy_score
from sklearn.metrics import classification_report

from train import extract_features_from_book1


def predict(data):
    # Load test data
    test_df = pd.read_csv("Book1.csv")  # replace with your actual test file

    # Preprocess features (same as in training)
    df_processed = extract_features_from_book1(test_df)

    # Numerical features (same order used in training)
    feature_columns = [
        "star_rating",
        "review_length",
        "word_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "positive_words",
        "negative_words",
        "rating_text_mismatch",
        "is_verified",
        "helpfulness_ratio",
        "has_helpful_votes",
        "duplicate_review",
    ]
    X_numerical = df_processed[feature_columns].fillna(0)

    # Load saved model and vectorizer
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Combine features
    from scipy.sparse import hstack

    df_processed["review_text"] = df_processed["review_text"].fillna("")
    X_text = vectorizer.transform(df_processed["review_text"])
    X_combined = hstack([X_text, X_numerical.values])
    print("🧪 model expects:", model.n_features_in_)
    print("🧪 test shape:", X_combined.shape[1])

    # True labels
    y_true = df_processed["label"]

    # Predict
    y_pred = model.predict(X_combined)

    # # Evaluate
    # print("📊 Classification Report:")
    # print(classification_report(y_true, y_pred))
    # print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    return y_pred
