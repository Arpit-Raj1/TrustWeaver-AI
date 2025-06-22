from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
from flasgger import Swagger


warnings.filterwarnings("ignore")

# Import the training functions from train.py
from train import train_model
from predict import predict as make_prediction

app = Flask(__name__)
swagger = Swagger(app)


# Global variables to store the trained model and vectorizer
model = None
vectorizer = None
feature_columns = [
    "customer_id",
    "review_id",
    "product_id",
    "product_parent",
    "product_title",
    "product_category",
    "star_rating",
    "helpful_votes",
    "total_votes",
    "vine",
    "verified_purchase",
    "review_headline",
    "review_text",
    "review_date",
]


def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer

    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("✅ Model and vectorizer loaded from disk")
        return True
    else:
        print("❌ Model files not found. Please train the model first.")
        return False


@app.route("/")
def home():
    """Home endpoint with API information"""
    return jsonify(
        {
            "message": "Review Fraud Detection API",
            "version": "1.0",
            "endpoints": {
                "/": "API information",
                "/train": "Train the model (POST)",
                "/predict": "Predict if a review is fake (POST)",
                "/health": "Health check",
                "/model-info": "Get model information",
            },
            "status": "running",
        }
    )


@app.route("/health")
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    vectorizer_status = "loaded" if vectorizer is not None else "not loaded"

    return jsonify(
        {
            "status": "healthy",
            "model_status": model_status,
            "vectorizer_status": vectorizer_status,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/train", methods=["POST"])
def train():
    """Train the model endpoint"""
    try:
        global model, vectorizer

        # Get the data file path from request
        data = request.get_json()
        file_path = data.get("file_path", "/content/Book1.csv")

        print(f"🚀 Starting model training with data from: {file_path}")

        # Train the model (this will use the functions from train.py)
        model, vectorizer, metrics = train_model(file_path)

        print("server: Model and vectorizer saved to disk")

        return jsonify(
            {
                "message": "Model trained successfully",
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Training failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/predict", methods=["POST"])
def predict():
    """Predict if a review is fake"""
    try:
        global model, vectorizer

        # Load model if not already loaded
        if model is None or vectorizer is None:
            if not load_model():
                return (
                    jsonify(
                        {
                            "error": "Model not available. Please train the model first.",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    400,
                )

        # Get review data from request
        data = request.get_json()

        result = make_prediction(data)

        return jsonify(result)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Prediction failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/model-info")
def model_info():
    """Get information about the current model"""
    try:
        model_loaded = model is not None
        vectorizer_loaded = vectorizer is not None

        info = {
            "model_loaded": model_loaded,
            "vectorizer_loaded": vectorizer_loaded,
            "model_type": type(model).__name__ if model_loaded else None,
            "feature_columns": feature_columns,
            "files_exist": {
                "model.pkl": os.path.exists("model.pkl"),
                "vectorizer.pkl": os.path.exists("vectorizer.pkl"),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(info)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Failed to get model info: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Predict multiple reviews at once"""
    try:
        global model, vectorizer

        # Load model if not already loaded
        if model is None or vectorizer is None:
            if not load_model():
                return (
                    jsonify(
                        {
                            "error": "Model not available. Please train the model first.",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    400,
                )

        # Get reviews data from request
        data = request.get_json()
        reviews = data.get("reviews", [])

        if not reviews:
            return (
                jsonify(
                    {
                        "error": "No reviews provided",
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                400,
            )

        results = []

        for i, review_data in enumerate(reviews):
            try:
                # Validate required fields
                if "review_text" not in review_data or "star_rating" not in review_data:
                    results.append(
                        {
                            "index": i,
                            "error": "Missing required fields: review_text, star_rating",
                        }
                    )
                    continue

                # Set default values for optional fields
                complete_review_data = {
                    "review_text": review_data["review_text"],
                    "star_rating": review_data["star_rating"],
                    "verified_purchase": review_data.get("verified_purchase", "N"),
                    "helpful_votes": review_data.get("helpful_votes", 0),
                    "total_votes": review_data.get("total_votes", 0),
                }

                # Extract features and make prediction
                numerical_features = extract_single_review_features(
                    complete_review_data
                )
                text_features = vectorizer.transform(
                    [complete_review_data["review_text"]]
                )

                from scipy.sparse import hstack

                combined_features = hstack([text_features, numerical_features.values])

                prediction = model.predict(combined_features)[0]
                prediction_proba = model.predict_proba(combined_features)[0]

                results.append(
                    {
                        "index": i,
                        "prediction": int(prediction),
                        "prediction_label": "Genuine" if prediction == 1 else "Fake",
                        "confidence": {
                            "fake": float(prediction_proba[0]),
                            "genuine": float(prediction_proba[1]),
                        },
                        "risk_score": float(1 - prediction_proba[1]),
                    }
                )

            except Exception as e:
                results.append({"index": i, "error": f"Processing failed: {str(e)}"})

        return jsonify(
            {
                "results": results,
                "total_processed": len(reviews),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Batch prediction failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


# Helper function to extract features from a single review
def extract_single_review_features(review_data):
    """
    Extracts numerical features from a single review dictionary.
    Returns a pandas DataFrame with one row.
    """
    review_text = review_data.get("review_text", "")
    star_rating = float(review_data.get("star_rating", 0))
    verified_purchase = review_data.get("verified_purchase", "N")
    helpful_votes = int(review_data.get("helpful_votes", 0))
    total_votes = int(review_data.get("total_votes", 0))

    review_length = len(review_text)
    word_count = len(review_text.split())
    exclamation_count = review_text.count("!")
    question_count = review_text.count("?")
    caps_ratio = (
        sum(1 for c in review_text if c.isupper()) / len(review_text)
        if len(review_text) > 0
        else 0
    )

    # Example positive/negative word lists (customize as needed)
    positive_words_list = ["good", "great", "excellent", "amazing", "love", "wonderful"]
    negative_words_list = ["bad", "terrible", "awful", "hate", "poor", "worst"]

    positive_words = sum(word in review_text.lower() for word in positive_words_list)
    negative_words = sum(word in review_text.lower() for word in negative_words_list)

    # Rating-text mismatch: 1 if rating is high but text is negative, or vice versa
    rating_text_mismatch = int(
        (star_rating >= 4 and negative_words > positive_words)
        or (star_rating <= 2 and positive_words > negative_words)
    )

    is_verified = 1 if str(verified_purchase).upper() == "Y" else 0

    helpfulness_ratio = helpful_votes / total_votes if total_votes > 0 else 0
    has_helpful_votes = 1 if helpful_votes > 0 else 0

    # For single review, duplicate_review is always 0 (cannot check in isolation)
    duplicate_review = 0

    features = {
        "star_rating": star_rating,
        "review_length": review_length,
        "word_count": word_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_ratio": caps_ratio,
        "positive_words": positive_words,
        "negative_words": negative_words,
        "rating_text_mismatch": rating_text_mismatch,
        "is_verified": is_verified,
        "helpfulness_ratio": helpfulness_ratio,
        "has_helpful_votes": has_helpful_votes,
        "duplicate_review": duplicate_review,
    }

    return pd.DataFrame([features])


if __name__ == "__main__":
    print("🚀 Starting Review Fraud Detection API...")
    print("📝 Available endpoints:")
    print("   • GET  /          - API information")
    print("   • GET  /health    - Health check")
    print("   • POST /train     - Train the model")
    print("   • POST /predict   - Predict single review")
    print("   • POST /batch-predict - Predict multiple reviews")
    print("   • GET  /model-info - Model information")

    # Try to load existing model
    load_model()

    app.run(debug=True, host="0.0.0.0", port=5000)
