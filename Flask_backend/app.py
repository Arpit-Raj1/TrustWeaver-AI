from flask import Flask, request, jsonify
from flask_cors import CORS
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
from predict import predict as make_prediction, predict_batch, extract_single_review_features

app = Flask(__name__)
# Enable CORS for all routes - comprehensive origins list
CORS(app, origins=[
    "http://localhost:3000", 
    "http://localhost:5173", 
    "http://localhost:8080",
    "http://127.0.0.1:3000", 
    "http://127.0.0.1:5173", 
    "http://127.0.0.1:8080",
    "http://0.0.0.0:8080",
    "http://[::]:8080"
])
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
        global model, vectorizer        # Get the data file path from request
        data = request.get_json()
        file_path = data.get("file_path", "D:/AA/TrustWeaver-AI/Book1.csv")

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

        # Use the batch prediction function from predict.py
        results = predict_batch(reviews)

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
