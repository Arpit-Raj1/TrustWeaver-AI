import pandas as pd
import numpy as np
import os
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.metrics import classification_report

from train import extract_features_from_book1


def extract_6_key_features(review_data):
    """
    Extract only 6 key numerical features that the model expects.
    Returns a pandas DataFrame with one row and 6 columns.
    """
    review_text = review_data.get('review_text', '')
    star_rating = float(review_data.get('star_rating', 0))
    verified_purchase = review_data.get('verified_purchase', 'N')
    helpful_votes = int(review_data.get('helpful_votes', 0))
    total_votes = int(review_data.get('total_votes', 0))
    
    # Calculate the 6 most important features
    review_length = len(review_text)
    word_count = len(review_text.split())
    exclamation_count = review_text.count('!')
    is_verified = 1 if str(verified_purchase).upper() == 'Y' else 0
    helpfulness_ratio = helpful_votes / total_votes if total_votes > 0 else 0
    
    # Simple sentiment indicator
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'poor', 'worst']
    review_lower = review_text.lower()
    sentiment_score = sum(1 for word in positive_words if word in review_lower) - sum(1 for word in negative_words if word in review_lower)
    
    # Create feature dictionary with exactly 6 features
    features = {
        'star_rating': star_rating,
        'review_length': review_length,
        'word_count': word_count,
        'exclamation_count': exclamation_count,
        'is_verified': is_verified,
        'sentiment_score': sentiment_score
    }
    
    return pd.DataFrame([features])


def predict(data):
    """
    Predict if a single review is fake or genuine
    
    Args:
        data: Dictionary containing review data with required fields:
              - review_text: The text content of the review
              - star_rating: Rating from 1-5
              - verified_purchase: 'Y' or 'N' (optional, defaults to 'N')
              - helpful_votes: Number of helpful votes (optional, defaults to 0)
              - total_votes: Total votes (optional, defaults to 0)
    
    Returns:
        Dictionary with prediction results
    """
    
    # Validate input data
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary")
    
    if 'review_text' not in data:
        raise ValueError("Missing required field: review_text")
    
    if 'star_rating' not in data:
        raise ValueError("Missing required field: star_rating")
    
    # Extract review data with defaults for optional fields
    review_text = data.get('review_text', '')
    star_rating = float(data.get('star_rating', 0))
    verified_purchase = data.get('verified_purchase', 'N')
    helpful_votes = int(data.get('helpful_votes', 0))
    total_votes = int(data.get('total_votes', 0))
      # Load saved model (no need for vectorizer with 6 features)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Extract only 6 key features that the model expects
    features_6 = extract_6_key_features({
        'review_text': review_text,
        'star_rating': star_rating,
        'verified_purchase': verified_purchase,
        'helpful_votes': helpful_votes,
        'total_votes': total_votes
    })
    
    print(f"Debug: Using 6 features: {features_6.shape}")
    print(f"Debug: Features: {list(features_6.columns)}")
    
    # Use only the 6 features (no text features, no vectorizer needed)
    combined_features = features_6.values
    
    # Make prediction
    prediction = model.predict(combined_features)[0]
    prediction_proba = model.predict_proba(combined_features)[0]
    
    # Prepare result
    result = {
        "prediction": int(prediction),
        "prediction_label": "Genuine" if prediction == 1 else "Fake",
        "confidence": {
            "fake": float(prediction_proba[0]),
            "genuine": float(prediction_proba[1])
        },        "risk_score": float(1 - prediction_proba[1]),  # Higher score = more likely fake
        "features_extracted": {
            "review_length": features_6.iloc[0]['review_length'],
            "word_count": features_6.iloc[0]['word_count'],
            "star_rating": features_6.iloc[0]['star_rating'],
            "is_verified": features_6.iloc[0]['is_verified']
        }
    }
    
    return result


def extract_single_review_features(review_data):
    """
    Extract numerical features from a single review dictionary.
    Returns a pandas DataFrame with one row.
    """
    review_text = review_data.get('review_text', '')
    star_rating = float(review_data.get('star_rating', 0))
    verified_purchase = review_data.get('verified_purchase', 'N')
    helpful_votes = int(review_data.get('helpful_votes', 0))
    total_votes = int(review_data.get('total_votes', 0))
    
    # Basic text features
    review_length = len(review_text)
    word_count = len(review_text.split())
    exclamation_count = review_text.count('!')
    question_count = review_text.count('?')
    caps_ratio = sum(1 for c in review_text if c.isupper()) / len(review_text) if len(review_text) > 0 else 0
    
    # Sentiment indicators
    positive_words_list = ['good', 'great', 'excellent', 'amazing', 'love', 'wonderful', 'perfect', 'best']
    negative_words_list = ['bad', 'terrible', 'awful', 'hate', 'poor', 'worst', 'horrible', 'disappointing']
    
    review_lower = review_text.lower()
    positive_words = sum(1 for word in positive_words_list if word in review_lower)
    negative_words = sum(1 for word in negative_words_list if word in review_lower)
    
    # Rating-text mismatch detection
    rating_text_mismatch = int(
        (star_rating >= 4 and negative_words > positive_words) or
        (star_rating <= 2 and positive_words > negative_words)
    )
    
    # Purchase verification
    is_verified = 1 if str(verified_purchase).upper() == 'Y' else 0
    
    # Helpfulness features
    helpfulness_ratio = helpful_votes / total_votes if total_votes > 0 else 0
    has_helpful_votes = 1 if helpful_votes > 0 else 0
    
    # For single review, duplicate_review is always 0
    duplicate_review = 0
    
    # Create feature dictionary matching training format
    features = {
        'star_rating': star_rating,
        'review_length': review_length,
        'word_count': word_count,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'caps_ratio': caps_ratio,
        'positive_words': positive_words,
        'negative_words': negative_words,
        'rating_text_mismatch': rating_text_mismatch,
        'is_verified': is_verified,
        'helpfulness_ratio': helpfulness_ratio,
        'has_helpful_votes': has_helpful_votes,
        'duplicate_review': duplicate_review
    }
    
    return pd.DataFrame([features])


def predict_batch(reviews_data):
    """
    Predict multiple reviews at once
    
    Args:
        reviews_data: List of dictionaries, each containing review data
    
    Returns:
        List of prediction results
    """
    results = []
    
    for i, review_data in enumerate(reviews_data):
        try:
            result = predict(review_data)
            result['index'] = i
            results.append(result)
        except Exception as e:
            results.append({
                'index': i,
                'error': f"Prediction failed: {str(e)}"
            })
    
    return results
