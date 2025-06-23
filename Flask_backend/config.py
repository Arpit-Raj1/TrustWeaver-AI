import os
from datetime import timedelta


class Config:
    """Base configuration class"""

    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY") or "your-secret-key-here"
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

    # API settings
    API_TITLE = "Review Fraud Detection API"
    API_VERSION = "1.0"

    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
    VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

    # Data settings
    DEFAULT_DATA_PATH = "/content/Book1.csv"
    TRAIN_TEST_SPLIT = 0.3
    RANDOM_STATE = 42

    # Feature settings
    FEATURE_COLUMNS = [
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

    # TF-IDF settings
    TFIDF_MAX_FEATURES = 1000
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    TFIDF_NGRAM_RANGE = (1, 2)

    # Model settings
    RF_N_ESTIMATORS = 100
    RF_RANDOM_STATE = 42


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
