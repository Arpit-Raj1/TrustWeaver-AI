import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import os
import json
from datetime import datetime
import joblib


import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle

# Install missing dependencies for TrustWeaver
import subprocess
import sys
import pandas as pd
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer


# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import re

import os

# Train multiple models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import time


def install_package(package):
    """Install a package if it's not already installed"""
    try:
        __import__(package)
        print(f"{package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully")


def extract_features_from_book1(df):
    """Extract features from Book1.csv dataset for model training"""
    print("\n🔧 EXTRACTING FEATURES FROM BOOK1.CSV...")

    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Text-based features
    if "review_text" in df.columns:
        # Review length
        df_processed["review_length"] = df["review_text"].str.len()

        # Word count
        df_processed["word_count"] = df["review_text"].str.split().str.len()

        # Exclamation marks count
        df_processed["exclamation_count"] = df["review_text"].str.count("!")

        # Question marks count (fixed escape sequence)
        df_processed["question_count"] = df["review_text"].str.count(r"\?")

        # Capital letters percentage
        df_processed["caps_percentage"] = df["review_text"].apply(
            lambda x: (
                sum(1 for c in str(x) if c.isupper()) / len(str(x))
                if len(str(x)) > 0
                else 0
            )
        )

        # Sentiment analysis
        df_processed["sentiment_score"] = df["review_text"].apply(
            lambda x: sia.polarity_scores(str(x))["compound"]
        )

        # Rating-sentiment mismatch (high rating but negative sentiment or vice versa)
        if "star_rating" in df.columns:
            df_processed["rating_sentiment_mismatch"] = (
                ((df["star_rating"] >= 4) & (df_processed["sentiment_score"] < -0.2))
                | ((df["star_rating"] <= 2) & (df_processed["sentiment_score"] > 0.2))
            ).astype(int)

    # Review metadata features
    if "verified_purchase" in df.columns:
        df_processed["verified_purchase_num"] = (df["verified_purchase"] == "Y").astype(
            int
        )

    if "helpful_votes" in df.columns:
        df_processed["has_helpful_votes"] = (df["helpful_votes"] > 0).astype(int)

    print(f"✅ Extracted features for {len(df_processed)} reviews")

    # Display sample of processed data
    print("\n📊 Sample processed data:")
    print(df_processed.head(3))

    return df_processed


def generate_synthetic_data(num_samples=5000, fake_ratio=0.3):
    """Generate synthetic review data for testing"""
    print("🧪 Generating synthetic data for demonstration...")
    np.random.seed(42)

    fake_count = int(num_samples * fake_ratio)
    real_count = num_samples - fake_count

    # Generate fake reviews (more extreme, repetitive patterns)
    fake_reviews = []
    fake_templates = [
        "This product is amazing! Best purchase ever! 5 stars! Highly recommend!",
        "Excellent quality! Fast shipping! Perfect! Love it! Buy now!",
        "Terrible product! Waste of money! Don't buy! Poor quality!",
        "Outstanding! Fantastic! Wonderful! Great value! Perfect condition!",
    ]

    for _ in range(fake_count):
        template = np.random.choice(fake_templates)
        # Add some variation
        variations = [" Really great!", " Awesome!", " Perfect!", " Amazing!"]
        review = template + np.random.choice(variations)
        fake_reviews.append(review)

    # Generate real reviews (more natural, varied language)
    real_reviews = []
    real_templates = [
        "Good product for the price. Delivery was on time and packaging was adequate.",
        "Decent quality. Had some minor issues but customer service resolved them quickly.",
        "Works as expected. Nothing special but does the job. Fair value for money.",
        "Pretty satisfied with this purchase. Build quality is solid and it works well.",
        "Not bad overall. Some room for improvement but generally happy with it.",
    ]

    for _ in range(real_count):
        template = np.random.choice(real_templates)
        # Add natural variations
        variations = [
            " Would recommend.",
            " Good experience.",
            " Satisfied.",
            " Happy with purchase.",
        ]
        review = template + np.random.choice(variations)
        real_reviews.append(review)

    # Combine reviews
    all_reviews = fake_reviews + real_reviews
    fake_labels = [1] * fake_count + [0] * real_count

    # Generate other features
    star_ratings = []
    for i, label in enumerate(fake_labels):
        if label == 1:  # Fake reviews tend to be more extreme
            star_ratings.append(np.random.choice([1, 5], p=[0.3, 0.7]))
        else:  # Real reviews more distributed
            star_ratings.append(
                np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.2, 0.35, 0.2])
            )

    # Generate synthetic metadata
    product_titles = [
        f"Product {np.random.randint(1, 100)}" for _ in range(num_samples)
    ]
    product_categories = np.random.choice(
        ["Electronics", "Books", "Clothing", "Home", "Sports"], num_samples
    )
    customer_id_list = [
        f"customer{np.random.randint(1, 1000)}" for _ in range(num_samples)
    ]
    seller_id_list = [f"seller{np.random.randint(1, 50)}" for _ in range(num_samples)]

    # Create timestamps
    base_time = pd.Timestamp("2024-01-01").timestamp()
    time_range = pd.Timestamp("2024-12-31").timestamp() - base_time
    transaction_timestamps = [
        base_time + np.random.uniform(0, time_range) for _ in range(num_samples)
    ]

    # Create dataframe
    data = {
        "review_text": all_reviews,
        "star_rating": star_ratings,
        "product_title": product_titles,
        "product_category": product_categories,
        "transaction_timestamp": transaction_timestamps,
        "seller_id": seller_id_list,
        "customer_id": customer_id_list,
        "label": fake_labels,
    }

    df = pd.DataFrame(data)

    # Add behavioral patterns for fake reviews
    ip_addresses = [
        f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        for _ in range(num_samples)
    ]

    # Make some fake reviews share IP addresses (suspicious pattern)
    fake_indices = df[df.label == 1].index.tolist()
    for i in range(0, len(fake_indices), 3):
        if i + 2 < len(fake_indices):
            shared_ip = f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            ip_addresses[fake_indices[i]] = shared_ip
            ip_addresses[fake_indices[i + 1]] = shared_ip
            ip_addresses[fake_indices[i + 2]] = shared_ip

    df["ip_address"] = ip_addresses

    return df


# 📂 EXTERNAL DATA LOADING WITH TRAIN-TEST SPLIT


def load_external_data(file_path=None, data_format="auto"):
    """
    Load external data from CSV or JSON files

    Args:
        file_path: Path to the data file (CSV or JSON)
        data_format: 'csv', 'json', or 'auto' (auto-detect from extension)

    Returns:
        DataFrame with loaded data
    """

    if file_path is None:
        print("🧪 No external file provided. Using synthetic data for demonstration...")
        return generate_synthetic_data(num_samples=5000, fake_ratio=0.3)

    # Auto-detect format if not specified
    if data_format == "auto":
        if file_path.endswith(".csv"):
            data_format = "csv"
        elif file_path.endswith(".json"):
            data_format = "json"
        else:
            raise ValueError(
                "Cannot auto-detect format. Please specify 'csv' or 'json'"
            )

    print(f"📁 Loading data from: {file_path}")
    print(f"📋 Format: {data_format.upper()}")

    try:
        if data_format == "csv":
            df = pd.read_csv(file_path)
        elif data_format == "json":
            df = pd.read_json(file_path)
        else:
            raise ValueError("Supported formats: 'csv' or 'json'")

        print(f"✅ Successfully loaded {len(df)} records")
        return df

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("🧪 Falling back to synthetic data...")
        return generate_synthetic_data(num_samples=5000, fake_ratio=0.3)
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        print("🧪 Falling back to synthetic data...")
        return generate_synthetic_data(num_samples=5000, fake_ratio=0.3)


def extract_features_from_book1(df):
    """Extract comprehensive features from Book1.csv data"""
    print("🔍 Extracting features from Book1.csv...")

    # Text-based features
    df["review_length"] = df["review_text"].str.len()
    df["word_count"] = df["review_text"].str.split().str.len()
    df["exclamation_count"] = df["review_text"].str.count("!")
    df["question_count"] = df["review_text"].str.count(r"\?")
    df["caps_ratio"] = df["review_text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )

    # Sentiment indicators
    positive_words = [
        "good",
        "great",
        "excellent",
        "amazing",
        "love",
        "perfect",
        "best",
        "wonderful",
    ]
    negative_words = [
        "bad",
        "terrible",
        "awful",
        "hate",
        "worst",
        "horrible",
        "disappointing",
    ]

    df["positive_words"] = (
        df["review_text"]
        .str.lower()
        .apply(lambda x: sum(1 for word in positive_words if word in x))
    )
    df["negative_words"] = (
        df["review_text"]
        .str.lower()
        .apply(lambda x: sum(1 for word in negative_words if word in x))
    )

    # Rating consistency features
    df["rating_text_mismatch"] = 0

    # High rating but negative words
    df.loc[
        (df["star_rating"] >= 4) & (df["negative_words"] > df["positive_words"]),
        "rating_text_mismatch",
    ] = 1

    # Low rating but positive words
    df.loc[
        (df["star_rating"] <= 2) & (df["positive_words"] > df["negative_words"]),
        "rating_text_mismatch",
    ] = 1

    # Purchase verification features
    df["is_verified"] = (df["verified_purchase"] == "Y").astype(int)

    # Helpfulness features
    df["helpfulness_ratio"] = df["helpful_votes"] / (
        df["total_votes"] + 1
    )  # Add 1 to avoid division by zero
    df["has_helpful_votes"] = (df["helpful_votes"] > 0).astype(int)

    # Duplicate detection (simplified)
    df["duplicate_review"] = df.duplicated(subset=["review_text"], keep=False).astype(
        int
    )

    print(f"✅ Extracted features for {len(df)} reviews")
    return df


def train_model(path="D:/AA/TrustWeaver-AI/Book1.csv"):
    # Install required packages
    required_packages = [
        "textstat",  # For readability analysis
        "networkx",  # For graph analysis (usually included)
        "seaborn",  # For visualizations (usually included)
        "tqdm",  # For progress bars (usually included)
        "scikit-learn",  # For ML models (usually included)
    ]

    print("CHECKING AND INSTALLING DEPENDENCIES...")
    print("=" * 50)

    for package in required_packages:
        try:
            install_package(package)
        except Exception as e:
            print(f"Warning: Could not install {package}: {e}")
            print(f"You may need to install it manually: pip install {package}")

    print("\nDependency check completed!")

    try:
        # Load data directly from Book1.csv
        file_path = path
        print(f"Loading data from: {file_path}")

        df_raw = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {len(df_raw)} records from Book1.csv")

        # Map Book1.csv columns to our required format
        df = df_raw.copy()

        # Display basic info about the dataset
        print(f"\nBook1.csv Dataset Summary:")
        print(f"   • Total reviews: {len(df)}")

        if "label" in df.columns:
            genuine_count = (df["label"] == 1).sum()  # 1 = genuine in Book1.csv
            fake_count = (df["label"] == 0).sum()  # 0 = fake in Book1.csv
            print(
                f"   • Genuine reviews: {genuine_count} ({genuine_count/len(df)*100:.1f}%)"
            )
            print(f"   • Fake reviews: {fake_count} ({fake_count/len(df)*100:.1f}%)")

        # Additional Book1.csv specific analysis
        print(f"\nBook1.csv Data Quality:")
        if "star_rating" in df.columns:
            print(
                f"   • Star ratings range: {df['star_rating'].min()} - {df['star_rating'].max()}"
            )
        if "review_text" in df.columns:
            print(
                f"   • Average review length: {df['review_text'].str.len().mean():.1f} characters"
            )
        if "verified_purchase" in df.columns:
            print(
                f"   • Verified purchases: {(df['verified_purchase'] == 'Y').sum()} ({(df['verified_purchase'] == 'Y').mean()*100:.1f}%)"
            )
        if "helpful_votes" in df.columns:
            print(f"   • Reviews with helpful votes: {(df['helpful_votes'] > 0).sum()}")

        # Display sample data
        print(f"\nSample Book1.csv data:")
        print(df.head(3))

    except Exception as e:
        print(f"❌ Error loading Book1.csv: {str(e)}")
        print(f"   Make sure Book1.csv is in the current directory: {os.getcwd()}")

    print("GPU Status:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        device = torch.device("cuda")
        print("✅ GPU ready for training!")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU mode")

    # Download necessary NLTK data
    try:
        nltk.data.find("vader_lexicon")
    except:
        nltk.download("vader_lexicon")

    # Extract features
    df_processed = extract_features_from_book1(df.copy())

    # Essential imports for TrustWeaver ML pipeline

    print("✅ All imports loaded successfully!")

    # Prepare features for training
    print("\nPREPARING FEATURES FOR TRAINING...")

    # Select features for model training based on what's available
    feature_columns = []

    # Add available features to the list
    if "review_length" in df_processed.columns:
        feature_columns.append("review_length")
    if "word_count" in df_processed.columns:
        feature_columns.append("word_count")
    if "exclamation_count" in df_processed.columns:
        feature_columns.append("exclamation_count")
    if "question_count" in df_processed.columns:
        feature_columns.append("question_count")
    if "caps_percentage" in df_processed.columns:
        feature_columns.append("caps_percentage")
    if "sentiment_score" in df_processed.columns:
        feature_columns.append("sentiment_score")
    if "rating_sentiment_mismatch" in df_processed.columns:
        feature_columns.append("rating_sentiment_mismatch")
    if "verified_purchase_num" in df_processed.columns:
        feature_columns.append("verified_purchase_num")
    if "has_helpful_votes" in df_processed.columns:
        feature_columns.append("has_helpful_votes")
    if "star_rating" in df_processed.columns:
        feature_columns.append("star_rating")

    # Validate if we have the label column
    if "label" not in df_processed.columns:
        print("❌ Error: 'label' column not found in the dataset")
        # If no label column, check if we need to create one based on other columns
        if "is_fake" in df_processed.columns:
            df_processed["label"] = df_processed["is_fake"]
            print("✅ Using 'is_fake' column as label")
        else:
            print(
                "❓ No label column found. Creating a synthetic label for demonstration"
            )
            # Create a synthetic label based on available features (for demonstration only)
            df_processed["label"] = (df_processed["sentiment_score"] < 0).astype(int)

    # Prepare feature set and label
    X = df_processed[feature_columns].fillna(0)
    y = df_processed["label"]

    # Train-Test Split (70-30)
    TRAIN_RATIO = 0.7
    RANDOM_STATE = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    print(f"✅ Features prepared for training")
    print(
        f"   • Selected {len(feature_columns)} features: {', '.join(feature_columns)}"
    )
    print(f"   • Training set: {X_train.shape[0]} samples")
    print(f"   • Testing set: {X_test.shape[0]} samples")
    print(f"   • Label distribution: {y.value_counts().to_dict()}")

    # 📂 Data Loading - Replace with Your Real Dataset

    # Option 1: Load your own dataset (RECOMMENDED)
    # df = pd.read_csv('path/to/your/dataset.csv')
    # print(f"Loaded dataset with {len(df)} reviews")

    # Option 2: Use synthetic data for testing (current approach)    # 🔧 CONFIGURATION: Using Book1.csv for real review data
    EXTERNAL_DATA_PATH = path
    DATA_FORMAT = "csv"

    # Load data (Book1.csv)
    print("🚀 LOADING BOOK1.CSV DATA FOR TRAINING...")
    print("=" * 50)

    try:
        print(f"📂 Loading data from: {EXTERNAL_DATA_PATH}")
        df_raw = pd.read_csv(EXTERNAL_DATA_PATH)
        print(f"✅ Successfully loaded {len(df_raw)} records from Book1.csv")        # Map Book1.csv columns to our required format
        df = pd.DataFrame(
            {
                "review_text": df_raw["review_body"].fillna("")
                + " "
                + df_raw["review_headline"].fillna(""),
                "star_rating": df_raw["star_rating"],
                "product_title": df_raw["product_title"],
                "customer_id": df_raw["customer_id"],
                "seller_id": df_raw[
                    "product_parent"
                ],  # Using product_parent as seller proxy
                "label": df_raw["label"],
                "verified_purchase": df_raw["verified_purchase"],
                "helpful_votes": df_raw["helpful_votes"],
                "total_votes": df_raw["total_votes"],
            }
        )

        # Clean the data
        df["review_text"] = df["review_text"].str.strip()
        df = df[df["review_text"] != ""].reset_index(drop=True)

        print(f"✅ Processed and cleaned data: {len(df)} reviews")

    except Exception as e:
        print(f"❌ Error loading Book1.csv: {str(e)}")
        print("🧪 Falling back to synthetic data...")
        df = load_external_data(None, DATA_FORMAT)

    # Validate data format for Book1.csv
    required_columns = [
        "review_text",
        "star_rating",
        "product_title",
        "customer_id",
        "seller_id",
        "label",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    print(f"\n📊 Book1.csv Dataset Summary:")
    print(f"   • Total reviews: {len(df)}")
    print(f"   • Columns available: {list(df.columns)}")

    if "label" in df.columns:
        genuine_count = (df["label"] == 1).sum()  # 1 = genuine in Book1.csv
        fake_count = (df["label"] == 0).sum()  # 0 = fake in Book1.csv
        print(
            f"   • Genuine reviews: {genuine_count} ({genuine_count/len(df)*100:.1f}%)"
        )
        print(f"   • Fake reviews: {fake_count} ({fake_count/len(df)*100:.1f}%)")

    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print("📋 Required columns for training:")
        for col in required_columns:
            print(f"   • {col}: {'✅' if col in df.columns else '❌'}")
    else:
        print("✅ All required columns present!")

    # Additional Book1.csv specific analysis
    print(f"\n🔍 Book1.csv Data Quality:")
    print(
        f"   • Star ratings range: {df['star_rating'].min()} - {df['star_rating'].max()}"
    )
    print(
        f"   • Average review length: {df['review_text'].str.len().mean():.1f} characters"
    )
    print(
        f"   • Verified purchases: {(df['verified_purchase'] == 'Y').sum()} ({(df['verified_purchase'] == 'Y').mean()*100:.1f}%)"
    )
    print(f"   • Reviews with helpful votes: {(df['helpful_votes'] > 0).sum()}")

    # 🎯 TRAIN-TEST SPLIT CONFIGURATION (70-30 split as requested)
    TRAIN_RATIO = 0.7  # 70% for training, 30% for testing
    RANDOM_STATE = 42  # For reproducible results

    print(f"\n🔄 TRAIN-TEST SPLIT CONFIGURATION:")
    print(f"   • Training: {TRAIN_RATIO*100:.0f}%")
    print(f"   • Testing: {(1-TRAIN_RATIO)*100:.0f}%")
    print(f"   • Random state: {RANDOM_STATE}")

    # Display sample data from Book1.csv
    print(f"\n🔍 Sample Book1.csv data:")
    sample_cols = [
        "review_text",
        "star_rating",
        "product_title",
        "label",
        "verified_purchase",
    ]
    print(df[sample_cols].head())

    print("\n🚀 TRAINING MODELS...")
    print("=" * 60)

    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }

    # Store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        start_time = time.time()
        print(f"\n🔄 Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Try to calculate AUC, but it might not be possible for all models
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = None
        except:
            auc = None

        training_time = time.time() - start_time

        # Store results
        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc,
            "training_time": training_time,
            "predictions": y_pred,
        }

        print(f"   ✅ {name} trained in {training_time:.2f}s")
        print(
            f"      Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
        )

    # Display comprehensive results
    print(f"\n🏆 MODEL PERFORMANCE RESULTS:")
    print("=" * 60)

    # Sort models by accuracy
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    print(
        f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Time(s)':<10}"
    )
    print("-" * 80)

    for name, metrics in sorted_results:
        acc = metrics["accuracy"]
        prec = metrics["precision"]
        rec = metrics["recall"]
        f1 = metrics["f1_score"]
        auc = metrics["auc"] if metrics["auc"] else 0.0
        time_taken = metrics["training_time"]

        print(
            f"{name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc:<10.4f} {time_taken:<10.2f}"
        )

    # Best model analysis
    best_model_name = sorted_results[0][0]
    best_model_metrics = sorted_results[0][1]
    best_model = best_model_metrics["model"]

    # Process Book1.csv Data with 70-30 Train-Test Split
    print("🔄 PROCESSING BOOK1.CSV DATA...")
    print("=" * 50)

    # Extract features
    df_processed = extract_features_from_book1(df.copy())

    # Prepare features for training
    print("\n📊 PREPARING FEATURES FOR TRAINING...")

    # Select features for model training
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

    # Text features using TF-IDF
    print("🔤 Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_text = tfidf.fit_transform(df_processed["review_text"])

    # Combine features
    from scipy.sparse import hstack

    X_combined = hstack([X_text, X_numerical.values])

    # Target variable
    y = df_processed["label"].values

    print(f"📈 Feature matrix shape: {X_combined.shape}")
    print(f"🎯 Target distribution:")
    print(f"   - Genuine reviews (1): {(y == 1).sum()} ({(y == 1).mean():.1%})")
    print(f"   - Fake reviews (0): {(y == 0).sum()} ({(y == 0).mean():.1%})")

    # 70-30 Train-Test Split (as requested)
    print(f"\n🔄 PERFORMING 70-30 TRAIN-TEST SPLIT...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined,
        y,
        test_size=0.3,  # 30% for testing
        random_state=42,
        stratify=y,  # Maintain class distribution
    )

    print(f"✅ Split completed:")
    print(
        f"   📚 Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_processed):.1%})"
    )
    print(
        f"   🧪 Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df_processed):.1%})"
    )
    print(
        f"   🎯 Training labels: Genuine={(y_train==1).sum()}, Fake={(y_train==0).sum()}"
    )
    print(f"   🎯 Test labels: Genuine={(y_test==1).sum()}, Fake={(y_test==0).sum()}")

    # Feature importance analysis
    print(f"\n🔍 FEATURE ANALYSIS:")
    print(f"📊 Numerical features:")
    for i, col in enumerate(feature_columns):
        mean_genuine = df_processed[df_processed["label"] == 1][col].mean()
        mean_fake = df_processed[df_processed["label"] == 0][col].mean()
        print(f"   • {col}: Genuine={mean_genuine:.2f}, Fake={mean_fake:.2f}")

    print(f"\n✅ Data preprocessing complete - ready for model training!")

    # Model training and evaluation
    print("\n🤖 MODEL TRAINING AND EVALUATION...")
    print("=" * 50)

    # Define models and parameters for training
    models = {
        "Logistic Regression": LogisticRegression(solver="liblinear"),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Support Vector Classifier": SVC(probability=True),
    }

    # Hyperparameter tuning (if applicable) - currently using default parameters

    # Train models and evaluate
    model_metrics = {}

    for model_name, model in models.items():
        print(f"\n🔄 TRAINING {model_name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[
            :, 1
        ]  # Probability of positive class

        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Store metrics
        model_metrics[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "predictions": y_pred,
        }

        print(f"✅ {model_name} trained and evaluated")

    # Identify best model based on accuracy
    best_model_name = max(model_metrics, key=lambda x: model_metrics[x]["accuracy"])
    best_model_metrics = model_metrics[best_model_name]

    print(f"\n🥇 BEST MODEL: {best_model_name}")
    print(
        f"   🎯 Accuracy: {best_model_metrics['accuracy']:.4f} ({best_model_metrics['accuracy']*100:.2f}%)"
    )
    print(f"   📊 Precision: {best_model_metrics['precision']:.4f}")
    print(f"   🔍 Recall: {best_model_metrics['recall']:.4f}")
    print(f"   ⚖️ F1-Score: {best_model_metrics['f1_score']:.4f}")
    if best_model_metrics["auc"]:
        print(f"   📈 AUC: {best_model_metrics['auc']:.4f}")

    # Detailed classification report for best model
    print(f"\n📋 DETAILED CLASSIFICATION REPORT ({best_model_name}):")
    print("-" * 50)
    target_names = ["Fake Review (0)", "Genuine Review (1)"]
    print(
        classification_report(
            y_test, best_model_metrics["predictions"], target_names=target_names
        )
    )

    # Confusion matrix
    print(f"\n🔄 CONFUSION MATRIX ({best_model_name}):")
    cm = confusion_matrix(y_test, best_model_metrics["predictions"])
    print(f"                 Predicted")
    print(f"                 Fake  Genuine")
    print(f"Actual Fake      {cm[0,0]:<4}  {cm[0,1]:<4}")
    print(f"       Genuine   {cm[1,0]:<4}  {cm[1,1]:<4}")

    # Calculate business metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)  # True negative rate
    sensitivity = tp / (tp + fn)  # True positive rate (recall)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)

    print(f"\n💼 BUSINESS IMPACT METRICS:")
    print(
        f"   🛡️ True Positive Rate (Sensitivity): {sensitivity:.4f} - {sensitivity*100:.1f}% of genuine reviews correctly identified"
    )
    print(
        f"   🚫 True Negative Rate (Specificity): {specificity:.4f} - {specificity*100:.1f}% of fake reviews correctly identified"
    )
    print(
        f"   ⚠️ False Positive Rate: {false_positive_rate:.4f} - {false_positive_rate*100:.1f}% of genuine reviews wrongly flagged"
    )
    print(
        f"   ❌ False Negative Rate: {false_negative_rate:.4f} - {false_negative_rate*100:.1f}% of fake reviews missed"
    )

    # Feature importance (for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES ({best_model_name}):")

        # Get numerical feature importances
        feature_names = feature_columns
        importances = best_model.feature_importances_

        # Sort by importance
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance_pairs[:10], 1):
            print(f"   {i:2d}. {feature:<25} {importance:.4f}")

    print(f"\n🎉 MODEL TRAINING COMPLETE!")
    print(
        f"✅ Best accuracy achieved: {best_model_metrics['accuracy']*100:.2f}% with {best_model_name}"
    )
    print(f"📊 Dataset: {len(df_processed)} reviews from Book1.csv")
    print(
        f"🔄 Split: 70% training ({X_train.shape[0]} samples) / 30% testing ({X_test.shape[0]} samples)"
    )
    print(f"🎯 Ready for deployment and real-time review fraud detection!")

    model = best_model
    metrics = {
        "model_name": best_model_name,
        "accuracy": float(best_model_metrics["accuracy"]),
        "precision": float(best_model_metrics["precision"]),
        "recall": float(best_model_metrics["recall"]),
        "f1_score": float(best_model_metrics["f1_score"]),
        "auc": (
            float(best_model_metrics["auc"])
            if best_model_metrics["auc"] is not None
            else None
        ),
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    return model, tfidf, metrics
