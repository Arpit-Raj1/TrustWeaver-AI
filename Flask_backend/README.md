# Review Fraud Detection Flask API

This Flask application provides a REST API for detecting fake reviews using machine learning. The API is built around the training script in `train.py` and provides endpoints for training models and making predictions.

## Features

-   **Train Model**: Train a machine learning model on review data
-   **Single Prediction**: Predict if a single review is fake or genuine
-   **Batch Prediction**: Predict multiple reviews at once
-   **Model Management**: Load/save trained models
-   **Health Monitoring**: Check API and model status

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your training data CSV file in the project directory. The CSV should have these columns:

-   `review_text`: The review content
-   `review_headline`: Review title (optional)
-   `star_rating`: Rating (1-5)
-   `label`: 1 for genuine, 0 for fake
-   `verified_purchase`: Y/N
-   `helpful_votes`: Number of helpful votes
-   `total_votes`: Total votes
-   `product_title`: Product name
-   `customer_id`: Customer identifier
-   `product_parent`: Product identifier

### 3. Run the Flask Application

```bash
python app.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check

```http
GET /health
```

Response:

```json
{
	"status": "healthy",
	"model_status": "loaded",
	"vectorizer_status": "loaded",
	"timestamp": "2024-06-23T10:30:00"
}
```

### 2. Train Model

```http
POST /train
Content-Type: application/json

{
  "file_path": "/path/to/your/data.csv"
}
```

Response:

```json
{
	"message": "Model trained successfully",
	"metrics": {
		"accuracy": 0.8542,
		"precision": 0.8234,
		"recall": 0.8756,
		"f1_score": 0.8487,
		"auc": 0.9123
	},
	"timestamp": "2024-06-23T10:30:00"
}
```

### 3. Single Review Prediction

```http
POST /predict
Content-Type: application/json

{
  "review_text": "This product is amazing! Best purchase ever!",
  "star_rating": 5,
  "verified_purchase": "Y",
  "helpful_votes": 0,
  "total_votes": 0
}
```

Response:

```json
{
	"prediction": 0,
	"prediction_label": "Fake",
	"confidence": {
		"fake": 0.8234,
		"genuine": 0.1766
	},
	"risk_score": 0.8234,
	"features_analyzed": {
		"review_length": 45,
		"word_count": 8,
		"exclamation_count": 2,
		"caps_ratio": 0.0667,
		"positive_words": 3,
		"negative_words": 0,
		"rating_text_mismatch": 0
	},
	"timestamp": "2024-06-23T10:30:00"
}
```

### 4. Batch Prediction

```http
POST /batch-predict
Content-Type: application/json

{
  "reviews": [
    {
      "review_text": "Great product!",
      "star_rating": 5
    },
    {
      "review_text": "Terrible quality.",
      "star_rating": 1
    }
  ]
}
```

Response:

```json
{
	"results": [
		{
			"index": 0,
			"prediction": 1,
			"prediction_label": "Genuine",
			"confidence": { "fake": 0.2345, "genuine": 0.7655 },
			"risk_score": 0.2345
		},
		{
			"index": 1,
			"prediction": 1,
			"prediction_label": "Genuine",
			"confidence": { "fake": 0.3456, "genuine": 0.6544 },
			"risk_score": 0.3456
		}
	],
	"total_processed": 2,
	"timestamp": "2024-06-23T10:30:00"
}
```

### 5. Model Information

```http
GET /model-info
```

Response:

```json
{
  "model_loaded": true,
  "vectorizer_loaded": true,
  "model_type": "RandomForestClassifier",
  "feature_columns": ["star_rating", "review_length", "word_count", ...],
  "files_exist": {
    "model.pkl": true,
    "vectorizer.pkl": true
  },
  "timestamp": "2024-06-23T10:30:00"
}
```

## Usage Example

### Python Client Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

# Train the model
train_data = {"file_path": "/path/to/Book1.csv"}
response = requests.post(f"{BASE_URL}/train", json=train_data)
print("Training result:", response.json())

# Predict a single review
review_data = {
    "review_text": "This product is absolutely amazing! Best purchase ever!",
    "star_rating": 5,
    "verified_purchase": "Y"
}
response = requests.post(f"{BASE_URL}/predict", json=review_data)
result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Risk Score: {result['risk_score']:.4f}")
```

### cURL Example

```bash
# Health check
curl -X GET http://localhost:5000/health

# Train model
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/content/Book1.csv"}'

# Predict single review
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "review_text": "Amazing product! Highly recommend!",
    "star_rating": 5
  }'
```

## Testing

Run the test script to verify all endpoints:

```bash
python test_api.py
```

## Model Features

The model analyzes these features to detect fake reviews:

-   **Text Features**: Length, word count, punctuation usage
-   **Sentiment Analysis**: Positive/negative word counts
-   **Rating Consistency**: Mismatch between rating and sentiment
-   **Verification Status**: Whether purchase was verified
-   **Helpfulness**: Community feedback on the review
-   **TF-IDF Vectors**: Text content analysis

## Production Deployment

For production deployment:

1. Set environment variables:

    ```bash
    export FLASK_ENV=production
    export SECRET_KEY=your-secret-key
    ```

2. Use a production WSGI server like Gunicorn:

    ```bash
    pip install gunicorn
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
    ```

3. Consider using a reverse proxy like Nginx for better performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Model Not Found**: Train the model first using `/train` endpoint
3. **Data Format Issues**: Ensure CSV has required columns
4. **Memory Issues**: Large datasets may require more RAM

### Logs

Check the console output for detailed error messages and training progress.

## File Structure

```
Flask_backend/
├── app.py              # Main Flask application
├── train.py            # ML training script
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── test_api.py         # API testing script
├── README.md           # This file
├── model.pkl           # Trained model (generated)
└── vectorizer.pkl      # TF-IDF vectorizer (generated)
```

## License

This project is for educational and research purposes.

## Batch Prediction Script

The `predict.py` script allows you to make predictions on a CSV file containing multiple reviews.

### Usage

```bash
# Basic usage
python predict.py input_reviews.csv

# Specify output file
python predict.py input_reviews.csv predictions.csv

# Analysis only (no output file)
python predict.py input_reviews.csv --analysis-only

# Help
python predict.py --help
```

### Input CSV Format

Your input CSV file must contain at least a `review_text` column. Optional columns include:

-   `star_rating`: Rating (1-5) - defaults to 3
-   `verified_purchase`: Y/N - defaults to N
-   `helpful_votes`: Number of helpful votes - defaults to 0
-   `total_votes`: Total votes - defaults to 0

Example CSV:

```csv
review_text,star_rating,verified_purchase,helpful_votes,total_votes
"Great product! Love it!",5,Y,2,3
"Terrible quality. Don't buy.",1,N,0,1
"Good value for money.",4,Y,1,2
```

### Output

The script generates:

1. **Main output CSV** with original data plus:

    - `prediction`: 0 (fake) or 1 (genuine)
    - `prediction_label`: "Fake" or "Genuine"
    - `fake_probability`: Probability of being fake (0-1)
    - `genuine_probability`: Probability of being genuine (0-1)
    - `risk_score`: Risk score (0-1, higher = more suspicious)
    - `confidence`: Model confidence in prediction
    - `feature_*`: Various extracted features

2. **Summary file** with analysis statistics

### Easy Usage

Use the provided batch scripts:

**Windows Batch:**

```cmd
run_predictions.bat sample_reviews.csv
```

**PowerShell:**

```powershell
.\run_predictions.ps1 sample_reviews.csv
```

## API Endpoints
