import requests
import json

# Flask API base URL
BASE_URL = "http://localhost:5000"


def test_api():
    """Test the Flask API endpoints"""

    print("🧪 Testing Review Fraud Detection API")
    print("=" * 50)

    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Model info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Train model (GET method - easier for testing)
    print("\n3. Testing model training (GET method)...")
    try:
        response = requests.get(f"{BASE_URL}/train?file_path=/content/Book1.csv")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Message: {result['message']}")
            if 'metrics' in result:
                print(f"   Accuracy: {result['metrics'].get('accuracy', 'N/A')}")
        else:
            print(f"   Error: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Train model (POST method with proper headers)
    print("\n4. Testing model training (POST method)...")
    train_data = {"file_path": "/content/Book1.csv"}
    try:
        response = requests.post(
            f"{BASE_URL}/train",
            json=train_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Message: {result['message']}")
            if 'metrics' in result:
                print(f"   Accuracy: {result['metrics'].get('accuracy', 'N/A')}")        else:
            print(f"   Error: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Single review prediction
    print("\n5. Testing single review prediction...")
    sample_review = {
        "review_text": "This product is absolutely amazing! Best purchase ever! 5 stars! Highly recommend!",
        "star_rating": 5,
        "verified_purchase": "Y",
        "helpful_votes": 0,
        "total_votes": 0,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_review,
            headers={"Content-Type": "application/json"},
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Prediction: {result['prediction_label']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Risk Score: {result['risk_score']:.4f}")
        else:
            print(f"   Error: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Batch prediction
    print("\n4. Testing batch review prediction...")
    batch_reviews = {
        "reviews": [
            {"review_text": "Great product! Love it!", "star_rating": 5},
            {
                "review_text": "Terrible quality. Waste of money. Don't buy this!",
                "star_rating": 1,
            },
            {
                "review_text": "Good value for money. Works as expected. Satisfied with the purchase.",
                "star_rating": 4,
            },
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/batch-predict",
            json=batch_reviews,
            headers={"Content-Type": "application/json"},
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"   Processed: {results['total_processed']} reviews")
            for result in results["results"]:
                if "error" not in result:
                    print(
                        f"     Review {result['index']}: {result['prediction_label']} (Risk: {result['risk_score']:.4f})"
                    )
                else:
                    print(f"     Review {result['index']}: Error - {result['error']}")
        else:
            print(f"   Error: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    test_api()
