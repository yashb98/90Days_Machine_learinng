
from app import app
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_root_endpoint(client):
    """Test that the root endpoint returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    # Some apps may return HTML or JSON, so check general keywords
    assert b"Sentiment" in response.data or b"Welcome" in response.data


def test_predict_endpoint(client):
    """Test /predict endpoint with sample text."""
    payload = {"text": "I love this product!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert "predicted_sentiment" in data
    assert data["predicted_sentiment"] in ["positive", "negative"]
    assert data["text"] == payload["text"]
