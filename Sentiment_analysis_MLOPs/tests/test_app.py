# tests/test_app.py

import pytest
from backend.app import app


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
    assert "prediction" in data
    assert data["prediction"] in ["positive", "negative", "neutral"]
