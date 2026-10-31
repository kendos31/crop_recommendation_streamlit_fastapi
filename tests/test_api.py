
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# -----------------------------------------
# Test 1: Health check
# -----------------------------------------

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# -----------------------------------------
# Test 2: Valid crop prediction
# -----------------------------------------

def test_valid_prediction():

    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_crop" in data
    assert "confidence" in data
    assert "top_predictions" in data

    assert isinstance(
        data["predicted_crop"], str
    )

    assert 0 <= data["confidence"] <= 1

    assert len(data["top_predictions"]) == 3


# -----------------------------------------
# Test 3: Missing input feature
# -----------------------------------------

def test_missing_feature():

    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# -----------------------------------------
# Test 4: Invalid humidity
# -----------------------------------------

def test_invalid_humidity():

    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 150,
        "ph": 6.5,
        "rainfall": 202.9
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# -----------------------------------------
# Test 5: Invalid soil pH
# -----------------------------------------

def test_invalid_ph():

    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 15,
        "rainfall": 202.9
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422
