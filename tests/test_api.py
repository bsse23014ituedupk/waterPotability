"""
FastAPI endpoint tests — health check, valid prediction, invalid input rejection.

Uses FastAPI's TestClient (built on httpx) for in-process API testing
without needing a running server. The predictor is mocked to avoid
requiring trained model artifacts during CI.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_SAMPLE = {
    "ph": 7.0,
    "Hardness": 196.0,
    "Solids": 20791.0,
    "Chloramines": 7.3,
    "Sulfate": 368.0,
    "Conductivity": 564.0,
    "Organic_carbon": 10.4,
    "Trihalomethanes": 86.0,
    "Turbidity": 2.96,
}

MOCK_PREDICTION = {
    "potability": 1,
    "probability": 0.7234,
    "threshold_used": 0.48,
    "interpretation": "POTABLE",
    "confidence": "HIGH",
}


@pytest.fixture
def client():
    """FastAPI test client with predictor mocked to avoid needing model artifacts."""
    mock_predictor = MagicMock()
    mock_predictor.model = MagicMock()
    mock_predictor.model_version = "test_20260101_000000"
    mock_predictor.threshold = 0.48
    mock_predictor.predict.return_value = MOCK_PREDICTION

    with patch("api.main.predictor", mock_predictor):
        from api.main import app
        with TestClient(app) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_endpoint_returns_200(self, client):
        """Health endpoint must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response must contain all required fields."""
        response = client.get("/health")
        data = response.json()

        required_fields = {"status", "model_loaded", "model_version", "threshold"}
        assert required_fields.issubset(set(data.keys())), (
            f"Missing fields: {required_fields - set(data.keys())}"
        )

    def test_health_model_loaded_true(self, client):
        """model_loaded must be True when predictor is available."""
        response = client.get("/health")
        assert response.json()["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_input_returns_200(self, client):
        """Valid input must return HTTP 200."""
        response = client.post("/predict", json=VALID_SAMPLE)
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        """Prediction response must contain all required fields."""
        response = client.post("/predict", json=VALID_SAMPLE)
        data = response.json()

        required_fields = {
            "potability", "probability", "threshold_used",
            "interpretation", "confidence"
        }
        assert required_fields.issubset(set(data.keys())), (
            f"Missing fields: {required_fields - set(data.keys())}"
        )

    def test_predict_potability_is_binary(self, client):
        """Potability field must be 0 or 1."""
        response = client.post("/predict", json=VALID_SAMPLE)
        potability = response.json()["potability"]
        assert potability in [0, 1], (
            f"Potability must be 0 or 1, got: {potability}"
        )

    def test_predict_probability_in_range(self, client):
        """Probability must be in [0, 1]."""
        response = client.post("/predict", json=VALID_SAMPLE)
        probability = response.json()["probability"]
        assert 0.0 <= probability <= 1.0, (
            f"Probability {probability} is outside [0, 1]."
        )

    def test_predict_invalid_ph_rejected(self, client):
        """pH value outside [0, 14] must be rejected with 422."""
        invalid_sample = {**VALID_SAMPLE, "ph": 20.0}
        response = client.post("/predict", json=invalid_sample)
        assert response.status_code == 422, (
            f"Expected 422 for pH=20.0, got {response.status_code}"
        )

    def test_predict_negative_ph_rejected(self, client):
        """Negative pH must be rejected with 422."""
        invalid_sample = {**VALID_SAMPLE, "ph": -1.0}
        response = client.post("/predict", json=invalid_sample)
        assert response.status_code == 422

    def test_predict_missing_field_rejected(self, client):
        """Request missing required fields must return 422."""
        incomplete_sample = {"ph": 7.0}  # Missing 8 fields
        response = client.post("/predict", json=incomplete_sample)
        assert response.status_code == 422, (
            f"Expected 422 for incomplete input, got {response.status_code}"
        )

    def test_predict_empty_body_rejected(self, client):
        """Empty request body must return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_negative_hardness_rejected(self, client):
        """Negative Hardness value must be rejected with 422."""
        invalid_sample = {**VALID_SAMPLE, "Hardness": -10.0}
        response = client.post("/predict", json=invalid_sample)
        assert response.status_code == 422
