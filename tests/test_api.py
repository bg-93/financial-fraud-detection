from fastapi.testclient import TestClient

from api import TransactionRequest, app, get_ready_predictor
from fraud_detection import InputValidationError, PredictionResult, REQUEST_FEATURES


VALID_REQUEST = {
    "type": "TRANSFER",
    "amount": 8_700.0,
    "hour": 2,
    "sender_tx_count_before": 12,
    "sender_mean_amount_before": 190.0,
    "recipient_tx_count_before": 2,
    "pair_tx_count_before": 0,
    "is_new_recipient": 1,
    "hours_since_sender_tx": 3.0,
}


class ReadyPredictor:
    ready = True
    model_name = "Test model"
    load_error = None
    deployment_warning = None

    def predict(self, transaction):
        if transaction["pair_tx_count_before"] and transaction["is_new_recipient"]:
            raise InputValidationError("pair history is inconsistent")
        return PredictionResult(
            fraud_score=0.86,
            send_for_review=True,
            review_threshold=0.20,
            model=self.model_name,
        )


def test_api_schema_matches_shared_request_contract():
    assert set(TransactionRequest.model_fields) == set(REQUEST_FEATURES)


def test_predict_returns_typed_decision():
    app.dependency_overrides[get_ready_predictor] = lambda: ReadyPredictor()
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "fraud_score": 0.86,
        "send_for_review": True,
        "review_threshold": 0.20,
        "model": "Test model",
    }


def test_predict_rejects_unknown_fields():
    request = {**VALID_REQUEST, "account_id": "secret"}
    app.dependency_overrides[get_ready_predictor] = lambda: ReadyPredictor()
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=request)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_predict_rejects_boolean_numeric_fields():
    request = {**VALID_REQUEST, "hour": True}
    app.dependency_overrides[get_ready_predictor] = lambda: ReadyPredictor()
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=request)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_predict_maps_shared_validation_errors_to_422():
    request = {**VALID_REQUEST, "pair_tx_count_before": 1, "is_new_recipient": 1}
    app.dependency_overrides[get_ready_predictor] = lambda: ReadyPredictor()
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=request)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {"detail": "pair history is inconsistent"}


def test_health_reports_ready_model():
    with TestClient(app) as client:
        client.app.state.predictor = ReadyPredictor()
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model"] == "Test model"
