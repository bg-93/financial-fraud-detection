import hashlib
import json
import math

import joblib
import numpy as np
import pytest

from fraud_detection import (
    MODEL_FEATURES,
    FraudPredictor,
    InputValidationError,
    ModelNotReadyError,
    prepare_transaction,
)


class FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features):
        assert list(features.columns) == list(MODEL_FEATURES)
        return np.array([[1.0 - self.probability, self.probability]])


@pytest.fixture
def valid_transaction():
    return {
        "type": "transfer",
        "amount": 8_700.0,
        "hour": 2,
        "sender_tx_count_before": 12,
        "sender_mean_amount_before": 190.0,
        "recipient_tx_count_before": 2,
        "pair_tx_count_before": 0,
        "is_new_recipient": 1,
        "hours_since_sender_tx": 3.0,
    }


def _write_artifacts(directory, probability=0.86, threshold=0.20):
    directory.mkdir()
    model_path = directory / "fraud_model.joblib"
    joblib.dump(FixedProbabilityModel(probability), model_path)
    (directory / "model_metadata.json").write_text(
        json.dumps(
            {
                "model": "Test model",
                "review_threshold": threshold,
                "review_capacity": 0.01,
                "features": list(MODEL_FEATURES),
                "test_metrics": {},
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_prepare_transaction_derives_training_features(valid_transaction):
    valid_transaction["amount_vs_sender_mean"] = 0.0
    features = prepare_transaction(valid_transaction)

    assert list(features.columns) == list(MODEL_FEATURES)
    assert features.loc[0, "type"] == "TRANSFER"
    assert features.loc[0, "log_amount"] == pytest.approx(math.log1p(8_700.0))
    assert features.loc[0, "amount_vs_sender_mean"] == pytest.approx(8_700.0 / 190.0)


def test_prepare_transaction_uses_new_sender_ratio_convention(valid_transaction):
    valid_transaction["sender_mean_amount_before"] = 0
    features = prepare_transaction(valid_transaction)
    assert features.loc[0, "amount_vs_sender_mean"] == 1.0


def test_prepare_transaction_rejects_inconsistent_pair_history(valid_transaction):
    valid_transaction["pair_tx_count_before"] = 3
    with pytest.raises(InputValidationError, match="is_new_recipient"):
        prepare_transaction(valid_transaction)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "sender_tx_count_before": 1,
                "recipient_tx_count_before": 2,
                "pair_tx_count_before": 2,
                "is_new_recipient": 0,
            },
            "cannot exceed sender",
        ),
        (
            {
                "sender_tx_count_before": 3,
                "recipient_tx_count_before": 1,
                "pair_tx_count_before": 2,
                "is_new_recipient": 0,
            },
            "cannot exceed recipient",
        ),
        (
            {
                "sender_tx_count_before": 0,
                "sender_mean_amount_before": 1,
                "pair_tx_count_before": 0,
                "is_new_recipient": 1,
                "hours_since_sender_tx": 10_000,
            },
            "sender_mean_amount_before must be 0",
        ),
        (
            {
                "sender_tx_count_before": 0,
                "sender_mean_amount_before": 0,
                "pair_tx_count_before": 0,
                "is_new_recipient": 1,
                "hours_since_sender_tx": 3,
            },
            "hours_since_sender_tx must be 10000",
        ),
    ],
)
def test_prepare_transaction_rejects_impossible_history(
    valid_transaction, updates, message
):
    valid_transaction.update(updates)
    with pytest.raises(InputValidationError, match=message):
        prepare_transaction(valid_transaction)


def test_prepare_transaction_maps_numeric_overflow_to_validation_error(valid_transaction):
    valid_transaction["sender_tx_count_before"] = 10**10_000
    with pytest.raises(InputValidationError, match="must be a number"):
        prepare_transaction(valid_transaction)


def test_missing_artifacts_leave_predictor_diagnostic_and_unready(tmp_path, valid_transaction):
    predictor = FraudPredictor(tmp_path / "missing")

    assert predictor.ready is False
    assert "Missing model artifact" in (predictor.load_error or "")
    with pytest.raises(ModelNotReadyError):
        predictor.predict(valid_transaction)


def test_predictor_loads_artifacts_and_applies_threshold(tmp_path, valid_transaction):
    artifact_dir = tmp_path / "artifacts"
    _write_artifacts(artifact_dir)

    predictor = FraudPredictor(artifact_dir)
    result = predictor.predict(valid_transaction)

    assert predictor.ready is True
    assert result.fraud_score == pytest.approx(0.86)
    assert result.send_for_review is True
    assert result.review_threshold == pytest.approx(0.20)
    assert result.model == "Test model"


def test_predictor_rejects_artifact_with_different_feature_contract(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    _write_artifacts(artifact_dir)
    metadata_path = artifact_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["features"] = ["amount"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    predictor = FraudPredictor(artifact_dir)

    assert predictor.ready is False
    assert "feature contract" in (predictor.load_error or "")


def test_failed_hot_reload_preserves_last_known_good_model(tmp_path, valid_transaction):
    artifact_dir = tmp_path / "artifacts"
    _write_artifacts(artifact_dir)
    predictor = FraudPredictor(artifact_dir)

    metadata_path = artifact_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["features"] = ["incompatible"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert predictor.reload() is False
    assert predictor.ready is True
    assert predictor.predict(valid_transaction).fraud_score == pytest.approx(0.86)
    assert "feature contract" in (predictor.load_error or "")


def test_predictor_rejects_recorded_runtime_version_mismatch(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    _write_artifacts(artifact_dir)
    metadata_path = artifact_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["library_versions"] = {"scikit_learn": "0.0.0"}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    predictor = FraudPredictor(artifact_dir)

    assert predictor.ready is False
    assert "library versions" in (predictor.load_error or "")


def test_predictor_rejects_model_checksum_mismatch(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    _write_artifacts(artifact_dir)
    with (artifact_dir / "fraud_model.joblib").open("ab") as model_file:
        model_file.write(b"changed")

    predictor = FraudPredictor(artifact_dir)

    assert predictor.ready is False
    assert "checksum" in (predictor.load_error or "")


def test_predictor_accepts_stale_marker_for_checksummed_generation(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    _write_artifacts(artifact_dir)
    (artifact_dir / ".publishing").write_text("in progress", encoding="utf-8")

    predictor = FraudPredictor(artifact_dir)

    assert predictor.ready is True


def test_predictor_rejects_marker_for_legacy_generation_without_checksum(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    _write_artifacts(artifact_dir)
    metadata_path = artifact_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("model_sha256")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (artifact_dir / ".publishing").write_text("in progress", encoding="utf-8")

    predictor = FraudPredictor(artifact_dir)

    assert predictor.ready is False
    assert "publication marker" in (predictor.load_error or "")
