import json

import pandas as pd
import pytest

from fraud_detection import FraudPredictor
from train_model import DEMO_DATA_PATH, load_and_validate_data, parse_args, run_training


def test_training_normalizes_supported_transaction_types(tmp_path):
    data = pd.read_csv(DEMO_DATA_PATH)
    data.loc[0, "type"] = " transfer "
    data_path = tmp_path / "normalized.csv"
    data.to_csv(data_path, index=False)

    loaded = load_and_validate_data(data_path)

    assert loaded.loc[0, "type"] == "TRANSFER"


def test_training_rejects_unknown_transaction_types(tmp_path):
    data = pd.read_csv(DEMO_DATA_PATH)
    data.loc[0, "type"] = "CRYPTO"
    data_path = tmp_path / "unknown.csv"
    data.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="unsupported transaction categories"):
        load_and_validate_data(data_path)


def test_demo_training_artifact_loads_and_scores(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    args = parse_args(
        [
            "--use-demo",
            "--skip-xgboost",
            "--artifacts-dir",
            str(artifact_dir),
        ]
    )

    outputs = run_training(args)
    predictor = FraudPredictor(artifact_dir)
    result = predictor.predict(
        {
            "type": "TRANSFER",
            "amount": 8_700,
            "hour": 2,
            "sender_tx_count_before": 12,
            "sender_mean_amount_before": 190,
            "recipient_tx_count_before": 2,
            "pair_tx_count_before": 0,
            "is_new_recipient": 1,
            "hours_since_sender_tx": 3,
        }
    )

    assert {path.name for path in outputs} == {
        "fraud_model.joblib",
        "validation_results.csv",
        "monitoring_over_time.csv",
        "model_metadata.json",
    }
    assert predictor.ready is True
    metadata = json.loads(
        (artifact_dir / "model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["demo_data"] is True
    assert len(metadata["model_sha256"]) == 64
    assert "precision_at_review_capacity" in metadata["test_metrics"]
    assert not (artifact_dir / ".publishing").exists()
    assert predictor.deployment_warning
    assert 0 <= result.fraud_score <= 1
