"""Shared, framework-independent inference code for the fraud model.

Both the Streamlit dashboard and FastAPI service use :class:`FraudPredictor` so
that artifact loading, feature ordering, and decision-threshold behavior cannot
drift between the two deployment surfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
import math
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Mapping
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

# This order is the contract used when the notebook fits the sklearn pipeline.
MODEL_FEATURES = (
    "amount",
    "log_amount",
    "hour",
    "sender_tx_count_before",
    "sender_mean_amount_before",
    "amount_vs_sender_mean",
    "recipient_tx_count_before",
    "pair_tx_count_before",
    "is_new_recipient",
    "hours_since_sender_tx",
    "type",
)

REQUEST_FEATURES = tuple(
    feature
    for feature in MODEL_FEATURES
    if feature not in {"log_amount", "amount_vs_sender_mean"}
)


class FraudDetectionError(RuntimeError):
    """Base exception for deployment-time fraud detection failures."""


class ModelNotReadyError(FraudDetectionError):
    """Raised when inference is requested before valid artifacts are loaded."""


class InputValidationError(FraudDetectionError, ValueError):
    """Raised when transaction features do not satisfy the model contract."""


class PredictionError(FraudDetectionError):
    """Raised when the fitted pipeline cannot score a valid feature row."""


@dataclass(frozen=True)
class PredictionResult:
    """Serializable result returned by the shared predictor."""

    fraud_score: float
    send_for_review: bool
    review_threshold: float
    model: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def _finite_number(
    transaction: Mapping[str, Any],
    name: str,
    *,
    minimum: float = 0.0,
) -> float:
    if name not in transaction:
        raise InputValidationError(f"Missing required field: {name}")

    value = transaction[name]
    if isinstance(value, bool):
        raise InputValidationError(f"{name} must be a number, not a boolean")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{name} must be a number") from exc

    if not math.isfinite(number):
        raise InputValidationError(f"{name} must be finite")
    if number < minimum:
        raise InputValidationError(f"{name} must be at least {minimum:g}")
    return number


def _non_negative_integer(transaction: Mapping[str, Any], name: str) -> int:
    number = _finite_number(transaction, name)
    if not number.is_integer():
        raise InputValidationError(f"{name} must be a whole number")
    return int(number)


def prepare_transaction(transaction: Mapping[str, Any]) -> pd.DataFrame:
    """Validate one request and create the exact row expected by the pipeline.

    Historical aggregates are supplied by the caller because the deployed demo
    is stateless. ``log_amount`` and ``amount_vs_sender_mean`` are deterministic
    and are always derived here to keep them consistent with training.
    """

    if not isinstance(transaction, Mapping):
        raise InputValidationError("Transaction must be a mapping of feature names to values")

    raw_type = transaction.get("type")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise InputValidationError("type must be a non-empty string")
    transaction_type = raw_type.strip().upper()
    if transaction_type not in TRANSACTION_TYPES:
        allowed = ", ".join(TRANSACTION_TYPES)
        raise InputValidationError(f"type must be one of: {allowed}")

    amount = _finite_number(transaction, "amount")
    hour = _non_negative_integer(transaction, "hour")
    if hour > 23:
        raise InputValidationError("hour must be between 0 and 23")

    sender_count = _non_negative_integer(transaction, "sender_tx_count_before")
    sender_mean = _finite_number(transaction, "sender_mean_amount_before")
    amount_ratio = amount / sender_mean if sender_mean > 0 else 1.0
    amount_ratio = min(amount_ratio, 100.0)

    recipient_count = _non_negative_integer(transaction, "recipient_tx_count_before")
    pair_count = _non_negative_integer(transaction, "pair_tx_count_before")
    is_new_recipient = _non_negative_integer(transaction, "is_new_recipient")
    if is_new_recipient not in (0, 1):
        raise InputValidationError("is_new_recipient must be either 0 or 1")
    if (pair_count == 0) != (is_new_recipient == 1):
        raise InputValidationError(
            "is_new_recipient must be 1 when pair_tx_count_before is 0, and 0 otherwise"
        )

    hours_since = _finite_number(transaction, "hours_since_sender_tx")
    if pair_count > sender_count:
        raise InputValidationError(
            "pair_tx_count_before cannot exceed sender_tx_count_before"
        )
    if pair_count > recipient_count:
        raise InputValidationError(
            "pair_tx_count_before cannot exceed recipient_tx_count_before"
        )
    if sender_count == 0 and sender_mean != 0:
        raise InputValidationError(
            "sender_mean_amount_before must be 0 when the sender has no prior transactions"
        )
    if sender_count == 0 and hours_since != 10_000:
        raise InputValidationError(
            "hours_since_sender_tx must be 10000 when the sender has no prior transactions"
        )

    row = {
        "amount": amount,
        "log_amount": math.log1p(amount),
        "hour": hour,
        "sender_tx_count_before": sender_count,
        "sender_mean_amount_before": sender_mean,
        "amount_vs_sender_mean": amount_ratio,
        "recipient_tx_count_before": recipient_count,
        "pair_tx_count_before": pair_count,
        "is_new_recipient": is_new_recipient,
        "hours_since_sender_tx": hours_since,
        "type": transaction_type,
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def _validate_recorded_versions(metadata: Mapping[str, Any]) -> None:
    """Reject known-incompatible persisted-model runtime versions."""

    recorded = metadata.get("library_versions")
    if not isinstance(recorded, Mapping):
        return

    runtime_versions = {
        "joblib": getattr(joblib, "__version__", None),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    mismatches = [
        f"{name} trained={recorded[name]} runtime={runtime}"
        for name, runtime in runtime_versions.items()
        if recorded.get(name) and runtime and recorded[name] != runtime
    ]

    if recorded.get("python"):
        trained_python = ".".join(str(recorded["python"]).split(".")[:2])
        runtime_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if trained_python != runtime_python:
            mismatches.append(
                f"python trained={trained_python} runtime={runtime_python}"
            )

    if metadata.get("model") == "XGBoost" and recorded.get("xgboost"):
        installed_xgboost = None
        for distribution in ("xgboost", "xgboost-cpu"):
            try:
                installed_xgboost = distribution_version(distribution)
                break
            except PackageNotFoundError:
                continue
        if recorded["xgboost"] != installed_xgboost:
            mismatches.append(
                "xgboost "
                f"trained={recorded['xgboost']} runtime={installed_xgboost or 'missing'}"
            )

    if mismatches:
        raise ValueError(
            "artifact library versions do not match the serving runtime: "
            + "; ".join(mismatches)
        )


class FraudPredictor:
    """Load the exported sklearn pipeline and score individual transactions.

    Construction is intentionally non-fatal when artifacts are missing. This
    lets the API expose a useful liveness/readiness response instead of crashing
    its process before diagnostics can be queried.
    """

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        configured_dir = artifact_dir or os.getenv("FRAUD_ARTIFACT_DIR")
        self.artifact_dir = Path(configured_dir or DEFAULT_ARTIFACT_DIR).expanduser().resolve()
        self._lock = RLock()
        self._model: Any | None = None
        self._metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self.reload()

    @property
    def ready(self) -> bool:
        with self._lock:
            return (
                self._model is not None
                and self._metadata.get("review_threshold") is not None
            )

    @property
    def model_name(self) -> str | None:
        with self._lock:
            value = self._metadata.get("model")
        return str(value) if value is not None else None

    @property
    def threshold(self) -> float | None:
        with self._lock:
            value = self._metadata.get("review_threshold")
        return float(value) if value is not None else None

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._metadata)

    @property
    def deployment_warning(self) -> str | None:
        with self._lock:
            warning = self._metadata.get("deployment_warning")
            demo_data = self._metadata.get("demo_data")
        if warning:
            return str(warning)
        if demo_data is True:
            return "This model was trained on demo data and is not suitable for production."
        return None

    def reload(self) -> bool:
        """Atomically reload model artifacts, retaining a diagnostic on failure."""

        model_path = self.artifact_dir / "fraud_model.joblib"
        metadata_path = self.artifact_dir / "model_metadata.json"
        publication_marker = self.artifact_dir / ".publishing"

        missing = [path.name for path in (model_path, metadata_path) if not path.is_file()]
        if missing:
            with self._lock:
                self.load_error = (
                    f"Missing model artifact(s) in {self.artifact_dir}: {', '.join(missing)}. "
                    "Run train_model.py or execute the notebook before starting deployment."
                )
            return False

        try:
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            if not isinstance(metadata, dict):
                raise ValueError("model_metadata.json must contain a JSON object")

            threshold = float(metadata["review_threshold"])
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("review_threshold must be a finite value between 0 and 1")

            artifact_features = metadata.get("features")
            if artifact_features != list(MODEL_FEATURES):
                raise ValueError(
                    "artifact feature contract does not match this deployment version"
                )

            _validate_recorded_versions(metadata)
            with model_path.open("rb") as model_file:
                digest = hashlib.sha256()
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
                recorded_digest = metadata.get("model_sha256")
                if recorded_digest and not hmac.compare_digest(
                    str(recorded_digest), digest.hexdigest()
                ):
                    raise ValueError("fraud_model.joblib checksum does not match metadata")
                if publication_marker.exists() and not recorded_digest:
                    raise ValueError(
                        "an incomplete artifact publication marker exists and the "
                        "legacy metadata has no model checksum"
                    )
                model_file.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    model = joblib.load(model_file)
            if not callable(getattr(model, "predict_proba", None)):
                raise ValueError("fraud_model.joblib does not expose predict_proba")
            classes = np.asarray(getattr(model, "classes_", []))
            if classes.shape != (2,) or not np.array_equal(classes, np.array([0, 1])):
                raise ValueError("fraud_model.joblib must use binary classes ordered as [0, 1]")
            fitted_features = getattr(model, "feature_names_in_", None)
            if fitted_features is not None and list(fitted_features) != list(MODEL_FEATURES):
                raise ValueError("fitted model feature names do not match metadata")
        except Exception as exc:  # Artifact/library incompatibilities vary by backend.
            with self._lock:
                self.load_error = f"Could not load model artifacts: {exc}"
            return False

        metadata["review_threshold"] = threshold
        with self._lock:
            self._metadata = metadata
            self._model = model
            self.load_error = None
        return True

    def predict(self, transaction: Mapping[str, Any]) -> PredictionResult:
        with self._lock:
            model = self._model
            metadata = dict(self._metadata)
            load_error = self.load_error
        if model is None or metadata.get("review_threshold") is None:
            raise ModelNotReadyError(load_error or "Fraud model is not ready")

        features = prepare_transaction(transaction)
        try:
            probabilities = np.asarray(model.predict_proba(features))
            if probabilities.ndim != 2 or probabilities.shape != (1, 2):
                raise ValueError(
                    f"expected predict_proba shape (1, 2), received {probabilities.shape}"
                )
            score = float(probabilities[0, 1])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("model returned an invalid fraud score")
        except Exception as exc:
            raise PredictionError(f"The model could not score this transaction: {exc}") from exc

        threshold = float(metadata["review_threshold"])

        return PredictionResult(
            fraud_score=score,
            send_for_review=score >= threshold,
            review_threshold=threshold,
            model=str(metadata.get("model") or "Unknown"),
        )


__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "MODEL_FEATURES",
    "REQUEST_FEATURES",
    "TRANSACTION_TYPES",
    "FraudPredictor",
    "InputValidationError",
    "ModelNotReadyError",
    "PredictionError",
    "PredictionResult",
    "prepare_transaction",
]
