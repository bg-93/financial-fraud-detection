"""Streamlit dashboard for scoring a single PaySim-style transaction."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st

from fraud_detection import (
    FraudPredictor,
    InputValidationError,
    ModelNotReadyError,
    PredictionError,
    PredictionResult,
    TRANSACTION_TYPES,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
# Increment whenever artifact-loading compatibility rules change. Streamlit's
# resource cache can otherwise retain an instance of the previous predictor
# class across a source-code rerun on managed hosting.
PREDICTOR_CACHE_VERSION = "python-major-compat-v2"


def _artifact_directory() -> Path:
    """Resolve the artifact directory without depending on the launch directory."""
    configured_path = os.getenv("FRAUD_ARTIFACT_DIR")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_ARTIFACT_DIR


@st.cache_resource(show_spinner="Loading fraud model…")
def _load_predictor(artifact_dir: str, cache_version: str) -> FraudPredictor:
    """Load the model once per Streamlit process."""
    del cache_version  # Its value is intentionally part of the Streamlit cache key.
    return FraudPredictor(artifact_dir=Path(artifact_dir))


def _render_model_status(
    predictor: FraudPredictor | None,
    artifact_dir: Path,
    load_error: str | None,
) -> None:
    """Show concise operational model information in the sidebar."""
    st.sidebar.header("Model status")

    if st.sidebar.button(
        "Reload model artifacts",
        use_container_width=True,
        help="Use this after retraining or replacing the saved model files.",
    ):
        if predictor is not None and predictor.reload():
            st.rerun()
        elif predictor is None:
            _load_predictor.clear()
            st.rerun()

    if predictor is not None and predictor.ready:
        st.sidebar.success("Ready")
        st.sidebar.metric("Model", predictor.model_name or "Saved pipeline")
        if predictor.threshold is not None:
            st.sidebar.metric("Review threshold", f"{predictor.threshold:.2%}")
        if predictor.load_error:
            st.sidebar.warning("Reload failed; continuing with the last known-good model.")
            st.sidebar.caption(predictor.load_error)
        if predictor.deployment_warning:
            st.sidebar.warning(predictor.deployment_warning)
    else:
        st.sidebar.error("Model unavailable")
        if load_error:
            st.sidebar.caption(load_error)

    st.sidebar.caption(f"Artifacts: {artifact_dir}")
    st.sidebar.divider()
    st.sidebar.caption(
        "The score prioritises transactions for review. It is not proof that a "
        "transaction is fraudulent."
    )


def _prediction_payload(
    transaction_type: str,
    amount: float,
    hour: int,
    sender_tx_count: int,
    sender_mean_amount: float,
    recipient_tx_count: int,
    pair_tx_count: int,
    hours_since_sender_tx: float,
) -> dict[str, str | float | int]:
    """Build the shared inference contract from dashboard values."""
    return {
        "type": transaction_type,
        "amount": float(amount),
        "hour": int(hour),
        "sender_tx_count_before": int(sender_tx_count),
        "sender_mean_amount_before": float(sender_mean_amount),
        "recipient_tx_count_before": int(recipient_tx_count),
        "pair_tx_count_before": int(pair_tx_count),
        "is_new_recipient": int(pair_tx_count == 0),
        "hours_since_sender_tx": float(hours_since_sender_tx),
    }


def _render_prediction(
    result: PredictionResult,
    payload: dict[str, str | float | int],
) -> None:
    """Render a prediction in both decision-oriented and machine-readable forms."""
    score = float(result.fraud_score)
    threshold = float(result.review_threshold)
    send_for_review = bool(result.send_for_review)

    st.subheader("Assessment")
    if send_for_review:
        st.error(
            "Send for manual review — the score meets or exceeds the configured "
            "review threshold."
        )
    else:
        st.success(
            "Below the manual-review threshold — continue with the organisation's "
            "normal controls."
        )

    score_column, threshold_column, decision_column = st.columns(3)
    score_column.metric("Fraud risk score", f"{score:.2%}")
    threshold_column.metric("Review threshold", f"{threshold:.2%}")
    decision_column.metric("Routing", "Manual review" if send_for_review else "Allow")

    bounded_score = min(max(score, 0.0), 1.0)
    st.progress(bounded_score, text=f"Risk score: {score:.2%}")
    st.caption(
        f"The decision boundary is {threshold:.2%}. This is an uncalibrated ranking "
        "score, not the estimated real-world likelihood of fraud. Interpret it with "
        "the validation metrics and review policy."
    )

    with st.expander("Prediction details"):
        details = result.to_dict()
        details["input"] = payload
        st.json(details)


def main() -> None:
    st.set_page_config(
        page_title="Payment Fraud Review",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("Payment Fraud Review")
    st.write(
        "Score one transaction using the saved fraud-detection pipeline and route "
        "high-risk activity to an analyst."
    )

    artifact_dir = _artifact_directory()
    predictor: FraudPredictor | None = None
    load_error: str | None = None

    try:
        predictor = _load_predictor(str(artifact_dir), PREDICTOR_CACHE_VERSION)
        # A cached predictor may have been created before training completed.
        # Retry lightweight artifact discovery on every rerun until it is ready.
        if not predictor.ready:
            predictor.reload()
        if not predictor.ready:
            load_error = predictor.load_error or "The model artifacts could not be loaded."
    except (ModelNotReadyError, PredictionError, OSError, ValueError) as exc:
        load_error = str(exc)
        LOGGER.warning("Fraud model is not ready: %s", exc)
    except Exception:
        load_error = "An unexpected error occurred while loading the model."
        LOGGER.exception("Unexpected fraud model loading error")

    _render_model_status(predictor, artifact_dir, load_error)

    if predictor is None or not predictor.ready:
        st.warning(
            "No deployable model is available yet. Run `python train_model.py` or "
            "execute every cell in `fraud_detection_notebook.ipynb` to create "
            "`artifacts/fraud_model.joblib` and `artifacts/model_metadata.json`, "
            "then use **Reload model artifacts**."
        )

    with st.form("transaction_form"):
        st.subheader("Transaction")
        transaction_type_column, amount_column, hour_column = st.columns(3)
        with transaction_type_column:
            transaction_type = st.selectbox(
                "Transaction type",
                TRANSACTION_TYPES,
                index=4,
                help="The PaySim payment category supplied to the model.",
            )
        with amount_column:
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=8_700.0,
                step=100.0,
                format="%.2f",
                help="Transaction amount in the same currency units used for training.",
            )
        with hour_column:
            hour = st.number_input(
                "Hour of day",
                min_value=0,
                max_value=23,
                value=2,
                step=1,
                help="Hour from 0 (midnight) through 23.",
            )

        st.subheader("Prior account activity")
        st.caption(
            "Use only activity observed before this transaction. A new recipient is "
            "derived automatically when the prior sender-recipient count is zero."
        )

        sender_count_column, sender_mean_column = st.columns(2)
        with sender_count_column:
            sender_tx_count = st.number_input(
                "Sender transactions before this one",
                min_value=0,
                value=12,
                step=1,
            )
        with sender_mean_column:
            sender_mean_amount = st.number_input(
                "Sender's prior mean amount",
                min_value=0.0,
                value=190.0,
                step=10.0,
                format="%.2f",
            )

        recipient_count_column, pair_count_column, recency_column = st.columns(3)
        with recipient_count_column:
            recipient_tx_count = st.number_input(
                "Recipient payments received before this one",
                min_value=0,
                value=2,
                step=1,
            )
        with pair_count_column:
            pair_tx_count = st.number_input(
                "Prior sender-recipient transactions",
                min_value=0,
                value=0,
                step=1,
            )
        with recency_column:
            hours_since_sender_tx = st.number_input(
                "Hours since sender's previous transaction",
                min_value=0.0,
                value=3.0,
                step=1.0,
                help="Use 10000 when the sender has no previous transaction.",
            )

        submitted = st.form_submit_button(
            "Score transaction",
            type="primary",
            use_container_width=True,
            disabled=predictor is None or not predictor.ready,
        )

    if not submitted or predictor is None:
        return

    payload = _prediction_payload(
        transaction_type=transaction_type,
        amount=amount,
        hour=hour,
        sender_tx_count=sender_tx_count,
        sender_mean_amount=sender_mean_amount,
        recipient_tx_count=recipient_tx_count,
        pair_tx_count=pair_tx_count,
        hours_since_sender_tx=hours_since_sender_tx,
    )

    try:
        with st.spinner("Scoring transaction…"):
            result = predictor.predict(payload)
    except InputValidationError as exc:
        st.warning(f"Check the transaction values: {exc}")
        return
    except ModelNotReadyError as exc:
        st.error(f"The model is not ready: {exc}")
        return
    except PredictionError:
        LOGGER.exception("Fraud prediction failed")
        st.error("The transaction could not be scored. Check the server logs and try again.")
        return
    except Exception:
        LOGGER.exception("Unexpected fraud prediction error")
        st.error("An unexpected error occurred while scoring the transaction.")
        return

    _render_prediction(result, payload)


if __name__ == "__main__":
    main()
