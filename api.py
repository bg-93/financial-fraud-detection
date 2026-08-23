"""HTTP API for serving fraud predictions.

Run locally with::

    uvicorn api:app --host 0.0.0.0 --port 8000

The model and metadata are loaded from ``FRAUD_ARTIFACT_DIR`` (``artifacts`` by
default). The process still starts if those artifacts are unavailable so that
``/health`` can explain the service state; prediction requests return HTTP 503
when loading fails. Supply valid artifacts and restart the API process to retry.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fraud_detection import (
    FraudPredictor,
    InputValidationError,
    ModelNotReadyError,
    PredictionError,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
HTTP_UNPROCESSABLE_ENTITY = 422


class TransactionRequest(BaseModel):
    """Features available immediately before a transaction is evaluated."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "type": "TRANSFER",
                "amount": 8700.0,
                "hour": 2,
                "sender_tx_count_before": 12,
                "sender_mean_amount_before": 190.0,
                "recipient_tx_count_before": 2,
                "pair_tx_count_before": 0,
                "is_new_recipient": 1,
                "hours_since_sender_tx": 3.0,
            }
        },
    )

    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(
        description="PaySim transaction category."
    )
    amount: float = Field(ge=0, description="Transaction value.")
    hour: int = Field(ge=0, le=23, description="Hour of day, from 0 to 23.")
    sender_tx_count_before: int = Field(
        ge=0, description="Number of earlier transactions from this sender."
    )
    sender_mean_amount_before: float = Field(
        ge=0, description="Mean value of the sender's earlier transactions."
    )
    recipient_tx_count_before: int = Field(
        ge=0, description="Number of earlier transactions received by this recipient."
    )
    pair_tx_count_before: int = Field(
        ge=0, description="Number of earlier transactions between this sender and recipient."
    )
    is_new_recipient: Literal[0, 1] = Field(
        description="1 when the sender has not previously paid this recipient."
    )
    hours_since_sender_tx: float = Field(
        ge=0, description="Hours since the sender's previous transaction."
    )


class PredictionResponse(BaseModel):
    fraud_score: float = Field(
        ge=0,
        le=1,
        description="Uncalibrated ranking score; do not interpret as a real-world probability.",
    )
    send_for_review: bool
    review_threshold: float = Field(ge=0, le=1)
    model: str


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model: str | None = None
    detail: str
    warning: str | None = None


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    detail: str | list[ValidationIssue]


def _artifact_dir() -> Path:
    configured_dir = os.getenv("FRAUD_ARTIFACT_DIR")
    return Path(configured_dir or DEFAULT_ARTIFACT_DIR).expanduser()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts once per server process without hiding startup failures."""

    artifact_dir = _artifact_dir()
    try:
        app.state.predictor = FraudPredictor(artifact_dir=artifact_dir)
        app.state.startup_error = None
        if not app.state.predictor.ready:
            LOGGER.error(
                "Fraud model is not ready: %s",
                app.state.predictor.load_error or "unknown artifact error",
            )
        elif app.state.predictor.deployment_warning:
            LOGGER.warning("%s", app.state.predictor.deployment_warning)
    except Exception as exc:  # The health endpoint reports a safe 503 response.
        app.state.predictor = None
        app.state.startup_error = str(exc)
        LOGGER.exception("Could not initialise fraud model from %s", artifact_dir)

    yield


app = FastAPI(
    title="Payment Fraud Detection API",
    description=(
        "Scores PaySim-style transactions using the model trained by the project notebook."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_ready_predictor(request: Request) -> FraudPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Fraud model is unavailable. Run the notebook to generate the artifacts "
                "or mount them into FRAUD_ARTIFACT_DIR, then restart the service."
            ),
        )
    return predictor


@app.get("/", include_in_schema=False)
def service_info() -> dict[str, str]:
    return {
        "service": "Payment Fraud Detection API",
        "health": "/health",
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
def health(request: Request) -> HealthResponse | JSONResponse:
    """Report whether the model is loaded and predictions can be served."""

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is not None and predictor.ready:
        return HealthResponse(
            status="ready",
            model=predictor.model_name,
            detail="Fraud model is loaded and ready for predictions.",
            warning=predictor.deployment_warning,
        )

    payload = HealthResponse(
        status="not_ready",
        model=None,
        detail=(
            "Fraud model artifacts are unavailable or invalid. Generate them with the "
            "notebook or mount them into FRAUD_ARTIFACT_DIR."
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        HTTP_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
def predict(
    transaction: TransactionRequest,
    predictor: FraudPredictor = Depends(get_ready_predictor),
) -> PredictionResponse:
    """Return the fraud score and the notebook-selected review decision."""

    try:
        result = predictor.predict(transaction.model_dump())
        return PredictionResponse.model_validate(result.to_dict())
    except InputValidationError as exc:
        raise HTTPException(
            status_code=HTTP_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud model became unavailable while processing the request.",
        ) from exc
    except PredictionError as exc:
        LOGGER.exception("Fraud prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The transaction could not be scored.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected fraud prediction failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The transaction could not be scored.",
        ) from exc
