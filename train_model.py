#!/usr/bin/env python3
"""Train and export the PaySim fraud model used by the deployment apps.

The implementation mirrors ``fraud_detection_notebook.ipynb``: it creates
history-only features, splits transactions chronologically, compares the
notebook's candidate models using validation PR-AUC, chooses a review threshold
from validation scores, and evaluates once on the held-out test period.

The real PaySim file is required by default. The bundled demo data is used only
when ``--use-demo`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from fraud_detection import TRANSACTION_TYPES
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:  # Keep the error more useful than a long import traceback.
    raise SystemExit(
        "Training dependencies are missing. Install them with "
        "`pip install -r requirements.txt` before running this script. "
        f"Original import error: {exc}"
    ) from exc

XGBOOST_IMPORT_ERROR: Exception | None = None
try:
    import xgboost
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as exc:  # Native OpenMP loading can fail after import starts.
    xgboost = None
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = exc


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "PaySim.csv"
DEMO_DATA_PATH = PROJECT_DIR / "demo_paysim.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "artifacts"

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
}

NUMERIC_FEATURES = [
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
]
CATEGORICAL_FEATURES = ["type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MIN_TRANSACTIONS = 30
MIN_TIME_STEPS = 4
TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85


def fraction(value: str) -> float:
    """Argparse type for a fraction in the interval (0, 1]."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def non_negative_int(value: str) -> int:
    """Argparse type for a non-negative integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the fraud detector and write a fitted pipeline, metadata, "
            "validation results, and monitoring results."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "PaySim CSV to train on. Relative paths are resolved from the current "
            "directory. Defaults to data/PaySim.csv beside this script."
        ),
    )
    source.add_argument(
        "--use-demo",
        action="store_true",
        help=(
            "Explicitly train on the bundled demo_paysim.csv for a smoke test. "
            "Demo artifacts are not suitable for a real deployment."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Output directory (default: artifacts/ beside this script).",
    )
    parser.add_argument(
        "--random-state",
        type=non_negative_int,
        default=42,
        help="Random seed used by all candidate models (default: 42).",
    )
    parser.add_argument(
        "--review-capacity",
        type=fraction,
        default=0.01,
        help=(
            "Fraction of transactions the fraud team can review, expressed as a "
            "number in (0, 1] (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Compare only Logistic Regression and Random Forest.",
    )
    return parser.parse_args(argv)


def load_and_validate_data(path: Path) -> pd.DataFrame:
    """Read PaySim data and reject inputs that cannot support valid training."""

    if not path.exists():
        default_hint = (
            " Download PaySim to data/PaySim.csv, pass --data PATH, or use "
            "--use-demo for a non-production smoke test."
            if path == DEFAULT_DATA_PATH
            else ""
        )
        raise FileNotFoundError(f"Training data was not found: {path}.{default_hint}")
    if not path.is_file():
        raise ValueError(f"Training data path is not a file: {path}")

    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read training CSV {path}: {exc}") from exc

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(data) < MIN_TRANSACTIONS:
        raise ValueError(
            f"Dataset has {len(data):,} rows; at least {MIN_TRANSACTIONS:,} are "
            "required for chronological train/validation/test splits."
        )

    for column in ("step", "amount", "isFraud"):
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted)
        if invalid.any():
            example_rows = data.index[invalid].tolist()[:5]
            raise ValueError(
                f"Column {column!r} contains missing or non-numeric values "
                f"(example row indices: {example_rows})."
            )
        data[column] = converted

    if (data["step"] < 0).any() or not np.equal(
        data["step"], np.floor(data["step"])
    ).all():
        raise ValueError("Column 'step' must contain non-negative integer time steps.")
    data["step"] = data["step"].astype(np.int64)

    if (data["amount"] < 0).any():
        raise ValueError("Column 'amount' must not contain negative values.")

    if not np.equal(data["isFraud"], np.floor(data["isFraud"])).all():
        raise ValueError("Column 'isFraud' must contain only binary values 0 and 1.")
    target_values = set(data["isFraud"].unique())
    if not target_values <= {0, 1}:
        raise ValueError(
            "Column 'isFraud' must contain only binary values 0 and 1; "
            f"found {sorted(target_values)}."
        )
    if target_values != {0, 1}:
        raise ValueError("Training data must contain both legitimate and fraud rows.")
    data["isFraud"] = data["isFraud"].astype(np.int8)

    for column in ("type", "nameOrig", "nameDest"):
        invalid = data[column].isna() | data[column].astype(str).str.strip().eq("")
        if invalid.any():
            example_rows = data.index[invalid].tolist()[:5]
            raise ValueError(
                f"Column {column!r} contains missing or empty values "
                f"(example row indices: {example_rows})."
            )
        data[column] = data[column].astype(str)

    data["type"] = data["type"].str.strip().str.upper()
    unknown_types = sorted(set(data["type"]) - set(TRANSACTION_TYPES))
    if unknown_types:
        raise ValueError(
            "Column 'type' contains unsupported transaction categories: "
            f"{unknown_types}. Expected one of {list(TRANSACTION_TYPES)}."
        )

    unique_steps = data["step"].nunique()
    if unique_steps < MIN_TIME_STEPS:
        raise ValueError(
            f"Dataset has {unique_steps} unique time steps; at least "
            f"{MIN_TIME_STEPS} are required for chronological splits."
        )

    return data.sort_values("step", kind="stable").reset_index(drop=True)


def build_history_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features using only transactions from strictly earlier steps."""

    featured = data.copy()
    featured["log_amount"] = np.log1p(featured["amount"].clip(lower=0))
    featured["hour"] = featured["step"] % 24

    sender_step = (
        featured.groupby(["nameOrig", "step"], as_index=False)
        .agg(step_count=("amount", "size"), step_amount=("amount", "sum"))
        .sort_values(["nameOrig", "step"])
    )
    sender_group = sender_step.groupby("nameOrig")
    sender_step["sender_tx_count_before"] = (
        sender_group["step_count"].cumsum() - sender_step["step_count"]
    )
    sender_step["sender_amount_sum_before"] = (
        sender_group["step_amount"].cumsum() - sender_step["step_amount"]
    )
    sender_step["sender_previous_step"] = sender_group["step"].shift(1)
    sender_step["sender_mean_amount_before"] = np.where(
        sender_step["sender_tx_count_before"] > 0,
        sender_step["sender_amount_sum_before"]
        / sender_step["sender_tx_count_before"],
        0.0,
    )
    sender_step["hours_since_sender_tx"] = (
        sender_step["step"] - sender_step["sender_previous_step"]
    ).fillna(10_000)

    recipient_step = (
        featured.groupby(["nameDest", "step"], as_index=False)
        .size()
        .rename(columns={"size": "step_count"})
        .sort_values(["nameDest", "step"])
    )
    recipient_step["recipient_tx_count_before"] = (
        recipient_step.groupby("nameDest")["step_count"].cumsum()
        - recipient_step["step_count"]
    )

    pair_step = (
        featured.groupby(["nameOrig", "nameDest", "step"], as_index=False)
        .size()
        .rename(columns={"size": "step_count"})
        .sort_values(["nameOrig", "nameDest", "step"])
    )
    pair_step["pair_tx_count_before"] = (
        pair_step.groupby(["nameOrig", "nameDest"])["step_count"].cumsum()
        - pair_step["step_count"]
    )

    featured = featured.merge(
        sender_step[
            [
                "nameOrig",
                "step",
                "sender_tx_count_before",
                "sender_mean_amount_before",
                "hours_since_sender_tx",
            ]
        ],
        on=["nameOrig", "step"],
        how="left",
    )
    featured = featured.merge(
        recipient_step[
            ["nameDest", "step", "recipient_tx_count_before"]
        ],
        on=["nameDest", "step"],
        how="left",
    )
    featured = featured.merge(
        pair_step[
            ["nameOrig", "nameDest", "step", "pair_tx_count_before"]
        ],
        on=["nameOrig", "nameDest", "step"],
        how="left",
    )

    featured["is_new_recipient"] = (
        featured["pair_tx_count_before"] == 0
    ).astype(int)
    featured["amount_vs_sender_mean"] = np.where(
        featured["sender_mean_amount_before"] > 0,
        featured["amount"] / featured["sender_mean_amount_before"],
        1.0,
    )
    featured["amount_vs_sender_mean"] = featured[
        "amount_vs_sender_mean"
    ].clip(upper=100)

    invalid_features = ~np.isfinite(featured[NUMERIC_FEATURES].to_numpy(dtype=float))
    if invalid_features.any():
        raise ValueError(
            "Feature engineering produced missing or non-finite numeric values. "
            "Check the source amounts and time steps."
        )
    return featured


def chronological_split(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """Use the first 70%, next 15%, and final 15% of distinct time steps."""

    steps = np.sort(features["step"].unique())
    train_index = int(len(steps) * TRAIN_FRACTION)
    validation_end_index = int(len(steps) * VALIDATION_END_FRACTION)

    if not 0 < train_index < validation_end_index < len(steps):
        raise ValueError(
            "The distinct time steps cannot form non-empty 70/15/15 splits. "
            "Provide data covering more time steps."
        )

    train_end = int(steps[train_index])
    validation_end = int(steps[validation_end_index])
    train = features[features["step"] < train_end].copy()
    valid = features[
        (features["step"] >= train_end)
        & (features["step"] < validation_end)
    ].copy()
    test = features[features["step"] >= validation_end].copy()

    for name, split in (("training", train), ("validation", valid), ("test", test)):
        if split.empty:
            raise ValueError(f"The chronological {name} split is empty.")
        counts = split["isFraud"].value_counts()
        if not {0, 1} <= set(counts.index):
            raise ValueError(
                f"The chronological {name} split must contain at least one fraud "
                "and one legitimate transaction. Collect a longer time period or "
                "adjust the source data."
            )

    return train, valid, test, train_end, validation_end


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer: Any = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models(
    train_labels: pd.Series,
    *,
    random_state: int,
    include_xgboost: bool,
) -> dict[str, Pipeline]:
    """Create fresh preprocessing for each notebook candidate model."""

    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    class_ratio = negative / positive

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if include_xgboost:
        if not XGBOOST_AVAILABLE:
            import_detail = (
                str(XGBOOST_IMPORT_ERROR).strip().splitlines()[0]
                if XGBOOST_IMPORT_ERROR
                else "unknown import error"
            )
            print(
                "Warning: xgboost is unavailable; continuing with Logistic "
                "Regression and Random Forest. "
                f"Import error: {import_detail}",
                file=sys.stderr,
            )
        else:
            assert XGBClassifier is not None
            models["XGBoost"] = Pipeline(
                [
                    ("preprocess", make_preprocessor(scale_numeric=False)),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=250,
                            max_depth=5,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            scale_pos_weight=class_ratio,
                            eval_metric="aucpr",
                            tree_method="hist",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
    return models


def review_metrics(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    amounts: pd.Series | np.ndarray,
    *,
    capacity: float,
) -> dict[str, float]:
    """Evaluate the highest-scored transactions under the review constraint."""

    y_array = np.asarray(y_true)
    score_array = np.asarray(scores)
    amount_array = np.asarray(amounts)
    if len(y_array) == 0 or not (
        len(y_array) == len(score_array) == len(amount_array)
    ):
        raise ValueError("Metric inputs must be non-empty and have equal lengths.")
    if not np.isfinite(score_array).all():
        raise ValueError("The model produced non-finite probability scores.")

    review_count = max(1, math.ceil(len(score_array) * capacity))
    review_index = np.argsort(score_array)[::-1][:review_count]
    frauds_found = int(y_array[review_index].sum())
    total_frauds = int(y_array.sum())
    if total_frauds == 0:
        raise ValueError("Review metrics require at least one fraud transaction.")

    fraud_value_total = float(amount_array[y_array == 1].sum())
    fraud_value_found = float(
        amount_array[review_index][y_array[review_index] == 1].sum()
    )

    return {
        "precision_at_review_capacity": frauds_found / review_count,
        "recall_at_review_capacity": frauds_found / total_frauds,
        "fraud_value_capture_at_review_capacity": (
            fraud_value_found / fraud_value_total if fraud_value_total > 0 else 0.0
        ),
    }


def evaluate_model(
    y_true: pd.Series,
    scores: np.ndarray,
    amounts: pd.Series,
    *,
    capacity: float,
) -> dict[str, float]:
    result = {"pr_auc": float(average_precision_score(y_true, scores))}
    result.update(review_metrics(y_true, scores, amounts, capacity=capacity))
    return result


def build_monitoring(
    test: pd.DataFrame,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    monitor = test[["step", "isFraud", "amount"]].copy()
    monitor["score"] = scores
    monitor["prediction"] = predictions
    number_of_blocks = min(5, int(monitor["step"].nunique()))
    monitor["time_block"] = pd.qcut(
        monitor["step"].rank(method="first"),
        q=number_of_blocks,
        labels=False,
        duplicates="drop",
    )

    rows: list[dict[str, int | float]] = []
    for block, part in monitor.groupby("time_block", observed=True):
        rows.append(
            {
                "time_block": int(block),
                "transactions": int(len(part)),
                "fraud_rate": float(part["isFraud"].mean()),
                "average_model_score": float(part["score"].mean()),
                "recall": float(
                    recall_score(
                        part["isFraud"], part["prediction"], zero_division=0
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def library_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "joblib": getattr(joblib, "__version__", None),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": getattr(xgboost, "__version__", None),
    }


def split_record(name: str, split: pd.DataFrame) -> dict[str, int | float | str]:
    return {
        "split": name,
        "transactions": int(len(split)),
        "frauds": int(split["isFraud"].sum()),
        "fraud_rate": float(split["isFraud"].mean()),
        "first_step": int(split["step"].min()),
        "last_step": int(split["step"].max()),
    }


def save_artifacts(
    *,
    artifact_dir: Path,
    model: Pipeline,
    validation_results: pd.DataFrame,
    monitoring: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[Path]:
    if artifact_dir.exists() and not artifact_dir.is_dir():
        raise ValueError(f"Artifact output path is not a directory: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "fraud_model.joblib"
    validation_path = artifact_dir / "validation_results.csv"
    monitoring_path = artifact_dir / "monitoring_over_time.csv"
    metadata_path = artifact_dir / "model_metadata.json"

    # Metadata is promoted last and contains the model digest. A failed publish
    # therefore becomes unavailable rather than combining a new model with an
    # old review threshold. The marker also protects first-time/legacy updates.
    publication_marker = artifact_dir / ".publishing"
    publication_marker.write_text(
        f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    published = False
    try:
        with tempfile.TemporaryDirectory(
            dir=artifact_dir, prefix=".artifact-staging-"
        ) as staging_name:
            staging_dir = Path(staging_name)
            staged_model = staging_dir / model_path.name
            staged_validation = staging_dir / validation_path.name
            staged_monitoring = staging_dir / monitoring_path.name
            staged_metadata = staging_dir / metadata_path.name

            joblib.dump(model, staged_model)
            validation_results.to_csv(staged_validation, index=False)
            monitoring.to_csv(staged_monitoring, index=False)

            digest = hashlib.sha256()
            with staged_model.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            published_metadata = dict(metadata)
            published_metadata["model_sha256"] = digest.hexdigest()
            with staged_metadata.open("w", encoding="utf-8") as file:
                json.dump(published_metadata, file, indent=2, allow_nan=False)
                file.write("\n")

            for staged_path, final_path in (
                (staged_model, model_path),
                (staged_validation, validation_path),
                (staged_monitoring, monitoring_path),
                (staged_metadata, metadata_path),
            ):
                staged_path.replace(final_path)
            published = True
    finally:
        if published:
            publication_marker.unlink(missing_ok=True)

    return [model_path, validation_path, monitoring_path, metadata_path]


def run_training(args: argparse.Namespace) -> list[Path]:
    data_path = DEMO_DATA_PATH if args.use_demo else args.data
    data_path = data_path.expanduser()
    artifact_dir = args.artifacts_dir.expanduser()
    is_demo = data_path.resolve() == DEMO_DATA_PATH.resolve()

    if is_demo:
        print(
            "WARNING: Using demo_paysim.csv. The resulting model is for local "
            "integration testing only and must not be treated as a production "
            "fraud model.",
            file=sys.stderr,
        )

    print(f"Loading transactions from {data_path}")
    data = load_and_validate_data(data_path)
    print(
        f"Loaded {len(data):,} transactions across {data['step'].nunique():,} "
        f"time steps; fraud rate={data['isFraud'].mean():.4%}"
    )

    print("Building leakage-safe historical features")
    features = build_history_features(data)
    train, valid, test, train_end, validation_end = chronological_split(features)
    split_rows = [
        split_record("train", train),
        split_record("validation", valid),
        split_record("test", test),
    ]
    for row in split_rows:
        print(
            f"{str(row['split']).capitalize()}: {row['transactions']:,} rows, "
            f"{row['frauds']:,} frauds, steps {row['first_step']}-"
            f"{row['last_step']}"
        )

    x_train = train[MODEL_FEATURES]
    y_train = train["isFraud"]
    x_valid = valid[MODEL_FEATURES]
    y_valid = valid["isFraud"]

    models = build_models(
        y_train,
        random_state=args.random_state,
        include_xgboost=not args.skip_xgboost,
    )
    validation_rows: list[dict[str, str | float]] = []
    trained_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        print(f"Training {name}")
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_valid)[:, 1]
        result = evaluate_model(
            y_valid,
            scores,
            valid["amount"],
            capacity=args.review_capacity,
        )
        validation_rows.append({"model": name, **result})
        trained_models[name] = model
        print(f"  validation PR-AUC={result['pr_auc']:.6f}")

    validation_results = (
        pd.DataFrame(validation_rows)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    if validation_results.empty:
        raise RuntimeError("No candidate model was trained.")

    best_model_name = str(validation_results.loc[0, "model"])
    best_model = trained_models[best_model_name]
    validation_scores = best_model.predict_proba(x_valid)[:, 1]
    if not np.isfinite(validation_scores).all():
        raise ValueError("The selected model produced invalid validation scores.")
    review_threshold = float(
        np.quantile(validation_scores, 1 - args.review_capacity)
    )

    x_test = test[MODEL_FEATURES]
    y_test = test["isFraud"]
    test_scores = best_model.predict_proba(x_test)[:, 1]
    if not np.isfinite(test_scores).all():
        raise ValueError("The selected model produced invalid test scores.")
    test_predictions = (test_scores >= review_threshold).astype(int)

    final_metrics: dict[str, str | float] = {
        "model": best_model_name,
        "pr_auc": float(average_precision_score(y_test, test_scores)),
        "precision_at_threshold": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall_at_threshold": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "f1_at_threshold": float(
            f1_score(y_test, test_predictions, zero_division=0)
        ),
        "flag_rate": float(test_predictions.mean()),
    }
    final_metrics.update(
        review_metrics(
            y_test,
            test_scores,
            test["amount"],
            capacity=args.review_capacity,
        )
    )
    monitoring = build_monitoring(test, test_scores, test_predictions)

    metadata: dict[str, Any] = {
        "artifact_schema_version": 1,
        "model": best_model_name,
        "review_threshold": review_threshold,
        "review_capacity": float(args.review_capacity),
        "features": list(MODEL_FEATURES),
        "test_metrics": final_metrics,
        "data_source": "demo" if is_demo else "PaySim CSV",
        "data_path": str(data_path.resolve()),
        "demo_data": is_demo,
        "deployment_warning": (
            "Demo-trained artifacts are for integration testing only."
            if is_demo
            else None
        ),
        "trained_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "random_state": int(args.random_state),
        "split_boundaries": {
            "train_end_exclusive": train_end,
            "validation_end_exclusive": validation_end,
        },
        "splits": split_rows,
        "library_versions": library_versions(),
    }

    output_paths = save_artifacts(
        artifact_dir=artifact_dir,
        model=best_model,
        validation_results=validation_results,
        monitoring=monitoring,
        metadata=metadata,
    )

    print(f"Selected model: {best_model_name}")
    print(f"Review threshold: {review_threshold:.6f}")
    print(f"Test PR-AUC: {float(final_metrics['pr_auc']):.6f}")
    print("Saved artifacts:")
    for path in output_paths:
        print(f"- {path}")
    return output_paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_training(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
