# Payment Fraud Detection — CBA Data Science Portfolio Project

## Project goal

This project builds a payment fraud detection model from PaySim-style transaction data. The aim is not to create the most complicated fraud model possible. The aim is to show a clear data science process that can be explained in an internship interview.

The notebook focuses on five ideas:

1. Fraud data is highly imbalanced, so accuracy is a poor metric.
2. Payment behaviour changes over time, so train and test data should be split chronologically.
3. Historical customer behaviour is often more useful than the transaction amount by itself.
4. A fraud team can only investigate a limited number of transactions, so the model should be evaluated under a review-capacity constraint.
5. A model should be checked over time rather than judged using a single test score.

## Dataset

The intended dataset is **PaySim**, a synthetic mobile-money transaction dataset containing millions of transactions and a small number of labelled fraud cases.

Download PaySim from Kaggle and place the CSV at:

```text
data/PaySim.csv
```

The repository also contains `demo_paysim.csv`. This is only a small synthetic file for checking that the notebook runs. Results from this demo file should not be reported as the final project results.

## Notebook structure

### 1. Data loading

The notebook loads the transactions, sorts them by the PaySim time variable `step`, and reports the fraud rate.

The target is:

```text
isFraud
```

### 2. Leakage check

Some PaySim columns are intentionally excluded.

`newbalanceOrig` and `newbalanceDest` describe balances after the transaction. A real fraud model would need to make a decision before these values are known.

`isFlaggedFraud` represents an existing fraud rule. Giving it to the model would let the new model copy information from an existing detector.

`nameOrig` and `nameDest` are account identifiers. They are used to construct historical behaviour, but the raw IDs are not given directly to the classifier. This avoids simply memorising specific account names.

### 3. Historical behaviour features

The notebook creates features from transactions that occurred before the current PaySim time step.

The main features are:

| Feature | Meaning |
|---|---|
| `amount` | Current transaction amount |
| `log_amount` | Log-transformed transaction amount |
| `hour` | Approximate hour of day from PaySim's time step |
| `sender_tx_count_before` | Number of earlier transactions from the sender |
| `sender_mean_amount_before` | Sender's average earlier transaction amount |
| `amount_vs_sender_mean` | Current amount relative to the sender's earlier average |
| `recipient_tx_count_before` | Number of earlier payments received by the recipient |
| `pair_tx_count_before` | Number of earlier transactions between this sender and recipient |
| `is_new_recipient` | Whether this sender has paid this recipient before |
| `hours_since_sender_tx` | Time since the sender was previously active |
| `type` | PaySim transaction type |

The feature engineering works at the `step` level. A transaction at step `t` only uses information from steps smaller than `t`. This avoids accidentally using later information when creating historical features.

## Why the data is split by time

A random train/test split is easy to use, but it is not a good representation of how a fraud model would operate.

In practice, a model is trained on past transactions and then used on future transactions. The notebook therefore uses approximately:

```text
first 70% of time -> training
next 15%          -> validation
final 15%         -> testing
```

The validation set is used for model selection and threshold selection. The test set is only used at the end.

## Models

Three models are compared.

### Logistic Regression

This is the baseline. It is useful because it is simple, fast and easy to interpret.

### Random Forest

Random Forest can model nonlinear relationships and interactions between features without requiring a complicated mathematical model.

### XGBoost

XGBoost is included as the strongest tabular model in the project. It builds an ensemble of decision trees sequentially, with each tree trying to improve on the errors of the previous trees.

No neural network is required here. The data is tabular and tree-based methods are a natural choice.

## Handling class imbalance

Fraud represents only a very small proportion of transactions. A model that predicts every transaction as legitimate could therefore have very high accuracy while detecting no fraud.

The notebook uses class weighting instead. Logistic Regression and Random Forest use balanced class weights. XGBoost uses the ratio of legitimate to fraudulent training observations through `scale_pos_weight`.

This keeps the original validation and test distributions unchanged.

## Evaluation

### PR-AUC

The main model-comparison metric is precision-recall area under the curve, or PR-AUC.

Precision asks:

> Of the transactions the model considers suspicious, how many are actually fraudulent?

Recall asks:

> Of all fraudulent transactions, how many did the model find?

PR-AUC summarises the trade-off between those quantities over many possible thresholds.

### Review capacity

A bank cannot manually inspect every transaction. The notebook therefore assumes analysts can inspect the highest-risk 1% of transactions.

It reports:

- precision among the top 1% of scores;
- recall among the top 1% of scores;
- proportion of total fraudulent dollar value captured in the top 1%.

This makes the results easier to discuss from a business perspective.

## Threshold selection

The decision threshold on the model score is not fixed at 0.5.

After the best model is chosen, the notebook finds the score corresponding to approximately the highest-risk 1% of validation transactions. This becomes the review threshold.

The same threshold is then applied to the final test set.

This is deliberately simpler than inventing detailed financial costs for false positives and false negatives that we do not actually know.

## Feature importance

The notebook reports simple model importance values.

For Random Forest and XGBoost it uses the model's built-in feature importance. For Logistic Regression it uses the absolute coefficient size after preprocessing.

These importance values are useful for understanding what the model relies on, but they should not be interpreted as causal effects.

SHAP can be added later as an extension if more detailed explanations are wanted, but it is not necessary for the main project.

## Monitoring over time

The final test period is divided into several time blocks. For each block the notebook records:

- fraud rate;
- average model score;
- recall;
- number of transactions.

If performance drops sharply in later blocks, this suggests the relationship between the model inputs and fraud may have changed. In a real system this would motivate further investigation or retraining.

The project deliberately avoids a large monitoring framework because the goal is to demonstrate the idea clearly rather than imitate an enterprise banking platform.

## Running the project

Python 3.12 is required by the pinned runtime and matches the container. Create a
virtual environment and install the project dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs the OpenMP runtime (`brew install libomp`). If that
runtime is unavailable, `python train_model.py --skip-xgboost` still trains and
compares the sklearn candidates; serving an XGBoost artifact still requires OpenMP.

You can train from the notebook as before:

```bash
jupyter notebook
```

Open `fraud_detection_notebook.ipynb`, choose the dataset and run every cell.

For repeatable training outside Jupyter, use the deployment training command:

```bash
python train_model.py --data data/PaySim.csv
```

The full PaySim CSV is required by default; the command fails instead of silently
training a demo model when it is missing. For a local integration smoke test only,
you can explicitly use the bundled synthetic sample:

```bash
python train_model.py --use-demo
```

Run `python train_model.py --help` for alternate artifact paths, review capacity,
random seed and an option to skip XGBoost.

## Model artifacts

The notebook and `train_model.py` both create an `artifacts/` directory containing:

```text
fraud_model.joblib
validation_results.csv
monitoring_over_time.csv
model_metadata.json
```

`fraud_model.joblib` contains both the preprocessing steps and the fitted classifier.

`model_metadata.json` records the chosen model, review threshold, features and final
test metrics. The training command also records its data source, split boundaries,
training time and library versions. Treat joblib files as trusted executable artifacts:
only load a model that your own training workflow produced. The model-bearing runtime
dependencies are pinned in both local and container requirements because joblib model
snapshots are not stable across sklearn or XGBoost versions.
The command-line trainer stages a complete artifact generation, publishes metadata last,
and records a SHA-256 model checksum so an interrupted retrain cannot silently combine a
new model with an old decision threshold.

## Deployment

The modelling workflow and deployed interfaces remain separate. Both Streamlit and
FastAPI call `fraud_detection.py`, which loads the same pipeline and metadata, validates
the feature contract and applies the same review threshold:

```text
new transaction
      |
      v
validated features
      |
      v
saved fraud model
      |
      v
fraud risk score
      |
      +--> allow
      +--> manual review
```

The deployment is stateless. It cannot reconstruct account history from one incoming
transaction, so the caller must supply aggregates calculated only from earlier activity:
sender count and mean, recipient count, sender-recipient pair count, and sender recency.
In a real system those values would normally come from an online feature store or a
transaction-history service.

### Streamlit dashboard

After generating the artifacts, start the reviewer dashboard:

```bash
streamlit run app.py
```

The dashboard accepts a transaction and its pre-transaction history, then displays the
fraud risk score, configured threshold and `Allow` or `Manual review` routing decision.
Because the candidates use class weighting and are not calibrated, the displayed score
is useful for ranking against the threshold but is not a real-world fraud likelihood.
Its sidebar reports model readiness and can reload artifacts after retraining.

For Streamlit Community Cloud, the repository must contain the two runtime files
`artifacts/fraud_model.joblib` and `artifacts/model_metadata.json`. The `.gitignore`
configuration intentionally permits those small files while continuing to exclude the
training dataset and generated evaluation reports. Train and publish them with the same
Python version declared in `.python-version` (currently Python 3.12), then commit both
files before deploying or rebooting the app:

```bash
python3.12 train_model.py --data data/PaySim.csv --artifacts-dir artifacts
git add .gitignore artifacts/fraud_model.joblib artifacts/model_metadata.json
git commit -m "Bundle Streamlit model artifacts"
git push
```

When replacing a model, always publish the model and metadata together; the loader
verifies their checksum and refuses a mismatched pair.

### FastAPI endpoint

Start the API locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The service provides:

- `GET /health` for model readiness;
- `POST /predict` for one transaction;
- `GET /docs` for interactive OpenAPI documentation.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "TRANSFER",
    "amount": 8700,
    "hour": 2,
    "sender_tx_count_before": 12,
    "sender_mean_amount_before": 190,
    "recipient_tx_count_before": 2,
    "pair_tx_count_before": 0,
    "is_new_recipient": 1,
    "hours_since_sender_tx": 3
  }'
```

Example response:

```json
{
  "fraud_score": 0.86,
  "send_for_review": true,
  "review_threshold": 0.20,
  "model": "XGBoost"
}
```

Malformed requests return `422`. If artifacts are missing or incompatible, the process
stays available for diagnostics while `/health` and `/predict` return `503`. The health
response also surfaces a warning when the loaded artifact was trained with `--use-demo`.
This portfolio service does not implement authentication or TLS; place it behind an
authenticated HTTPS gateway before exposing it outside a trusted environment.

### Docker image

Generate model artifacts before building if you want them embedded in the image, then run:

```bash
docker build -t fraud-detection-api .
docker run --rm -p 8000:8000 fraud-detection-api
```

For a cleaner production separation, build the image once and mount a trusted model at
runtime instead:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  fraud-detection-api
```

The image runs as a non-root user and includes a `/health` Docker health check. Set
`FRAUD_ARTIFACT_DIR` to use a different artifact directory outside Docker. Hosting
platforms can override the default port with `PORT`. A container
started without valid artifacts deliberately becomes unhealthy and refuses predictions.
After supplying or replacing mounted artifacts, restart the API container so it loads
the new immutable model snapshot.

### Tests

Install the development dependencies and run the inference and API tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## What to discuss in an interview

The strongest parts of this project are not the specific XGBoost parameters. The useful discussion points are the decisions made around the model.

You should be able to explain:

- why accuracy is misleading for fraud;
- why the split is chronological rather than random;
- why post-transaction balance fields were removed;
- why account IDs are used to create history but not given directly to the model;
- why behaviour relative to a customer's history can be more informative than transaction amount alone;
- why the validation set is used to choose the model and threshold;
- what precision and recall mean for a fraud investigation team;
- why model performance should be checked over time;
- how the saved model could be exposed through Streamlit or FastAPI.

## Extension status

Only add these after the main notebook works and you can explain every existing section clearly.

1. Add rolling 6-hour or 24-hour customer transaction counts.
2. Add simple network features such as the number of different recipients previously paid by a sender.
3. Add SHAP explanations for individual XGBoost predictions.
4. Calibrate predicted probabilities.
5. **Implemented:** Streamlit dashboard.
6. **Implemented:** FastAPI prediction endpoint and Docker image.
7. Compare performance under several review capacities such as 0.5%, 1% and 2%.

These are extensions rather than requirements. A smaller project that you understand completely is stronger in an interview than a more complicated project that is difficult to defend.

## Suggested repository structure

```text
fraud-detection/
├── api.py                       # FastAPI service
├── app.py                       # Streamlit dashboard
├── fraud_detection.py           # shared artifact loading and inference
├── train_model.py               # reproducible command-line training
├── fraud_detection_notebook.ipynb
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt
├── requirements-api.txt         # smaller container dependency set
├── requirements-dev.txt
├── demo_paysim.csv
├── tests/
├── data/
│   └── PaySim.csv              # not committed to GitHub
└── artifacts/                   # generated locally or mounted at runtime
```

The included `.gitignore` keeps the full training data and generated model artifacts out
of Git. Supply trusted artifacts separately in your deployment workflow.

## Project summary

The final project demonstrates a complete but understandable data science workflow:

```text
transaction data
    -> leakage checks
    -> historical behaviour features
    -> chronological validation
    -> imbalanced classification
    -> model comparison
    -> review-capacity threshold
    -> final test evaluation
    -> feature importance
    -> performance monitoring
    -> saved model
    -> shared inference validation
    -> Streamlit dashboard or containerised FastAPI API
```

That is enough technical depth for a strong undergraduate data science portfolio project without making the implementation unnecessarily complicated.
