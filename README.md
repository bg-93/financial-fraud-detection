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

The probability threshold is not fixed at 0.5.

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

Create a virtual environment if desired, then install the requirements:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
fraud_detection_notebook.ipynb
```

To run on the full PaySim dataset, create the `data` folder and place the CSV at:

```text
data/PaySim.csv
```

Then restart the notebook and run all cells from the beginning.

## Files produced by the notebook

The notebook creates an `artifacts/` directory containing:

```text
fraud_model.joblib
validation_results.csv
monitoring_over_time.csv
model_metadata.json
```

`fraud_model.joblib` contains both the preprocessing steps and the fitted classifier.

`model_metadata.json` records the chosen model, review threshold, features and final test metrics.

## Deployment approach

The modelling notebook and the deployed application should be kept separate.

A simple deployment would have three parts:

```text
new transaction
      |
      v
feature calculation
      |
      v
saved fraud model
      |
      v
fraud probability
      |
      +--> allow
      +--> manual review
```

### Option 1: Streamlit

For a portfolio demonstration, Streamlit is the simplest option. A small page can allow a reviewer to enter transaction details and display the predicted fraud probability and whether the transaction crosses the review threshold.

This is the best first deployment because the focus remains on the data science rather than frontend development.

### Option 2: FastAPI

A more software-focused version can load `fraud_model.joblib` in a FastAPI service and expose a `/predict` endpoint.

A request could look like:

```json
{
  "type": "TRANSFER",
  "amount": 8700,
  "sender_tx_count_before": 12,
  "sender_mean_amount_before": 190,
  "amount_vs_sender_mean": 45.8,
  "recipient_tx_count_before": 2,
  "pair_tx_count_before": 0,
  "is_new_recipient": 1,
  "hours_since_sender_tx": 3,
  "hour": 2
}
```

The response could contain:

```json
{
  "fraud_probability": 0.86,
  "send_for_review": true
}
```

The FastAPI service could then be containerised with Docker and hosted on a small cloud service. That deployment work is a useful extension, but it does not need to be mixed into the modelling notebook.

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

## Possible extensions

Only add these after the main notebook works and you can explain every existing section clearly.

1. Add rolling 6-hour or 24-hour customer transaction counts.
2. Add simple network features such as the number of different recipients previously paid by a sender.
3. Add SHAP explanations for individual XGBoost predictions.
4. Calibrate predicted probabilities.
5. Build a Streamlit dashboard.
6. Create a FastAPI prediction endpoint and Docker image.
7. Compare performance under several review capacities such as 0.5%, 1% and 2%.

These are extensions rather than requirements. A smaller project that you understand completely is stronger in an interview than a more complicated project that is difficult to defend.

## Suggested repository structure

```text
fraud-detection/
├── fraud_detection_notebook.ipynb
├── README.md
├── requirements.txt
├── demo_paysim.csv
├── data/
│   └── PaySim.csv              # not committed to GitHub
└── artifacts/                  # generated after running notebook
```

Add `data/PaySim.csv` and the generated model file to `.gitignore` if they are too large for GitHub.

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
    -> saved model for deployment
```

That is enough technical depth for a strong undergraduate data science portfolio project without making the implementation unnecessarily complicated.
