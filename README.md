# 💧 AquaGuard — Water Potability Prediction System

A **production-grade machine learning system** for binary water safety classification, built with XGBoost, FastAPI, Optuna, SHAP, and MLflow.

---

## 📋 Project Overview

**Problem:** Given 9 water quality measurements, predict whether water is safe to drink (potable = 1) or not (potable = 0).

**Dataset:** [Kaggle Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability) — 3,276 samples, 9 features, binary target.

**Key dataset characteristics:**
- Missing values in `ph` (~15%), `Sulfate` (~24%), `Trihalomethanes` (~5%)
- Class imbalance: ~61% Not Potable, ~39% Potable
- High noise with overlapping feature distributions
- No single feature is strongly predictive alone
- **Realistic test accuracy ceiling: 68–78%** (inherently noisy dataset)

---

## 🏗 Architecture

```
Raw Data
   │
   ▼
[1] Load & Validate (src/data/loader.py)
   │
   ▼
[2] Stratified Split: 70% Train | 15% Val | 15% Test
   │
   ▼
[3] Preprocessing Pipeline (fit on X_train ONLY)
   ├── Median Imputer
   ├── Feature Engineering (+5 interaction features)
   ├── RobustScaler
   └── SelectFromModel (Random Forest, threshold=median)
   │
   ▼
[4] Conservative SMOTE (training split only)
   │
   ├─► [5] Random Forest Baseline
   │
   ▼
[6] Optuna Hyperparameter Optimisation (100 trials, validation set)
   │
   ▼
[7] XGBoost Training (early stopping on validation)
   │
   ▼
[8] Threshold Optimisation (composite score, validation set)
   │
   ▼
[9] Evaluation: Train | Val | Test
   │
   ├── Overfitting Detection
   ├── SHAP Explanations
   └── MLflow Logging
   │
   ▼
[10] FastAPI + Web UI
```

---

## ⚡ Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Training Pipeline
```bash
python main.py --mode train
```

### Start API Server
```bash
python main.py --mode serve
```

### Train then Serve
```bash
python main.py --mode all
```

### Endpoints
- **Web UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

---

## 🐳 Docker

```bash
# Build and start
docker-compose up --build

# Access
# Web UI:   http://localhost:8000/ui
# API Docs: http://localhost:8000/docs
```

---

## 📊 MLflow Tracking

```bash
mlflow ui --backend-store-uri mlruns/
# Visit http://localhost:5000
```

---

## 🔬 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.0,
    "Hardness": 196.0,
    "Solids": 20791.0,
    "Chloramines": 7.3,
    "Sulfate": 368.0,
    "Conductivity": 564.0,
    "Organic_carbon": 10.4,
    "Trihalomethanes": 86.0,
    "Turbidity": 2.96
  }'
```

### Predict (Python)
```python
import requests

sample = {
    "ph": 7.0, "Hardness": 196.0, "Solids": 20791.0,
    "Chloramines": 7.3, "Sulfate": 368.0, "Conductivity": 564.0,
    "Organic_carbon": 10.4, "Trihalomethanes": 86.0, "Turbidity": 2.96
}
response = requests.post("http://localhost:8000/predict", json=sample)
print(response.json())
# {
#   "potability": 1,
#   "probability": 0.7234,
#   "threshold_used": 0.48,
#   "interpretation": "POTABLE",
#   "confidence": "HIGH"
# }
```

---

## 📈 Model Performance

| Metric    | Train  | Val    | Test   |
|-----------|--------|--------|--------|
| Accuracy  | ~0.74  | ~0.72  | ~0.73  |
| Precision | ~0.72  | ~0.70  | ~0.71  |
| Recall    | ~0.70  | ~0.68  | ~0.69  |
| F1        | ~0.71  | ~0.69  | ~0.70  |
| ROC-AUC   | ~0.81  | ~0.80  | ~0.80  |

**Train-Val gap < 0.08** — healthy generalisation, not memorisation.

> ⚠️ Exact metrics depend on the random seed and Optuna trial outcomes.
> A model reporting 90%+ accuracy on this dataset is almost certainly overfit.

---

## 🔑 Model Selection Justification

### Why XGBoost?
- Gradient boosting with L1/L2 regularisation (reg_alpha, reg_lambda) prevents overfitting
- Native `early_stopping_rounds` halts training when validation loss plateaus
- `scale_pos_weight` handles class imbalance in the loss function directly
- Compatible with Optuna's Bayesian hyperparameter search
- Superior to single Decision Trees (no regularisation) and comparable to Random Forest with better calibration

### Why Not Random Forest as primary?
Random Forest is used as a baseline for comparison. It lacks early stopping and explicit regularisation terms, making XGBoost more controllable for anti-overfitting constraints.

### Why Conservative SMOTE (0.68, k=3)?
Default SMOTE creates a 50-50 balance which causes precision collapse on this noisy dataset. We use a conservative `sampling_strategy=0.68` (just above the natural 0.64 ratio) to handle imbalance without injecting too much synthetic noise.

### Why No scale_pos_weight?
Since we are already using SMOTE to rebalance the training data, adding `scale_pos_weight` on top causes extreme recall inflation. We rely on SMOTE for data-level balancing and standard logloss for training.

### Why Balanced Threshold Objective?
`composite = 0.40·F1 + 0.40·Accuracy + 0.20·ROC-AUC`
We use a search floor of **0.45** and balanced F1/Accuracy weights to prevent the model from collapsing into majority-class predictions while controlling recall inflation. We also use a `diagnose_probability_distribution` tool during training to monitor the model's calibration and class separation.

### Why No Feature Selection?
With a small dataset (2,293 samples), feature selection by importance can discard collective signals. We keep all 11 features (9 raw + 2 engineered) to maximize the information available to the model.

### Why RobustScaler?
This dataset has extreme outliers in `Solids` (max ~60k, median ~20k) and `Conductivity`. StandardScaler uses mean/std — both distorted by outliers. RobustScaler uses IQR and median, making scaling resistant to extreme values.

---

## 🏛 Key Design Decisions

### Data Leakage Prevention
All preprocessing transformers (imputer, scaler, feature selector) are fitted **exclusively on X_train** and applied to X_val and X_test. This is enforced by the sklearn Pipeline architecture.

### SMOTE applied after splitting
SMOTE is applied **only to the training split after preprocessing**, preventing synthetic samples from influencing validation/test evaluation.

### Threshold on validation, never test
The optimal threshold is selected on the validation set. The test set is reserved for final evaluation only and is never used during training or optimisation.

---

## ☁️ Cloud Deployment (AWS Elastic Beanstalk)

```bash
# 1. Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# 2. Build and tag
docker build -t water-potability .
docker tag water-potability:latest \
  <account>.dkr.ecr.us-east-1.amazonaws.com/water-potability:latest

# 3. Push
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/water-potability:latest

# 4. Deploy to Elastic Beanstalk
eb init water-potability --platform docker --region us-east-1
eb create water-potability-env
eb deploy
```

Update `Dockerrun.aws.json` with your AWS account ID before deploying.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage:
- `test_preprocessing.py` — data leakage, SMOTE isolation, feature engineering shape
- `test_training.py` — XGBoost training, early stopping, overfitting detection, threshold range
- `test_metrics.py` — evaluate_split output structure, value ranges, confusion matrix
- `test_api.py` — health endpoint, valid prediction, invalid input rejection (422)

---

## ⚠️ Known Limitations

1. **Dataset noise ceiling**: ~78% is the theoretical maximum accuracy for this dataset due to overlapping class distributions
2. **Missing values**: 15–24% missing in some features — median imputation is a reasonable but imperfect solution
3. **No external validation**: Model is validated on a holdout from the same source dataset
4. **Educational purpose**: Not certified for official water safety assessments

---

## 📁 Project Structure

```
water_potability/
├── src/
│   ├── config.py          # Centralised config loader (config.yaml)
│   ├── data/              # loader, splitter, validator
│   ├── preprocessing/     # imputer, scaler, feature_engineer, selector, pipeline
│   ├── training/          # smote_handler, baseline, trainer, tuner
│   ├── evaluation/        # metrics, threshold, overfitting_detector
│   ├── explainability/    # shap_explainer
│   └── utils/             # logger, mlflow_utils, artifact_manager
├── api/                   # FastAPI: main, schemas, predictor, middleware
├── ui/                    # Web UI: index.html, style.css, app.js
├── tests/                 # pytest test suite
├── config/config.yaml     # All tunable parameters
├── main.py                # Single pipeline entrypoint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
