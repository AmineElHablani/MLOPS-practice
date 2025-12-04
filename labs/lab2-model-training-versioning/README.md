# Lab 2: Model Training & Versioning with GitHub Actions

Automate machine learning model training, evaluation, versioning, and calibration using GitHub Actions.

## Learning Objectives

- Automate ML model training with GitHub Actions
- Version trained models automatically
- Evaluate model performance in CI/CD pipelines
- Implement model calibration workflows
- Store and track model artifacts

## Prerequisites

- GitHub account
- Python 3.8+
- Basic ML and scikit-learn knowledge
- Completed Lab 1 (recommended)

## Lab Structure
```
lab2-model-training-versioning/
├── src/
│   ├── train_model.py      # Model training script
│   └── evaluate_model.py   # Model evaluation script
├── models/                  # Trained model storage
├── metrics/                 # Evaluation metrics
├── test/                    # Test files
├── workflows/               # GitHub Actions workflows
│   ├── model_retraining_on_push.yml
│   └── model_calibration_on_push.yml
├── requirements.txt
└── README.md
```

## Getting Started

### Option 1: Use This Lab
```bash
# Clone the MLOPS repository
git clone https://github.com/AardwolfFizzler/MLOPS.git
cd MLOPS/labs/lab2-model-training-versioning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Create Your Own Repository
```bash
# Create new repository on GitHub, then:
git clone https://github.com/your-username/ml-training-versioning.git
cd ml-training-versioning

# Copy lab files or create structure
mkdir -p src models metrics test .github/workflows
```

## Step 1: Understanding the Training Script

The `src/train_model.py` script:
- Generates synthetic classification data
- Trains a RandomForest classifier
- Saves the model with timestamp versioning
- Stores model in `models/` directory

### Customize Training

Edit `src/train_model.py` to:
- Use your own dataset
- Change model parameters
- Modify feature engineering
- Add preprocessing steps

## Step 2: Understanding Model Evaluation

The `src/evaluate_model.py` script:
- Loads the latest trained model
- Evaluates on test data
- Calculates F1 Score and other metrics
- Saves results to `metrics/` directory

## Step 3: Set Up GitHub Actions Workflows

### Move Workflows to Correct Location
```bash
# Copy workflow files to .github/workflows/
mkdir -p .github/workflows
cp workflows/model_retraining_on_push.yml .github/workflows/
cp workflows/model_calibration_on_push.yml .github/workflows/
```

### Workflow 1: Model Retraining

**File:** `.github/workflows/model_retraining_on_push.yml`

**Purpose:** Automatically train and evaluate model on every push to main branch

**Steps:**
1. Generate timestamp for versioning
2. Train model using `train_model.py`
3. Evaluate model using `evaluate_model.py`
4. Store model with version timestamp
5. Commit metrics and model to repository

**Trigger:** Push to `main` branch

### Workflow 2: Model Calibration

**File:** `.github/workflows/model_calibration_on_push.yml`

**Purpose:** Calibrate model probabilities for better predictions

**Steps:**
1. Load latest trained model
2. Apply calibration (Platt scaling or isotonic regression)
3. Save calibrated model separately
4. Commit calibrated model to repository

**Trigger:** Push to `main` branch (runs after training)

## Step 4: Push and Trigger Workflows
```bash
# Add all files
git add .

# Commit changes
git commit -m "Add model training and versioning workflows"

# Push to GitHub
git push origin main
```

## Step 5: Monitor Workflow Execution

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Watch workflows execute in real-time
4. Check for green checkmarks ✅
5. View detailed logs for each step

## Step 6: Access Trained Models

After successful workflow execution:

1. Navigate to `models/` directory in your repository
2. Find models with timestamp names (e.g., `model_20241204_143022.pkl`)
3. Download models for local use or deployment
4. Compare different versions

## Understanding Model Versioning

### Automatic Versioning

Each workflow run creates a new model version:
- **Timestamp format:** `model_YYYYMMDD_HHMMSS.pkl`
- **Example:** `model_20241204_143022.pkl`
- **Purpose:** Track model evolution over time

### Benefits

- Track model improvements
- Roll back to previous versions
- Compare performance across versions
- Audit model changes

## Model Calibration Explained

### What is Calibration?

Calibration aligns predicted probabilities with actual outcomes. For example:
- **Uncalibrated:** Model predicts 70% probability, actual rate is 50%
- **Calibrated:** Model predicts 70% probability, actual rate is 70%

### Why Calibrate?

- Better decision-making with reliable probabilities
- Essential for medical, financial, and high-stakes applications
- Improves model trustworthiness

### Calibration Methods

1. **Platt Scaling:** Fits logistic regression to predictions
2. **Isotonic Regression:** Non-parametric, fits step function

### When to Use

- After training any classifier
- When probability estimates matter
- Before deploying to production

## Customization Ideas

### Modify Training Script
```python
# Use real dataset
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# Change model
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)

# Add hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200]}
model = GridSearchCV(RandomForestClassifier(), param_grid)
```

### Add More Metrics

Edit `evaluate_model.py`:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
```

### Schedule Retraining

Add to workflow:
```yaml
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  push:
    branches: [ main ]
```

## Workflow Details

### Model Retraining Workflow
```yaml
name: Model Retraining

on:
  push:
    branches: [ main ]

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Set up Python
      - Install dependencies
      - Generate timestamp
      - Train model
      - Evaluate model
      - Store versioned model
      - Commit changes
```

### Model Calibration Workflow
```yaml
name: Model Calibration

on:
  push:
    branches: [ main ]

jobs:
  calibrate:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Set up Python
      - Install dependencies
      - Load trained model
      - Calibrate probabilities
      - Save calibrated model
      - Commit changes
```

## Best Practices

1. **Version Everything:** Models, data, metrics, and code
2. **Track Metrics:** Store evaluation results for comparison
3. **Test Before Merge:** Run tests on pull requests
4. **Document Changes:** Clear commit messages for model updates
5. **Monitor Performance:** Set up alerts for metric degradation
6. **Separate Calibration:** Keep original and calibrated models

## Troubleshooting

**Workflow not triggering:**
- Ensure workflows are in `.github/workflows/` directory
- Check workflow is on `main` branch
- Verify YAML syntax is correct

**Model not saving:**
- Check `models/` directory exists
- Verify write permissions
- Review workflow logs for errors

**Import errors:**
- Ensure `requirements.txt` includes all dependencies
- Check Python version compatibility

**Calibration fails:**
- Verify trained model exists before calibration
- Check model format is compatible
- Review calibration method parameters

## Common Commands
```bash
# Local testing
python src/train_model.py
python src/evaluate_model.py

# Check workflow status
git log --oneline
git status

# View model files
ls -la models/
ls -la metrics/

# Compare metrics across versions
cat metrics/metrics_*.json
```

## Integration with Production

### Deploy Trained Models

1. Download model from repository
2. Load in production environment
3. Serve predictions via API
4. Monitor performance

### Example Deployment
```python
import joblib

# Load versioned model
model = joblib.load('models/model_20241204_143022.pkl')

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

## Next Steps

- Add data validation checks
- Implement A/B testing workflows
- Set up model monitoring
- Create model registry
- Add automated deployment
- Implement rollback mechanisms

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Model Calibration Guide](https://scikit-learn.org/stable/modules/calibration.html)
- [ML Versioning Best Practices](https://dvc.org/doc)

## License

Educational purposes

## Questions?

Open an issue in the repository for help or clarifications.
