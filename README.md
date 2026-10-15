# AI Model Evaluation & Drift Monitoring (Credit-Risk Classifier)

A beginner-friendly project that builds a machine learning model and then assesses it the way an audit or risk team would: accuracy, robustness, drift and human oversight.

## What it does
1. **Model evaluation:** compares Logistic Regression and Random Forest using accuracy, precision, recall and ROC-AUC with 5-fold cross-validation.
2. **Robustness testing:** adds noise to numeric inputs and measures how much AUC drops.
3. **Drift monitoring:** compares Year 1 vs a simulated Year 2 using Population Stability Index (PSI) and the Kolmogorov-Smirnov test on inputs and on predicted probabilities.
4. **Human-in-the-loop control:** routes low-confidence predictions (probability 0.35-0.65) to manual review and defines KPI alert thresholds (input PSI > 0.25 or review rate > 40%).

## Data
German Credit dataset from OpenML (`credit-g`). If offline, the script falls back to synthetic data.
**Note:** "Year 2" data is simulated by shifting two input features; it is not real time-series data.

## Tech stack
Python, pandas, NumPy, scikit-learn, SciPy

## Run it
```bash
pip install scikit-learn pandas numpy scipy
python ai_model_evaluation_drift_project.py
```

## Key takeaways
- A higher-accuracy model is not automatically a reliable one: check robustness and drift.
- PSI below 0.1 is stable, 0.1-0.25 needs watching, above 0.25 is significant drift.
- Human review of low-confidence cases is a simple control over AI-driven decisions.

## Limitations
Single public dataset, simulated drift, no hyperparameter tuning, no fairness analysis.
