# Employee_attrition_prediction_system
An end-to-end automated Machine Learning pipeline and Streamlit web application to predict employee attrition risk using Optuna, MLflow, and Scikit-Learn.

A machine learning system that flags employees most likely to leave — before they do.

Built end-to-end across four sprints: raw HR data → cleaned features → five trained classifiers → a tuned Random Forest that hits 83.9% ROC-AUC in production.

---

## Why This Project

Every company loses people. Most don't know who's about to walk out the door until it's too late.

Replacing one employee costs between half and twice their annual salary — recruitment, onboarding, ramp-up time, institutional knowledge gone. HR teams are left reacting instead of preventing.

I wanted to flip that. Instead of a company discovering someone left their resignation letter on a manager's desk, what if the system flagged them three months earlier? That's what this model does. It doesn't predict resignations — it surfaces risk scores so retention conversations happen at the right time, with the right people.

---

## The Dataset

59,598 employee records. 24 features. No missing values.

Standard HRIS data: age, income, tenure, job level, overtime, marital status, distance from home, work-life balance scores, performance ratings, and more.

Target: `attrition` — whether an employee stayed or left.

The class split sits at roughly 52.5% stayed and 47.5% left. That near-balance is unusual in real attrition data (typically 10–20% leavers), which suggests a curated or synthetic dataset. I treated it as-is and optimised for F1 rather than raw accuracy throughout, since the business cost of missing a leaver is always higher than flagging a stayer.

---

## Project Layout

```
employee_attrition_prediction/
│
├── data/
│   └── Employee_Attrition.csv
│
├── notebooks/
│   ├── Sprint1_EDA.ipynb
│   ├── Sprint2_Preprocessing.ipynb
│   ├── Sprint3_ModelBuilding.ipynb
│   └── Sprint4_Optimisation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── employee_attrition_pipeline.joblib
│   ├── standard_scaler.joblib
│   ├── selected_features.joblib
│   └── model_metadata.json
│
├── app/
│   └── app.py
│
├── logs/
│   └── prediction_log.csv
│
├── mlruns/
├── config.py
├── requirements.txt
└── README.md
```

---

## How I Built It

### Sprint 1 — Looking at the Data

Before any modelling, I spent time understanding what the data was actually saying.

Overtime was everywhere in profiles of people who left. Senior employees were leaving at a rate that didn't match their seniority. Distance from home kept surfacing as a quiet separator between stayers and leavers. Marital status split interestingly — single employees showed significantly higher attrition, likely because switching costs are lower when you don't have a mortgage or family anchoring you to a city.

The correlation heatmap was mostly clean. No severe multicollinearity. The numerical features had some skew — `years_at_company` in particular was right-tailed — but nothing unusual for workforce data.

### Sprint 2 — Cleaning and Preparing

Every preprocessing decision was deliberate:

`years_at_company` was right-skewed with 273 IQR outliers. I applied `log1p` transformation. This isn't default behaviour — I chose it because the underlying relationship between tenure and attrition is diminishing, not linear. The log compresses the tail and makes that relationship easier for the model to learn.

`monthly_income` had 50 upper outliers — real executive salaries, not data errors. I capped at the IQR upper bound instead of removing them. Removing legitimate high earners would distort the income distribution.

Binary categoricals (gender, overtime, remote work) got Label Encoding. The ten multi-category columns — job role, satisfaction scores, education level, and others — got One-Hot Encoding with `drop_first=True` to avoid the dummy variable trap.

The scaler was fit only on training data. That's not optional. Fitting on the full dataset before splitting is data leakage — the model implicitly sees test set statistics during training. Every metric looks better but the model fails in production.

Final split: 80% training, 20% test, stratified by attrition to preserve the class ratio in both halves.

### Sprint 3 — Five Models, One Winner

I trained Logistic Regression, Decision Tree, Random Forest, SVM (RBF kernel), and Naive Bayes against the same preprocessed dataset.

The results were telling:

Decision Tree hit 100% training accuracy and 67.3% on test. That's a textbook overfit — the tree memorised the training data. Without depth constraints, it found splits specific to noise.

Random Forest improved significantly — 74.9% test accuracy — but still carried a 25% overfitting gap. Bagging helps, but it doesn't fix unconstrained trees.

Logistic Regression was the steadiest performer: 75.0% test accuracy, 76.27% F1, 0.12% overfitting gap. The model that generalises best isn't always the most complex one.

Naive Bayes was stable but weak. Its independence assumption violated the real correlations in the data — income, tenure, and job level don't move independently, and the model couldn't capture that.

SVM with RBF kernel was competitive but too slow for iterative tuning on 59K rows.

I carried Logistic Regression and Random Forest into Sprint 4. One for its stability, one for its ceiling.

### Sprint 4 — Building What the Data Was Missing

The biggest single improvement in this sprint came before any hyperparameter tuning: feature engineering.

Four new features built from existing columns:

**`career_growth_rate`** — promotions divided by years at company plus one. An employee with zero promotions after eight years has a near-zero score. That number means something.

**`income_per_tenure`** — monthly income divided by years at company plus one. Identifies employees who've been around long enough to expect more but haven't received it.

**`loyalty_ratio`** — years at company divided by age. A 45-year-old who's spent two years at the company has a different risk profile than a 27-year-old who's spent two years there.

**`dependent_income`** — income divided by number of dependents plus one. A rough proxy for financial pressure. An employee supporting three dependents on ₹30,000 is in a very different position than someone supporting none on the same salary.

`dependent_income` ended up ranked 3rd overall by Random Forest feature importance. The engineered features pulled signal that the raw columns couldn't express individually.

Feature selection ran in three stages. Correlation filtering dropped `years_at_company_log` — it carried nearly identical information to `loyalty_ratio` and the redundancy was measurable. Random Forest importance ranked the full feature set. RFE with a Logistic Regression estimator selected the final 25 from 45, cutting the feature space nearly in half while holding F1 flat.

Tuning ran GridSearchCV on Logistic Regression (`C`, `penalty`, `solver`) and RandomizedSearchCV on Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`). Both optimised for F1 on 3 or 5-fold CV.

The final Random Forest used: `n_estimators=100`, `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=4`, `max_features='sqrt'`.

---

## Results

| | Sprint 3 RF | Sprint 4 RF (Final) |
|---|---|---|
| Train Accuracy | 100.00% | 82.11% |
| Test Accuracy | 74.93% | 75.29% |
| F1 Score | 76.12% | 76.63% |
| Precision | — | 76.04% |
| Recall | — | 77.24% |
| ROC-AUC | 74.87% | **83.89%** |
| Overfit Gap | 25.07% | **6.82%** |
| Features | 41 | 25 |

The number that matters most is ROC-AUC. It went from 74.87% to 83.89% — a 9-point jump — by combining feature engineering with tuning. In practical terms, the model correctly ranks a high-risk employee above a low-risk one in roughly 84 out of 100 comparisons. That's what an HR team acts on: not predictions, but rankings.

Cross-validation F1 came in at 76.33% ± 0.42%. Tight spread. The model isn't sensitive to which employees end up in which fold.

For context: production attrition models working on HRIS-only data — the same category as this dataset — typically land between 73–80% accuracy and 78–85% ROC-AUC. This model sits in the upper half of that range without any external data (engagement surveys, LinkedIn signals, manager ratings), which are the variables that push enterprise models past 85%.

---

## What Drives Attrition

The top features by Random Forest importance, in plain language:

**Job level (Senior)** sits at the top. Senior employees who've stagnated — no new challenges, no upward path — leave. They have the market value to do it.

**Distance from home** is underestimated by most HR teams. Long commutes compound over time. Three years of two-hour daily round trips isn't just an inconvenience — it's a drain that eventually becomes a reason to leave.

**Dependent income** (engineered). Financial pressure is invisible in raw HR data. This feature made it visible.

**Income per tenure** (engineered). The clearest signal of someone who feels underpaid relative to their investment in the company.

**Marital status (Single)**. Lower switching costs. No mortgage, no partner tied to the city, fewer constraints. This doesn't mean single employees are disloyal — it means the friction of leaving is lower.

If I were advising an HR team on where to focus first: **senior employees with long commutes and flat salary growth**. That's the highest-risk intersection in this data, and it's identifiable months before anyone hands in a notice.

---

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit, MLflow, Plotly.

---

## Setup

```bash
git clone https://github.com/<your-username>/employee-attrition-prediction.git
cd employee-attrition-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run the Streamlit app:**
```bash
streamlit run app/app.py
```

**Retrain from scratch:**
```bash
python src/train.py
```

**Single prediction:**
```bash
python src/predict.py --age 34 --overtime Yes --job_level Senior --monthly_income 7500 --years_at_company 6
```

**MLflow dashboard:**
```bash
mlflow ui --backend-store-uri mlruns/
```

---

## Model Artifacts

Everything needed to serve predictions without retraining:

```python
import joblib
import pandas as pd

model    = joblib.load("models/employee_attrition_pipeline.joblib")
scaler   = joblib.load("models/standard_scaler.joblib")
features = joblib.load("models/selected_features.joblib")

X = pd.DataFrame([your_employee_record])[features]
risk_score = model.predict_proba(X)[:, 1][0]

print(f"Attrition probability: {risk_score*100:.1f}%")
```

---

## MLOps

Every training run logs to MLflow automatically: parameters, metrics, and artifacts. Nothing is tracked manually.

Every prediction from the app appends a row to `logs/prediction_log.csv` with timestamp, inputs, prediction, and probability. That log is how model drift would be caught in production.

All paths, feature lists, encoding maps, hyperparameters, and risk thresholds live in `config.py`. There are no hardcoded values scattered across files. When the model is retrained with different parameters, one file changes.
