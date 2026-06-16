import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature

import joblib
import time
import os
import copy

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 1. MLflow Setup
# =====================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("EMPLOYEE_ATTRITION_FULL_AUTOMATION")

# =====================================================================
# 2. Load Raw Data
# =====================================================================
print("Loading raw data...")
data = pd.read_csv("Employee_Attrition.csv")

# Standardize column strings globally
data.columns = data.columns.str.replace(' ', '_').str.lower()

# Drop unique identifier to prevent data leakage
if 'employee_id' in data.columns:
    data.drop(columns=['employee_id'], inplace=True)

data = data.drop_duplicates()

# Segregate Features and Target
X = data.drop(columns=['attrition'])
y = data['attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'left' else 0)

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =====================================================================
# 3. Dynamic Pipeline Architecture
# =====================================================================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

feature_selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median',
    max_features=25
)

pipeline = Pipeline([
    ('Preprocessor', preprocessor),
    ('FeatureSelection', feature_selector),
    ('Model', KNeighborsClassifier())
])

# =====================================================================
# 4. Optuna Objective Functions
# =====================================================================
def objective_lr(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(Model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    pipeline.set_params(Model__C = trial.suggest_float('C', 1e-3, 1e2, log=True))
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv = skf).mean()

def objective_knn(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(Model = KNeighborsClassifier())
    pipeline.set_params(Model__n_neighbors = trial.suggest_int('n_neighbors', 3, 21, 2))
    pipeline.set_params(Model__weights = trial.suggest_categorical('weights', ['uniform', 'distance']))
    pipeline.set_params(Model__p = trial.suggest_int('p', 1, 3))
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv = skf).mean()

def objective_dt(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(Model = DecisionTreeClassifier(class_weight='balanced', random_state=42))
    pipeline.set_params(
        Model__criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        Model__max_depth=trial.suggest_int('max_depth', 2, 30),
        Model__min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        Model__min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        Model__max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()

def objective_svm(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    params = {'C': trial.suggest_float('C', 1e-3, 1e2, log=True), 'kernel': kernel, 'class_weight': 'balanced', 'random_state': 42}
    if kernel in ['rbf', 'poly', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    pipeline.set_params(Model=SVC(**params))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()

def objective_gnb(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(Model=GaussianNB(var_smoothing=trial.suggest_float('var_smoothing', 1e-11, 1e-7, log=True)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()

def objective_rf(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(
        Model=RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            max_depth=trial.suggest_int('max_depth', 5, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()

def objective_gb(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(Preprocessor__num__scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    pipeline.set_params(
        Model=GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()

# =====================================================================
# 5. Execution & Global Tracker
# =====================================================================
objectives = {
    "LogisticRegression": objective_lr, "KNN": objective_knn, "DecisionTree": objective_dt,
    "SVM": objective_svm, "GaussianNB": objective_gnb, "RandomForest": objective_rf, "GradientBoosting": objective_gb
}

results = {}
model_dict = {model: i for i, model in enumerate(objectives.keys())}
scaler_dict = {scaler_type: i for i, scaler_type in enumerate(['standard', 'minmax'])}

# --- GLOBAL TRACKERS TO SAVE THE ULTIMATE WINNER ---
best_global_f1 = 0
best_global_model_name = ""
best_global_pipeline = None

for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    mlflow_cb = MLflowCallback(tracking_uri=None, metric_name="cv_accuracy", mlflow_kwargs={"nested": True})
    study = optuna.create_study(direction="maximize")
    
    start_fit = time.time()
    # REDUCED TRIALS TO 12 FOR SPEED
    study.optimize(obj_fn, n_trials=12, callbacks=[mlflow_cb])
    fit_time = time.time() - start_fit

    best_params = study.best_params
    results[model_name] = {"best_params": best_params, "best_cv_accuracy": study.best_value}

    # Assign winning parameters to the Final Pipeline
    scaler = StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler()
    pipeline.set_params(Preprocessor__num__scaler=scaler)
    
    if model_name == "LogisticRegression":
        pipeline.set_params(Model=LogisticRegression(C=best_params["C"], class_weight='balanced', max_iter=1000, random_state=42))
    elif model_name == "KNN":
        pipeline.set_params(Model__n_neighbors=best_params["n_neighbors"], Model__weights=best_params["weights"], Model__p=best_params["p"])
    elif model_name == "DecisionTree":
        pipeline.set_params(Model__criterion=best_params["criterion"], Model__max_depth=best_params["max_depth"], Model__min_samples_split=best_params["min_samples_split"], Model__min_samples_leaf=best_params["min_samples_leaf"], Model__max_features=best_params["max_features"])
    elif model_name == "SVM":
        params = {"kernel": best_params["kernel"], "C": best_params["C"], "class_weight": "balanced", "random_state": 42}
        if best_params["kernel"] in ["rbf", "poly", "sigmoid"]: params["gamma"] = best_params["gamma"]
        if best_params["kernel"] == "poly": params["degree"] = best_params["degree"]
        pipeline.set_params(Model=SVC(**params))
    elif model_name == "GaussianNB":
        pipeline.set_params(Model__var_smoothing=best_params["var_smoothing"])
    elif model_name == "RandomForest":
        pipeline.set_params(Model__n_estimators=best_params["n_estimators"], Model__criterion=best_params["criterion"], Model__max_depth=best_params["max_depth"], Model__min_samples_split=best_params["min_samples_split"], Model__min_samples_leaf=best_params["min_samples_leaf"], Model__max_features=best_params["max_features"], Model__bootstrap=best_params["bootstrap"])
    elif model_name == "GradientBoosting":
        pipeline.set_params(Model__n_estimators=best_params["n_estimators"], Model__learning_rate=best_params["learning_rate"], Model__max_depth=best_params["max_depth"], Model__min_samples_split=best_params["min_samples_split"], Model__min_samples_leaf=best_params["min_samples_leaf"], Model__max_features=best_params["max_features"], Model__subsample=best_params["subsample"])

    # Train final top model 
    pipeline.fit(X_train, y_train)

    # Testing metrics
    start_test = time.time()
    y_pred = pipeline.predict(X_test)
    test_time = time.time() - start_test

    train_acc = pipeline.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"{model_name} Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    # --- CHECK IF THIS IS THE BEST OVERALL MODEL ---
    if test_f1 > best_global_f1:
        best_global_f1 = test_f1
        best_global_model_name = model_name
        best_global_pipeline = copy.deepcopy(pipeline) # Save a copy of the winner

    # MLflow Top-level Run Tracking
    mlflow.log_metric("model_id", model_dict[model_name])
    mlflow.log_metric("Scalar_id", scaler_dict[best_params["scaler_type"]])
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("train_time", fit_time)
    mlflow.log_metric("test_time", test_time)
    
    input_example = X_train.iloc[[0]]
    signature = infer_signature(input_example, pipeline.predict(input_example))

    mlflow.sklearn.log_model(
        pipeline, 
        name=f"{model_name}_attrition_model",
        signature=signature,
        input_example=input_example
    )
    
    results[model_name].update({"test_f1": test_f1})
    mlflow.end_run()

print("\n" + "="*50)
print("--- Final Project Summary ---")
for model, res in results.items():
    print(f"{model}: CV Acc={res['best_cv_accuracy']:.4f}, Test F1={res['test_f1']:.4f}")

# --- SAVE THE CHAMPION FOR STREAMLIT ---
print("="*50)
print(f"🏆 ULTIMATE CHAMPION: {best_global_model_name} (F1 Score: {best_global_f1:.4f})")
joblib.dump(best_global_pipeline, "best_model.pkl")
print("✅ Saved as 'best_model.pkl'. Ready for deployment!")