import os
import json
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report, confusion_matrix
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed", "accident_training_sample.csv")
MODEL_DIR = os.path.join(BASE, "models")
REPORT_DIR = os.path.join(BASE, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("ACCIDENT SEVERITY - EXPERIMENTAL MODEL")
print("=" * 60)

df = pd.read_csv(DATA)
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["severity"])
y = df["severity"].astype(int)

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

print("Dataset:", df.shape)
print("Numeric features:", len(numeric))
print("Categorical features:", len(categorical))
print("\nSeverity distribution:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def make_pipeline(classifier):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    return Pipeline([
        ("preprocessor", preprocess),
        ("classifier", classifier)
    ])

experiments = [
    (
        "baseline",
        RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    ),
    (
        "balanced_candidate",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=24,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    )
]

results = []
best_model = None
best_name = None
best_score = -1

for name, classifier in experiments:
    print("\n" + "=" * 50)
    print("Training:", name)
    start = time.time()

    model = make_pipeline(classifier)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    report = classification_report(
        y_test, predictions,
        labels=[1, 2, 3, 4],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test, predictions, labels=[1, 2, 3, 4]
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Balanced Accuracy:", round(balanced * 100, 2), "%")
    print("Macro F1:", round(macro_f1 * 100, 2), "%")
    print("\nPer-class results:")
    print(classification_report(
        y_test, predictions,
        labels=[1, 2, 3, 4],
        zero_division=0
    ))
    print("Confusion Matrix (labels 1,2,3,4):")
    print(matrix)
    print("Training time:", round(time.time() - start, 1), "seconds")

    results.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    })

    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = model
        best_name = name

print("\n" + "=" * 60)
print("EXPERIMENT COMPARISON")
print("=" * 60)

for result in results:
    print(
        result["model"],
        "| Accuracy:", round(result["accuracy"] * 100, 2),
        "| Balanced Accuracy:", round(result["balanced_accuracy"] * 100, 2),
        "| Macro F1:", round(result["macro_f1"] * 100, 2)
    )

model_path = os.path.join(
    MODEL_DIR, "accident_severity_experimental.pkl"
)
report_path = os.path.join(
    REPORT_DIR, "accident_model_experiment.json"
)

joblib.dump(best_model, model_path)

output = {
    "best_model": best_name,
    "selection_metric": "macro_f1",
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "classes": [1, 2, 3, 4],
    "experiments": results,
    "production_model_modified": False
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nBest experimental model:", best_name)
print("Saved model:", model_path)
print("Saved report:", report_path)
print("\nProduction model was NOT modified.")
print("This experimental model is NOT deployed.")
print("=" * 60)
