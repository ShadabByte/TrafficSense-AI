import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "data/processed/feature_engineered/indore_ml_dataset_features.csv"
MODEL_FILE = "models/traffic_congestion_model.pkl"

df = pd.read_csv(DATA_FILE)

X = df[["road_id", "hour", "day"]]
y = df["congestion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = joblib.load(MODEL_FILE)
predictions = model.predict(X_test)

print("\n===== TRAFFIC MODEL VALIDATION =====")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, predictions) * 100, 2), "%")
print("Macro F1:", round(f1_score(y_test, predictions, average="macro", zero_division=0) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("Class order:", model.classes_)
