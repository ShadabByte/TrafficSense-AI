
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR, "data", "processed",
    "feature_engineered", "indore_ml_dataset_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(
    MODEL_DIR, "traffic_congestion_model.pkl"
)

print("Loading traffic dataset...")

df = pd.read_csv(DATA_FILE)

# Delay is excluded to avoid possible target leakage
X = df[["road_id", "hour", "day"]]
y = df["congestion"]

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts())

categorical_columns = ["road_id", "day"]
numeric_columns = ["hour"]

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

classifier = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining Traffic Model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n========== TRAFFIC MODEL RESULTS ==========")

print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("Balanced Accuracy:", round(
    balanced_accuracy_score(y_test, predictions) * 100, 2
), "%")

print("Macro F1 Score:", round(
    f1_score(y_test, predictions, average="macro") * 100, 2
), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, MODEL_FILE)

print("\nTraffic model saved successfully!")
print("Location:", MODEL_FILE)
