
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
    "accident_training_sample.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(
    MODEL_DIR, "accident_severity_model.pkl"
)

print("Loading accident dataset...")

df = pd.read_csv(DATA_FILE)

# Separate target and features
X = df.drop(columns=["severity"])
y = df["severity"].astype(int)

# Identify column types
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("Dataset shape:", df.shape)
print("Numeric features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))

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

# Random Forest model
classifier = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("\nMaking predictions...")
predictions = model.predict(X_test)

# Evaluation
print("\n========== MODEL RESULTS ==========")

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
    labels=[1, 2, 3, 4],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=[1, 2, 3, 4]
))

# Save trained pipeline
joblib.dump(model, MODEL_FILE)

print("\nModel saved successfully!")
print("Location:", MODEL_FILE)
