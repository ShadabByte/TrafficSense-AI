import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_engineered",
    "indore_ml_dataset_features.csv"
)

PRODUCTION_MODEL = os.path.join(
    BASE_DIR, "models", "traffic_congestion_model.pkl"
)

EXPERIMENTAL_MODEL = os.path.join(
    BASE_DIR, "models",
    "traffic_congestion_model_experimental.pkl"
)


print("=" * 55)
print("TRAFFICSENSE AI - TRAFFIC MODEL IMPROVEMENT")
print("=" * 55)

df = pd.read_csv(DATA_FILE)

required_columns = ["road_id", "hour", "day", "congestion"]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Required dataset columns are missing.")

X = df[["road_id", "hour", "day"]]
y = df["congestion"]

if y.value_counts().min() < 3:
    raise ValueError("Insufficient samples for 3-fold validation.")

print("\nDataset records:", len(df))
print("Traffic classes:", sorted(y.unique()))

categorical_columns = ["road_id", "day"]
numeric_columns = ["hour"]


def make_preprocessor(scale_numeric=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])


print("\nLoading existing production model...")
baseline_model = joblib.load(PRODUCTION_MODEL)

candidates = {
    "Improved Random Forest": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            C=1.0,
            random_state=42
        ))
    ])
}

cv = RepeatedStratifiedKFold(
    n_splits=3,
    n_repeats=5,
    random_state=42
)

scoring = {
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro"
}

results = {}

print("\nEvaluating existing model...")

baseline_scores = cross_validate(
    baseline_model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=1
)

results["Existing Production Model"] = {
    "balanced_accuracy":
        baseline_scores["test_balanced_accuracy"].mean(),
    "macro_f1":
        baseline_scores["test_macro_f1"].mean()
}

print("\nEvaluating experimental models...")

for name, model in candidates.items():

    print("\nTesting:", name)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    results[name] = {
        "balanced_accuracy":
            scores["test_balanced_accuracy"].mean(),
        "macro_f1":
            scores["test_macro_f1"].mean()
    }

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

for name, scores in results.items():

    print("\n", name)
    print(
        "Balanced Accuracy:",
        round(scores["balanced_accuracy"] * 100, 2), "%"
    )
    print(
        "Macro F1:",
        round(scores["macro_f1"] * 100, 2), "%"
    )

best_name = max(
    candidates,
    key=lambda name: (
        results[name]["macro_f1"],
        results[name]["balanced_accuracy"]
    )
)

best_model = candidates[best_name]

print("\n" + "=" * 55)
print("BEST EXPERIMENTAL MODEL:", best_name)
print("=" * 55)

baseline_ba = results["Existing Production Model"]["balanced_accuracy"]
best_ba = results[best_name]["balanced_accuracy"]

baseline_f1 = results["Existing Production Model"]["macro_f1"]
best_f1 = results[best_name]["macro_f1"]

print(
    "Balanced Accuracy Change:",
    round((best_ba - baseline_ba) * 100, 2),
    "percentage points"
)

print(
    "Macro F1 Change:",
    round((best_f1 - baseline_f1) * 100, 2),
    "percentage points"
)

print("\nTraining best experimental model on full dataset...")
best_model.fit(X, y)

joblib.dump(best_model, EXPERIMENTAL_MODEL)

print("\nExperimental model saved:")
print(EXPERIMENTAL_MODEL)

print("\nGenerating 3-fold diagnostic report...")

report_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    best_model,
    X,
    y,
    cv=report_cv,
    n_jobs=1
)

labels = ["FREE", "HEAVY", "LIGHT", "MODERATE"]

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y,
    predictions,
    labels=labels,
    zero_division=0
))

print("CONFUSION MATRIX")
print(confusion_matrix(
    y,
    predictions,
    labels=labels
))

print("\n" + "=" * 55)
print("EXPERIMENT COMPLETED")
print("=" * 55)

print("Production model was NOT modified.")
print("Experimental model is saved separately.")
print("Do not deploy it until validation is reviewed.")
