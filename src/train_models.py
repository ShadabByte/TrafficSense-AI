
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR, "data", "processed", "feature_engineered"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(X, y, model_name, output_file):
    print(f"\n========== {model_name} ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    numeric_cols = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training started...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("Balanced Accuracy:", round(
        balanced_accuracy_score(y_test, predictions), 4
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    save_path = os.path.join(MODEL_DIR, output_file)
    joblib.dump(pipeline, save_path)

    print("Model saved:", save_path)


# ==========================================
# MODEL 1: TRAFFIC CONGESTION
# ==========================================

traffic_file = os.path.join(
    DATA_DIR, "indore_ml_dataset_features.csv"
)

traffic = pd.read_csv(traffic_file)

# Delay is excluded to avoid possible target leakage.
traffic = traffic.dropna(
    subset=["road_id", "hour", "day", "congestion"]
)

X_traffic = traffic[["road_id", "hour", "day"]]
y_traffic = traffic["congestion"]

print("Traffic dataset:", traffic.shape)
print("Traffic classes:", y_traffic.value_counts().to_dict())

train_model(
    X_traffic,
    y_traffic,
    "TRAFFIC CONGESTION MODEL",
    "traffic_model.joblib"
)


# ==========================================
# MODEL 2: ACCIDENT SEVERITY
# ==========================================

accident_file = os.path.join(
    DATA_DIR, "us_accidents_features.csv"
)

# Read a manageable sample from the large CSV.
samples = []

chunksize = 250000
sample_per_chunk = 4000

for chunk_number, chunk in enumerate(
    pd.read_csv(accident_file, chunksize=chunksize)
):
    chunk_sample = chunk.sample(
        n=min(sample_per_chunk, len(chunk)),
        random_state=42 + chunk_number
    )
    samples.append(chunk_sample)

    print(
        f"Accident data chunks processed: {chunk_number + 1}"
    )

accidents = pd.concat(samples, ignore_index=True)

# Only use information that could reasonably be available
# at or before the accident start time.
features = [
    "start_lat",
    "start_lng",
    "temperature(f)",
    "humidity(%)",
    "pressure(in)",
    "visibility(mi)",
    "wind_speed(mph)",
    "precipitation(in)",
    "weather_condition",
    "amenity",
    "bump",
    "crossing",
    "give_way",
    "junction",
    "railway",
    "roundabout",
    "station",
    "stop",
    "traffic_calming",
    "traffic_signal",
    "turning_loop",
    "start_time_hour",
    "start_time_month",
    "start_time_weekday",
    "start_time_weekend",
    "start_time_rush_hour",
    "start_time_hour_sin",
    "start_time_hour_cos"
]

# Keep only columns that exist in the dataset.
features = [
    col for col in features if col in accidents.columns
]

accidents = accidents.dropna(subset=["severity"])

X_accident = accidents[features].copy()
y_accident = accidents["severity"]

print("\nAccident sample shape:", accidents.shape)
print("Accident features:", len(features))
print("Severity classes:", y_accident.value_counts().to_dict())

train_model(
    X_accident,
    y_accident,
    "ACCIDENT SEVERITY MODEL",
    "accident_severity_model.joblib"
)

print("\n========== ALL MODEL TRAINING COMPLETED ==========")
