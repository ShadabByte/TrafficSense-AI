
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

OUTPUT_DIR = os.path.join(DATA_DIR, "feature_engineered")

os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = [
    "indore_ml_dataset_cleaned.csv",
    "indore_raw_traffic_cleaned.csv",
    "indore_event_log_cleaned.csv",
    "weather_cleaned.csv",
    "morth_accidents_2019_cleaned.csv",
    "us_accidents_cleaned.csv"
]

def feature_engineering(df):

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    date_columns = [
        col for col in df.columns
        if any(x in col for x in ["date", "time", "timestamp"])
    ]

    for col in date_columns:

        dt = pd.to_datetime(df[col], errors="coerce")

        if dt.notna().mean() < 0.5:
            continue

        df[col + "_hour"] = dt.dt.hour
        df[col + "_day"] = dt.dt.day
        df[col + "_month"] = dt.dt.month
        df[col + "_weekday"] = dt.dt.dayofweek

        df[col + "_weekend"] = (
            dt.dt.dayofweek >= 5
        ).astype("int8")

        df[col + "_rush_hour"] = (
            dt.dt.hour.isin([7, 8, 9, 17, 18, 19])
        ).astype("int8")

        df[col + "_hour_sin"] = np.sin(
            2 * np.pi * dt.dt.hour / 24
        )

        df[col + "_hour_cos"] = np.cos(
            2 * np.pi * dt.dt.hour / 24
        )

    return df


for filename in datasets:

    input_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(input_path):
        print("SKIPPED:", filename)
        continue

    try:

        print("\nProcessing:", filename)

        df = pd.read_csv(input_path, low_memory=False)

        result = feature_engineering(df)

        output_name = filename.replace(
            "_cleaned.csv", "_features.csv"
        )

        result.to_csv(
            os.path.join(OUTPUT_DIR, output_name),
            index=False
        )

        print("Original:", df.shape)
        print("Engineered:", result.shape)
        print("Saved:", output_name)

    except Exception as e:
        print("ERROR:", filename, e)

print("\nFEATURE ENGINEERING COMPLETED!")
