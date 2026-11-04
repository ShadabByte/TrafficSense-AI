import pandas as pd

files = [
    "data/processed/feature_engineered/indore_ml_dataset_features.csv",
    "data/processed/feature_engineered/indore_event_log_features.csv",
    "data/processed/feature_engineered/indore_raw_traffic_features.csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    df = pd.read_csv(file)

    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum())

    if "congestion" in df.columns:
        print("\nCongestion distribution:")
        print(df["congestion"].value_counts())

    if "delay" in df.columns:
        print("\nDelay statistics:")
        print(df["delay"].describe())

    if "timestamp" in df.columns:
        print("\nTimestamp range:")
        print("Start:", df["timestamp"].min())
        print("End:", df["timestamp"].max())

    if "road_id" in df.columns:
        print("\nUnique roads:", df["road_id"].nunique())