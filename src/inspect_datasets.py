
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR, "data", "processed", "feature_engineered"
)

files = [
    "indore_ml_dataset_features.csv",
    "indore_raw_traffic_features.csv",
    "indore_event_log_features.csv",
    "us_accidents_features.csv",
    "morth_accidents_2019_features.csv"
]

for file in files:

    path = os.path.join(DATA_DIR, file)

    if not os.path.exists(path):
        print("\nNOT FOUND:", file)
        continue

    df = pd.read_csv(path, nrows=5, low_memory=False)

    print("\n" + "=" * 70)
    print("DATASET:", file)
    print("=" * 70)

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nSAMPLE ROWS:")
    print(df.head(3).to_string(index=False))

print("\nDATASET INSPECTION COMPLETED!")
