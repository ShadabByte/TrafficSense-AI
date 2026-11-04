
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR, "data", "processed", "feature_engineered"
)

# 1. Traffic Dataset Analysis
traffic_file = os.path.join(
    DATA_DIR, "indore_ml_dataset_features.csv"
)

traffic = pd.read_csv(traffic_file)

print("\n========== TRAFFIC TARGET ==========")
print("Shape:", traffic.shape)
print("\nCongestion Distribution:")
print(traffic["congestion"].value_counts(dropna=False))

print("\nCongestion Percentages:")
print(
    (traffic["congestion"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nDelay Statistics:")
print(traffic["delay"].describe())


# 2. Event Dataset Analysis
event_file = os.path.join(
    DATA_DIR, "indore_event_log_features.csv"
)

events = pd.read_csv(event_file)

print("\n========== EVENT DATA ==========")
print("Shape:", events.shape)
print("\nCongestion Distribution:")
print(events["congestion"].value_counts(dropna=False))


# 3. US Accident Severity Analysis
accident_file = os.path.join(
    DATA_DIR, "us_accidents_features.csv"
)

severity_counts = {}

print("\n========== ACCIDENT TARGET ==========")

for chunk in pd.read_csv(
    accident_file,
    usecols=["severity"],
    chunksize=500000
):
    counts = chunk["severity"].value_counts(dropna=False)

    for label, count in counts.items():
        severity_counts[label] = (
            severity_counts.get(label, 0) + count
        )

severity = pd.Series(severity_counts).sort_index()

print("\nSeverity Distribution:")
print(severity)

print("\nSeverity Percentages:")
print((severity / severity.sum() * 100).round(2))

print("\nTotal Accident Records:", severity.sum())

print("\nTARGET ANALYSIS COMPLETED!")
