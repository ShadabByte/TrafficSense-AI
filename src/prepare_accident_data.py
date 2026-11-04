
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed",
    "feature_engineered", "us_accidents_features.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed",
    "accident_training_sample.csv"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Only use features available for prediction
wanted_columns = [
    "severity",
    "start_lat", "start_lng",
    "temperature(f)", "wind_chill(f)",
    "humidity(%)", "pressure(in)",
    "visibility(mi)", "wind_direction",
    "wind_speed(mph)", "precipitation(in)",
    "weather_condition",
    "amenity", "bump", "crossing", "give_way",
    "junction", "no_exit", "railway", "roundabout",
    "station", "stop", "traffic_calming",
    "traffic_signal", "turning_loop",
    "sunrise_sunset", "civil_twilight",
    "nautical_twilight", "astronomical_twilight",
    "start_time_hour", "start_time_day",
    "start_time_month", "start_time_weekday",
    "start_time_weekend", "start_time_rush_hour",
    "start_time_hour_sin", "start_time_hour_cos"
]

print("Reading full dataset in chunks...")

header = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
use_columns = [c for c in wanted_columns if c in header]

chunks = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=use_columns,
    chunksize=250000
):
    # Take a random 10% from every chunk
    part = chunk.sample(frac=0.10, random_state=42)
    chunks.append(part)

    print("Chunk sampled:", len(part))

sample = pd.concat(chunks, ignore_index=True)

# Final random sample of 200,000 rows
if len(sample) > 200000:
    sample = sample.sample(
        n=200000,
        random_state=42
    )

sample.dropna(subset=["severity"], inplace=True)

sample.to_csv(OUTPUT_FILE, index=False)

print("\nRANDOM SAMPLE READY!")
print("Shape:", sample.shape)
print("\nSeverity distribution:")
print(sample["severity"].value_counts().sort_index())
print("\nSeverity percentages:")
print(
    (sample["severity"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)
print("\nSaved:", OUTPUT_FILE)
