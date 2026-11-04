import pandas as pd
import os

file = os.path.join("data", "processed", "accident_training_sample.csv")

df = pd.read_csv(file, usecols=["severity"])

counts = df["severity"].value_counts().sort_index()
percentages = df["severity"].value_counts(normalize=True).sort_index() * 100

print("\n===== ACCIDENT TRAINING DATASET =====")
print("Total records:", len(df))

print("\nClass distribution:")
print(counts)

print("\nClass percentages:")
print(percentages.round(2))

print("\nMinority class to majority class ratio:")
print(round(counts.max() / counts.min(), 2))
