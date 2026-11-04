
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR, "data", "processed", "feature_engineered"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

report = []

for file in os.listdir(DATA_DIR):

    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)

    print("\n" + "=" * 65)
    print("DATASET:", file)

    # Sample only: avoids loading huge datasets into RAM
    df = pd.read_csv(path, nrows=10000, low_memory=False)

    print("Sample Shape:", df.shape)
    print("\nALL COLUMNS:")
    print(df.columns.tolist())

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nMISSING VALUES (%):")
    print((df.isnull().mean() * 100).round(2))

    print("\nSAMPLE DATA:")
    print(df.head(3).to_string())

    for col in df.columns:
        report.append({
            "Dataset": file,
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Missing Percentage": round(
                df[col].isnull().mean() * 100, 2
            ),
            "Unique Values": df[col].nunique()
        })

report_df = pd.DataFrame(report)

report_path = os.path.join(
    REPORT_DIR, "feature_quality_report.csv"
)

report_df.to_csv(report_path, index=False)

print("\nQUALITY CHECK COMPLETED!")
print("Report saved:", report_path)
