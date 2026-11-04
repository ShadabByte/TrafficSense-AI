
import geopandas as gpd
import pandas as pd
import numpy as np


ROAD_FILE = "data/processed/bhopal_roads.geojson"


def normalize_value(value, default="other"):
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [str(x) for x in value if x is not None]
        return values[0] if values else default

    if value is None or pd.isna(value):
        return default

    return str(value)


def load_bhopal_roads():
    roads = gpd.read_file(ROAD_FILE)

    roads["highway"] = roads["highway"].apply(
        lambda x: normalize_value(x, "other")
    )

    roads["name"] = roads["name"].apply(
        lambda x: normalize_value(x, "")
    )

    roads["road_id"] = (
        roads["u"].astype(str) + "_" +
        roads["v"].astype(str) + "_" +
        roads["key"].astype(str)
    )

    roads["display_name"] = roads.apply(
        lambda row: (
            row["name"]
            if row["name"]
            else row["highway"].replace("_", " ").title() + " Road"
        ),
        axis=1
    )

    return roads


def estimate_congestion(road_type, hour, day):
    """
    Illustrative demo estimate only.
    Not trained on Bhopal traffic observations.
    """

    road_type = normalize_value(road_type).lower()

    busy_roads = {
        "primary": 3,
        "primary_link": 3,
        "secondary": 2,
        "secondary_link": 2,
        "tertiary": 2,
        "tertiary_link": 2,
        "residential": 1,
        "service": 0,
        "unclassified": 1
    }

    score = busy_roads.get(road_type, 0)

    if 8 <= hour <= 11 or 17 <= hour <= 20:
        score += 2
    elif 12 <= hour <= 16:
        score += 1

    if day in ["Saturday", "Sunday"]:
        score -= 1

    if score <= 1:
        return "FREE"

    if score == 2:
        return "LIGHT"

    if score == 3:
        return "MODERATE"

    return "HEAVY"


if __name__ == "__main__":
    roads = load_bhopal_roads()

    print("Bhopal roads loaded:", len(roads))
    print("\nSample roads:")
    print(
        roads[
            ["road_id", "display_name", "highway"]
        ].head(10).to_string(index=False)
    )

    result = estimate_congestion(
        road_type="primary",
        hour=18,
        day="Monday"
    )

    print("\nDemo congestion estimate:", result)
