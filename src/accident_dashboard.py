
import streamlit as st
import pandas as pd
import numpy as np
import folium

from pathlib import Path
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


# =====================================================
# 1. PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="TrafficSense AI - Accident Dashboard",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 TrafficSense AI - Accident Hotspot Dashboard")

st.caption(
    "Location-based accident analysis using the US Accidents training sample."
)

st.warning(
    "DATA LIMITATION: This dataset contains US accident records, "
    "not Bhopal accident records. Map hotspots represent the US dataset only."
)


# =====================================================
# 2. PATH
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "accident_training_sample.csv"
)


# =====================================================
# 3. LOAD DATA
# =====================================================

@st.cache_data(show_spinner="Loading accident dataset...")
def load_accidents(file_path):

    df = pd.read_csv(file_path)

    required_columns = [
        "severity",
        "start_lat",
        "start_lng"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["severity"] = pd.to_numeric(
        df["severity"], errors="coerce"
    )

    df["start_lat"] = pd.to_numeric(
        df["start_lat"], errors="coerce"
    )

    df["start_lng"] = pd.to_numeric(
        df["start_lng"], errors="coerce"
    )

    df = df.dropna(
        subset=["severity", "start_lat", "start_lng"]
    ).copy()

    df = df[
        df["start_lat"].between(-90, 90)
        & df["start_lng"].between(-180, 180)
    ].copy()

    df["severity"] = df["severity"].astype(int)

    if "start_time_hour" in df.columns:
        df["start_time_hour"] = pd.to_numeric(
            df["start_time_hour"], errors="coerce"
        )

    if "start_time_month" in df.columns:
        df["start_time_month"] = pd.to_numeric(
            df["start_time_month"], errors="coerce"
        )

    return df


if not DATA_FILE.exists():

    st.error("Accident dataset not found.")

    st.code(str(DATA_FILE))

    st.stop()


try:

    accidents = load_accidents(str(DATA_FILE))

except Exception as error:

    st.error("Dataset loading failed.")

    st.exception(error)

    st.stop()


if accidents.empty:

    st.error("No valid accident records found.")

    st.stop()


# =====================================================
# 4. SEVERITY LABELS
# =====================================================

severity_labels = {
    1: "Severity 1",
    2: "Severity 2",
    3: "Severity 3",
    4: "Severity 4"
}

severity_colors = {
    1: "#16A34A",
    2: "#EAB308",
    3: "#F97316",
    4: "#DC2626"
}


# =====================================================
# 5. FILTERS
# =====================================================

st.subheader("⚙️ Accident Filters")

col1, col2 = st.columns(2)

available_severity = sorted(
    accidents["severity"].unique().tolist()
)

with col1:

    selected_severity = st.multiselect(
        "Select Accident Severity",
        available_severity,
        default=available_severity,
        format_func=lambda x: severity_labels.get(
            x, f"Severity {x}"
        )
    )

with col2:

    if "start_time_hour" in accidents.columns:

        hour_filter = st.slider(
            "Select Time Range",
            min_value=0,
            max_value=23,
            value=(0, 23)
        )

    else:

        hour_filter = (0, 23)


# =====================================================
# 6. APPLY FILTERS
# =====================================================

filtered = accidents[
    accidents["severity"].isin(selected_severity)
].copy()

if "start_time_hour" in filtered.columns:

    filtered = filtered[
        filtered["start_time_hour"].between(
            hour_filter[0],
            hour_filter[1]
        )
    ].copy()


# =====================================================
# 7. KPI METRICS
# =====================================================

st.subheader("📊 Accident Overview")

total = len(filtered)

high_severity = int(
    filtered["severity"].isin([3, 4]).sum()
)

average_severity = (
    filtered["severity"].mean()
    if total > 0 else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Accidents",
    f"{total:,}"
)

c2.metric(
    "High Severity Records (3–4)",
    f"{high_severity:,}"
)

c3.metric(
    "Average Severity",
    f"{average_severity:.2f}"
)

c4.metric(
    "Available Records",
    f"{len(accidents):,}"
)


if filtered.empty:

    st.warning("No records match the selected filters.")

    st.stop()


# =====================================================
# 8. SEVERITY DISTRIBUTION
# =====================================================

st.divider()

st.subheader("📈 Accident Severity Distribution")

severity_counts = (
    filtered["severity"]
    .value_counts()
    .sort_index()
)

severity_counts.index = [
    severity_labels.get(x, str(x))
    for x in severity_counts.index
]

st.bar_chart(severity_counts)


# =====================================================
# 9. TIME ANALYSIS
# =====================================================

if "start_time_hour" in filtered.columns:

    st.subheader("🕒 Accidents by Hour")

    hourly = (
        filtered
        .groupby("start_time_hour")
        .size()
        .reindex(range(24), fill_value=0)
    )

    hourly.index.name = "Hour"

    st.bar_chart(hourly)


# =====================================================
# 10. MONTH ANALYSIS
# =====================================================

if "start_time_month" in filtered.columns:

    st.subheader("📅 Accidents by Month")

    monthly = (
        filtered
        .groupby("start_time_month")
        .size()
        .reindex(range(1, 13), fill_value=0)
    )

    monthly.index = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    st.bar_chart(monthly)


# =====================================================
# 11. MAP DATA
# =====================================================

st.divider()

st.subheader("🗺️ Accident Location Heatmap")

st.info(
    "This map shows accident locations from the US training sample. "
    "It does not show Bhopal accident locations."
)

# Limit map points for performance
map_limit = st.selectbox(
    "Map display limit",
    [1000, 3000, 5000, 10000],
    index=2
)

if len(filtered) > map_limit:

    map_data = filtered.sample(
        n=map_limit,
        random_state=42
    )

else:

    map_data = filtered.copy()


# Map center
center_lat = float(map_data["start_lat"].median())
center_lon = float(map_data["start_lng"].median())

accident_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="CartoDB positron",
    control_scale=True
)


# =====================================================
# 12. HEATMAP LAYER
# =====================================================

heat_data = []

for row in map_data.itertuples():

    heat_data.append([
        float(row.start_lat),
        float(row.start_lng),
        float(row.severity)
    ])


HeatMap(
    heat_data,
    radius=13,
    blur=18,
    min_opacity=0.35,
    max_zoom=10,
    name="Accident Heatmap"
).add_to(accident_map)


# =====================================================
# 13. ACCIDENT MARKERS
# =====================================================

marker_cluster = MarkerCluster(
    name="Accident Records"
)

# Use a smaller marker subset
marker_data = map_data.head(1500)

for row in marker_data.itertuples():

    severity = int(row.severity)

    color = severity_colors.get(
        severity,
        "#64748B"
    )

    popup_text = (
        f"Severity: {severity}\n"
        f"Latitude: {row.start_lat:.5f}\n"
        f"Longitude: {row.start_lng:.5f}"
    )

    folium.CircleMarker(
        location=[
            float(row.start_lat),
            float(row.start_lng)
        ],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        popup=popup_text
    ).add_to(marker_cluster)


marker_cluster.add_to(accident_map)

folium.LayerControl().add_to(accident_map)


st_folium(
    accident_map,
    use_container_width=True,
    height=650,
    returned_objects=[]
)


# =====================================================
# 14. WEATHER ANALYSIS
# =====================================================

if "weather_condition" in filtered.columns:

    st.divider()

    st.subheader("🌦️ Weather and Accidents")

    weather_data = (
        filtered["weather_condition"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(10)
    )

    st.bar_chart(weather_data)


# =====================================================
# 15. ROAD FEATURE ANALYSIS
# =====================================================

road_features = [
    "traffic_signal",
    "junction",
    "crossing",
    "railway",
    "stop",
    "traffic_calming"
]

available_features = [
    col for col in road_features
    if col in filtered.columns
]

if available_features:

    st.subheader("🚦 Road Infrastructure Features")

    feature_counts = {}

    for feature in available_features:

        values = (
            filtered[feature]
            .astype(str)
            .str.lower()
        )

        feature_counts[feature] = int(
            values.isin(["true", "1", "yes"]).sum()
        )

    st.bar_chart(
        pd.Series(feature_counts)
    )


# =====================================================
# 16. DATA TABLE
# =====================================================

st.divider()

st.subheader("📋 Accident Records")

display_columns = [
    col for col in [
        "severity",
        "start_lat",
        "start_lng",
        "start_time_hour",
        "start_time_day",
        "start_time_month",
        "weather_condition"
    ]
    if col in filtered.columns
]

st.dataframe(
    filtered[display_columns].head(500),
    use_container_width=True
)


# =====================================================
# 17. DOWNLOAD FILTERED DATA
# =====================================================

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Accident Data",
    data=csv_data,
    file_name="accident_analysis.csv",
    mime="text/csv"
)


# =====================================================
# 18. FOOTER
# =====================================================

st.divider()

st.caption(
    "TrafficSense AI | Accident Analytics Dashboard"
)

st.caption(
    "Dataset: US Accidents training sample. "
    "Severity and location patterns are descriptive, "
    "not proof of accident causes or future risk."
)
