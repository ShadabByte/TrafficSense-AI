
import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st

from shapely.geometry import shape
from shapely.ops import nearest_points
from streamlit_folium import st_folium


# =====================================================
# 1. PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Bhopal Interactive Traffic Heatmap",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Bhopal Interactive Traffic Heatmap")

st.caption(
    "Explore roads, select road segments and view demo congestion estimates."
)

st.warning(
    "Demo only: Congestion is estimated using illustrative rules. "
    "This is not live or verified Bhopal traffic data."
)


# =====================================================
# 2. PATH
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

ROAD_FILE = (
    ROOT / "data" / "processed" / "bhopal_roads.geojson"
)


# =====================================================
# 3. SAFE GEOJSON LOADER
# =====================================================

@st.cache_data
def load_roads(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    records = []

    for feature in data.get("features", []):

        properties = feature.get("properties") or {}
        geometry_data = feature.get("geometry")

        if not geometry_data:
            continue

        try:

            geometry = shape(geometry_data)

            if geometry.is_empty:
                continue

            row = {}

            for key, value in properties.items():

                if isinstance(value, (list, tuple)):
                    value = ", ".join(map(str, value))

                elif isinstance(value, dict):
                    value = json.dumps(value)

                elif value is None:
                    value = "other"

                row[key] = value

            row["geometry"] = geometry

            records.append(row)

        except Exception:
            continue

    if not records:
        raise ValueError("No valid road features found.")

    roads = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326"
    )

    for column, default in [
        ("highway", "other"),
        ("name", "Unnamed Road")
    ]:

        if column not in roads.columns:
            roads[column] = default

        roads[column] = (
            roads[column]
            .fillna(default)
            .astype(str)
        )

    roads["highway"] = roads["highway"].str.lower()

    # Stable identifier for selecting roads
    roads = roads.reset_index(drop=True)
    roads["road_index"] = roads.index.astype(int)

    return roads


if not ROAD_FILE.exists():

    st.error(f"Road file not found: {ROAD_FILE}")
    st.stop()


try:

    roads = load_roads(str(ROAD_FILE))

except Exception as error:

    st.error("Road data loading failed.")
    st.exception(error)
    st.stop()


# =====================================================
# 4. DEMO CONGESTION
# =====================================================

def estimate_congestion(road_type, hour, day):

    score = 0

    major = [
        "motorway", "trunk", "primary",
        "secondary", "tertiary"
    ]

    medium = [
        "residential", "unclassified",
        "living_street"
    ]

    if road_type in major:
        score += 2

    elif road_type in medium:
        score += 1

    if 8 <= hour <= 11:
        score += 2

    elif 17 <= hour <= 20:
        score += 3

    elif 12 <= hour <= 14:
        score += 1

    if day in [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday"
    ]:
        score += 1

    if score >= 5:
        return "HEAVY"

    elif score >= 4:
        return "MODERATE"

    elif score >= 2:
        return "LIGHT"

    return "FREE"


COLORS = {
    "FREE": "#16A34A",
    "LIGHT": "#EAB308",
    "MODERATE": "#F97316",
    "HEAVY": "#DC2626"
}


# =====================================================
# 5. SETTINGS
# =====================================================

st.subheader("⚙️ Traffic Settings")

col1, col2 = st.columns(2)

with col1:

    hour = st.slider(
        "Select Hour",
        0, 23, 18
    )

with col2:

    day = st.selectbox(
        "Select Day",
        [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]
    )


road_types = sorted(roads["highway"].unique())

selected_type = st.selectbox(
    "Filter Road Type",
    ["All Roads"] + road_types
)


# =====================================================
# 6. ESTIMATE ALL ROADS
# =====================================================

roads = roads.copy()

roads["congestion"] = roads["highway"].apply(
    lambda x: estimate_congestion(x, hour, day)
)


# =====================================================
# 7. SELECTED ROAD STATE
# =====================================================

if "selected_bhopal_road" not in st.session_state:
    st.session_state.selected_bhopal_road = None

selected_id = st.session_state.selected_bhopal_road


# =====================================================
# 8. FILTER
# =====================================================

if selected_type == "All Roads":

    map_roads = roads.copy()

else:

    map_roads = roads[
        roads["highway"] == selected_type
    ].copy()


# =====================================================
# 9. METRICS
# =====================================================

counts = roads["congestion"].value_counts()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Roads", f"{len(roads):,}")
c2.metric("FREE", int(counts.get("FREE", 0)))
c3.metric("MODERATE", int(counts.get("MODERATE", 0)))
c4.metric("HEAVY", int(counts.get("HEAVY", 0)))


# =====================================================
# 10. BUILD MAP
# =====================================================

st.subheader("🗺️ Click Near Any Road to Select It")

if map_roads.empty:

    st.warning("No roads found.")
    st.stop()


bounds = map_roads.total_bounds

center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

traffic_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles="CartoDB positron",
    control_scale=True
)


# Add road features with their index in the tooltip
map_data = map_roads.copy()

map_data["road_label"] = (
    map_data["name"].astype(str)
    + " | "
    + map_data["highway"].astype(str)
)

def style_road(feature):

    index = feature["properties"]["road_index"]

    color = feature["properties"]["congestion"]

    if selected_id is not None and index == selected_id:

        return {
            "color": "#2563EB",
            "weight": 8,
            "opacity": 1
        }

    return {
        "color": COLORS.get(color, "#64748B"),
        "weight": 3,
        "opacity": 0.85
    }


folium.GeoJson(
    json.loads(map_data.to_json()),
    name="Road Network",
    style_function=style_road,
    highlight_function=lambda feature: {
        "weight": 6,
        "opacity": 1
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["road_label", "congestion"],
        aliases=["Road:", "Demo Congestion:"],
        sticky=True
    )
).add_to(traffic_map)


folium.LayerControl().add_to(traffic_map)


# =====================================================
# 11. DISPLAY MAP AND CAPTURE CLICK
# =====================================================

map_result = st_folium(
    traffic_map,
    use_container_width=True,
    height=650,
    returned_objects=["last_clicked"]
)


# =====================================================
# 12. FIND NEAREST ROAD
# =====================================================

if map_result and map_result.get("last_clicked"):

    clicked = map_result["last_clicked"]

    click_lat = clicked.get("lat")
    click_lon = clicked.get("lng")

    if click_lat is not None and click_lon is not None:

        click_signature = (
            round(float(click_lat), 6),
            round(float(click_lon), 6)
        )

        if (
            st.session_state.get("last_road_click")
            != click_signature
        ):

            st.session_state.last_road_click = click_signature

            try:

                click_point = gpd.GeoSeries(
                    gpd.points_from_xy(
                        [click_lon],
                        [click_lat]
                    ),
                    crs="EPSG:4326"
                ).to_crs("EPSG:3857").iloc[0]

                projected = map_roads.to_crs("EPSG:3857")

                distances = projected.geometry.distance(
                    click_point
                )

                nearest_position = distances.idxmin()

                nearest_distance = float(
                    distances.loc[nearest_position]
                )

                # Click must be reasonably close to a road
                if nearest_distance <= 150:

                    selected_road_index = int(
                        map_roads.loc[
                            nearest_position,
                            "road_index"
                        ]
                    )

                    st.session_state.selected_bhopal_road = (
                        selected_road_index
                    )

                    st.rerun()

                else:

                    st.info(
                        "No road found close to that map location. "
                        "Try clicking nearer to a road."
                    )

            except Exception as error:

                st.error(f"Road selection error: {error}")


# =====================================================
# 13. SELECTED ROAD DETAILS
# =====================================================

st.divider()

st.subheader("📍 Selected Road Details")

if selected_id is not None:

    selected_rows = roads[
        roads["road_index"] == selected_id
    ]

    if not selected_rows.empty:

        road = selected_rows.iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Road Name",
            str(road["name"])[:35]
        )

        b.metric(
            "Road Type",
            str(road["highway"]).upper()
        )

        c.metric(
            "Demo Congestion",
            str(road["congestion"])
        )

        st.markdown(
            f"**Selected Road:** {road['name']}"
        )

        st.markdown(
            f"**Congestion Status:** "
            f"<span style='color:{COLORS[road['congestion']]};'>"
            f"● {road['congestion']}</span>",
            unsafe_allow_html=True
        )

        if "length" in roads.columns:

            st.write(
                "**Road Segment Length (OSM data):**",
                road["length"]
            )

        if st.button("Clear Selected Road"):

            st.session_state.selected_bhopal_road = None
            st.session_state.last_road_click = None
            st.rerun()

    else:

        st.info("Selected road is not available.")

else:

    st.info(
        "No road selected yet. Click near a road on the map "
        "to view its details."
    )


# =====================================================
# 14. CONGESTION CHART
# =====================================================

st.divider()

st.subheader("📊 Congestion Distribution")

chart = (
    roads["congestion"]
    .value_counts()
    .reindex(
        ["FREE", "LIGHT", "MODERATE", "HEAVY"],
        fill_value=0
    )
)

st.bar_chart(chart)


# =====================================================
# 15. ROAD TYPE CHART
# =====================================================

st.subheader("🛣️ Road Type Distribution")

st.bar_chart(
    roads["highway"].value_counts().head(10)
)


# =====================================================
# 16. FOOTER
# =====================================================

st.divider()

st.caption(
    "TrafficSense AI | Bhopal OpenStreetMap Road Network"
)

st.caption(
    "Congestion levels are illustrative rule-based estimates, "
    "not live measurements or validated predictions."
)
