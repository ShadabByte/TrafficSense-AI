import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import folium
from datetime import date

from streamlit_folium import st_folium


# =====================================================
# 1. PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="TrafficSense AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAFFIC_MODEL_FILE = os.path.join(
    BASE_DIR, "models", "traffic_congestion_model.pkl"
)

ACCIDENT_MODEL_FILE = os.path.join(
    BASE_DIR, "models", "accident_severity_model.pkl"
)

TRAFFIC_DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_engineered",
    "indore_ml_dataset_features.csv"
)

ACCIDENT_DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "accident_training_sample.csv"
)

BHOPAL_MAP_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "bhopal_roads.geojson"
)


# =====================================================
# 3. CSS - PREMIUM TRAFFICSENSE UI
# =====================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #080f1e, #101d33, #0b1628);
    color: #f1f5f9;
}

/* REMOVE WHITE TOP HEADER */
header[data-testid="stHeader"] {
    background: #080f1e !important;
}

div[data-testid="stToolbar"] {
    background: transparent !important;
}

/* MAIN CONTENT */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #101a2c !important;
    border-right: 1px solid #263b58;
}

section[data-testid="stSidebar"] * {
    color: #e2eaf7 !important;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1b2b45, #17243a) !important;
    border: 1px solid #344b6c !important;
    border-radius: 16px !important;
    padding: 18px !important;
    box-shadow: 0 5px 18px rgba(0,0,0,0.18);
}

/* METRIC LABELS - IMPORTANT FIX */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label * {
    color: #cbdaf0 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* METRIC VALUES */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #60a5fa !important;
    font-weight: 800 !important;
}

/* ALL HEADINGS */
h1, h2, h3, h4 {
    color: #f1f5f9 !important;
    font-weight: 750 !important;
}

/* NORMAL TEXT */
.stApp p,
.stApp li,
.stApp .stMarkdown {
    color: #dbeafe;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0891b2) !important;
    color: #ffffff !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 9px 20px !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0e7490) !important;
    box-shadow: 0 0 14px rgba(56,189,248,0.3);
}

/* INPUT FIELDS */
.stSelectbox div[data-baseweb="select"],
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #1b2b45 !important;
    color: #ffffff !important;
    border: 1px solid #405a7d !important;
    border-radius: 9px !important;
}

/* INPUT LABELS */
.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stSlider label,
.stRadio label {
    color: #dbeafe !important;
    font-weight: 600 !important;
}

/* DATAFRAMES */
div[data-testid="stDataFrame"] {
    border: 1px solid #344b6c;
    border-radius: 12px;
}

/* ALERT BOXES */
div[data-testid="stAlert"] {
    background: #1b2b45 !important;
    border: 1px solid #3b587d !important;
    border-radius: 12px !important;
}

div[data-testid="stAlert"] p {
    color: #e2eaf7 !important;
}

/* DIVIDERS */
hr {
    border-color: #344b6c !important;
}

/* CAPTIONS */
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #a9bed9 !important;
}

/* LINKS */
.stApp a {
    color: #38bdf8 !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #3b82f6;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)



# =====================================================
# 4. NORMALIZE VALUES
# =====================================================

def normalize_value(value, default="other"):

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else default

    if value is None:
        return default

    try:
        missing = pd.isna(value)

        if isinstance(missing, (bool, np.bool_)) and missing:
            return default

    except (TypeError, ValueError):
        pass

    return str(value)


# =====================================================
# 5. LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    traffic_model = None
    accident_model = None

    if os.path.exists(TRAFFIC_MODEL_FILE):
        traffic_model = joblib.load(TRAFFIC_MODEL_FILE)

    if os.path.exists(ACCIDENT_MODEL_FILE):
        accident_model = joblib.load(ACCIDENT_MODEL_FILE)

    return traffic_model, accident_model


# =====================================================
# 6. LOAD DATASETS
# =====================================================

@st.cache_data
def load_data():

    traffic_data = pd.DataFrame()
    accident_data = pd.DataFrame()

    if os.path.exists(TRAFFIC_DATA_FILE):
        traffic_data = pd.read_csv(TRAFFIC_DATA_FILE)

    if os.path.exists(ACCIDENT_DATA_FILE):
        accident_data = pd.read_csv(ACCIDENT_DATA_FILE)

    return traffic_data, accident_data


# =====================================================
# 7. LOAD BHOPAL ROADS
# =====================================================

@st.cache_data
def load_bhopal_roads():

    if not os.path.exists(BHOPAL_MAP_FILE):
        raise FileNotFoundError(
            "Bhopal GeoJSON file not found."
        )

    roads = gpd.read_file(BHOPAL_MAP_FILE)

    if roads.crs is not None and roads.crs.to_epsg() != 4326:
        roads = roads.to_crs(epsg=4326)

    if "highway" not in roads.columns:
        roads["highway"] = "other"

    if "name" not in roads.columns:
        roads["name"] = None

    if "u" not in roads.columns:
        roads["u"] = roads.index.astype(str)

    if "v" not in roads.columns:
        roads["v"] = roads.index.astype(str)

    if "key" not in roads.columns:
        roads["key"] = 0

    roads["highway"] = roads["highway"].apply(
        lambda x: normalize_value(x, "other")
    )

    roads["name"] = roads["name"].apply(
        lambda x: normalize_value(x, "")
    )

    roads["road_id"] = (
        roads["u"].astype(str)
        + "_"
        + roads["v"].astype(str)
        + "_"
        + roads["key"].astype(str)
    )

    roads["display_name"] = roads.apply(
        lambda row:
            row["name"]
            if row["name"].strip()
            else row["highway"].replace("_", " ").title()
            + " Road",
        axis=1
    )

    roads = roads[
        roads.geometry.notna()
        & (~roads.geometry.is_empty)
    ].copy()

    return roads


# =====================================================
# 8. BHOPAL DEMO CONGESTION ESTIMATOR
# =====================================================

def estimate_bhopal_congestion(road_type, hour, day):

    road_type = normalize_value(
        road_type, "other"
    ).lower()

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

    elif score == 2:
        return "LIGHT"

    elif score == 3:
        return "MODERATE"

    return "HEAVY"


# =====================================================
# 9. CONGESTION COLORS
# =====================================================

def get_congestion_color(category):

    colors = {
        "FREE": "#2ecc71",
        "LIGHT": "#f1c40f",
        "MODERATE": "#e67e22",
        "HEAVY": "#e74c3c"
    }

    return colors.get(category, "#808080")


# =====================================================
# 10. ROAD COLORS
# =====================================================

ROAD_COLORS = {
    "motorway": "#e63946",
    "trunk": "#f77f00",
    "primary": "#ff9f1c",
    "secondary": "#e9c46a",
    "tertiary": "#2a9d8f",
    "residential": "#457b9d",
    "service": "#8d99ae",
    "unclassified": "#6c757d",
    "footway": "#adb5bd",
    "living_street": "#6c757d",
    "track": "#8a5a44"
}


# =====================================================
# 11. LOAD RESOURCES
# =====================================================

try:

    traffic_model, accident_model = load_models()

    traffic_data, accident_data = load_data()

except Exception as e:

    st.error(f"Resource loading error: {e}")
    st.stop()

# =====================================================
# 12. HEADER
# =====================================================

col_logo, col_status = st.columns([3, 1])

with col_logo:

    st.markdown(
        """
        <div style="
            background: linear-gradient(120deg, #17243a, #1e3a5f);
            padding: 25px;
            border-radius: 16px;
            border: 1px solid #334e70;
        ">
            <div style="
                font-size: 34px;
                font-weight: 800;
                color: #60a5fa;
            ">
                🚦 TrafficSense AI
            </div>
            <div style="
                font-size: 15px;
                color: #cbd5e1;
                margin-top: 6px;
            ">
                Smart Traffic & Accident Intelligence Platform
            </div>
            <div style="
                font-size: 12px;
                color: #94a3b8;
                margin-top: 10px;
            ">
                BHOPAL • SMART MOBILITY • AI ANALYTICS
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_status:

    st.markdown(
        """
        <div style="
            background: #17243a;
            padding: 25px;
            border-radius: 16px;
            border: 1px solid #334e70;
            text-align: center;
        ">
            <div style="font-size: 25px;">🟢</div>
            <div style="
                color: #4ade80;
                font-size: 18px;
                font-weight: 700;
            ">
                SYSTEM READY
            </div>
            <div style="
                color: #94a3b8;
                font-size: 12px;
            ">
                Research Prototype
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# =====================================================
# 13. SIDEBAR
# =====================================================

st.sidebar.title("🚦 TrafficSense AI")

st.sidebar.caption(
    "Smart Mobility Intelligence"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Traffic Congestion",
        "Bhopal Traffic Estimator",
        "Accident Severity",
        "Accident Severity Predictor",
        "Accident Analytics",
        "Accident Hotspot Map",
        "AI Traffic Intelligence",
        "Bhopal Smart Route Analyzer",
        "Bhopal Traffic Data Collector",
        "Bhopal Dataset Quality Monitor",
        "Bhopal Live Traffic",
        "Bhopal Road Network",
        "Project Evaluation & Documentation"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")

st.sidebar.write("📍 Bhopal, Madhya Pradesh")
st.sidebar.write("🧠 Machine Learning")
st.sidebar.write("🗺️ OpenStreetMap")
st.sidebar.write("🐍 Python + Streamlit")

st.sidebar.divider()

st.sidebar.caption(
    "TrafficSense AI | Research Prototype"
)


# =====================================================
# 14. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.subheader("📊 Project Overview")

    road_count = 0

    if os.path.exists(BHOPAL_MAP_FILE):

        try:
            road_count = len(load_bhopal_roads())

        except Exception:
            pass

    model_count = sum([
        traffic_model is not None,
        accident_model is not None
    ])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Traffic Records", f"{len(traffic_data):,}")
    c2.metric("Accident Records", f"{len(accident_data):,}")
    c3.metric("Bhopal Road Segments", f"{road_count:,}")
    c4.metric("Trained Models", model_count)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🚗 Traffic Congestion")
        st.write(
            "Machine learning based traffic congestion "
            "classification using Indore road and time features."
        )

    with c2:
        st.markdown("### 🚨 Accident Severity")
        st.write(
            "Classification of accident severity using "
            "weather, time and location features."
        )

    st.markdown("### 🗺️ Bhopal Road Network")

    st.write(
        "Explore road infrastructure and road segments "
        "extracted from OpenStreetMap."
    )

    st.info(
        "Research prototype: this application does not "
        "provide live traffic measurements or verified "
        "real-world accident forecasts."
    )


# =====================================================
# 15. TRAFFIC CONGESTION - INDORE MODEL
# =====================================================

elif page == "Traffic Congestion":

    st.subheader("🚗 Traffic Congestion - ML Model")

    st.warning(
        "This model was trained on Indore traffic data. "
        "Its predictions are not verified Bhopal traffic forecasts."
    )

    if traffic_model is None or traffic_data.empty:
        st.error("Traffic model or dataset not found.")
        st.stop()

    required_columns = ["road_id", "hour", "day"]

    if not all(
        column in traffic_data.columns
        for column in required_columns
    ):
        st.error("Required traffic dataset columns are missing.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    road_options = sorted(
        traffic_data["road_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    day_options = sorted(
        traffic_data["day"]
        .dropna()
        .unique()
        .tolist()
    )

    with c1:
        road = st.selectbox("Indore Road ID", road_options)

    with c2:
        hour = st.slider("Hour", 0, 23, 9)

    with c3:
        day = st.selectbox("Day", day_options)

    if st.button("Predict Congestion", type="primary"):

        input_data = pd.DataFrame([{
            "road_id": road,
            "hour": hour,
            "day": day
        }])

        try:

            prediction = traffic_model.predict(input_data)[0]

            probabilities = traffic_model.predict_proba(input_data)[0]

            classes = traffic_model.classes_

            st.divider()
            st.subheader("Prediction Result")

            st.success(f"Predicted Traffic: {prediction}")

            chart_data = pd.DataFrame({
                "Congestion": classes,
                "Probability (%)": probabilities * 100
            })

            st.bar_chart(
                chart_data.set_index("Congestion")
            )

            st.caption(
                "Trained on Indore traffic data. "
                "Not a verified Bhopal traffic forecast."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# =====================================================
# 16. BHOPAL TRAFFIC ESTIMATOR
# =====================================================

elif page == "Bhopal Traffic Estimator":

    st.subheader("🚦 Bhopal Traffic Congestion Estimator")

    st.warning(
        "DEMO ESTIMATE ONLY — This is a rule-based illustration, "
        "not live traffic data or a trained Bhopal traffic prediction."
    )

    try:

        roads = load_bhopal_roads()

        if roads.empty:
            st.error("No Bhopal road data available.")
            st.stop()

        road_options = roads[
            ["road_id", "display_name", "highway"]
        ].copy()

        road_options["selection_label"] = (
            road_options["display_name"].astype(str)
            + " | "
            + road_options["highway"].astype(str)
            + " | "
            + road_options["road_id"].astype(str)
        )

        st.markdown("### Select Road and Time")

        c1, c2, c3 = st.columns(3)

        with c1:

            current_id = st.session_state.get(
                "bhopal_selected_road_id",
                road_options.iloc[0]["road_id"]
            )

            matching = road_options.index[
                road_options["road_id"] == current_id
            ].tolist()

            default_index = matching[0] if matching else 0

            selected_label = st.selectbox(
                "Select Bhopal Road",
                road_options["selection_label"].tolist(),
                index=default_index
            )

        selected_row = road_options[
            road_options["selection_label"] == selected_label
        ].iloc[0]

        selected_road_id = selected_row["road_id"]

        st.session_state["bhopal_selected_road_id"] = selected_road_id

        with c2:

            selected_hour = st.slider(
                "Select Hour",
                min_value=0,
                max_value=23,
                value=18
            )

        with c3:

            selected_day = st.selectbox(
                "Select Day",
                [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ]
            )

        road_type = selected_row["highway"]

        congestion = estimate_bhopal_congestion(
            road_type,
            selected_hour,
            selected_day
        )

        st.divider()
        st.subheader("📊 Demo Estimate Result")

        col1, col2, col3 = st.columns(3)

        col1.metric("Selected Road", selected_row["display_name"])

        col2.metric(
            "Road Category",
            road_type.replace("_", " ").title()
        )

        col3.metric("Estimated Category", congestion)

        st.markdown(
            f"""
            <div style="
                background:{get_congestion_color(congestion)};
                padding:18px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:24px;
                font-weight:bold;
                margin:12px 0;
            ">
                {congestion} CONGESTION
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Selected time: {selected_hour:02d}:00 | Day: {selected_day}"
        )

        st.info(
            "This category is generated from illustrative rules based "
            "on road type and selected time. It is not measured traffic."
        )

        # Calculate demo category for every road
        roads["demo_category"] = roads["highway"].apply(
            lambda road_type_value: estimate_bhopal_congestion(
                road_type_value,
                selected_hour,
                selected_day
            )
        )

        roads["is_selected"] = (
            roads["road_id"].astype(str) == str(selected_road_id)
        )

        st.divider()
        st.subheader("🗺️ Interactive Bhopal Road Map")

        st.caption(
            "Map par kisi bhi road segment par click karo. "
            "Nearest road select hogi aur uski details update hongi."
        )

        traffic_map = folium.Map(
            location=[23.2599, 77.4126],
            zoom_start=12,
            tiles="OpenStreetMap",
            control_scale=True
        )

        def traffic_road_style(feature):

            properties = feature.get("properties", {})

            if properties.get("is_selected", False):

                category = properties.get("demo_category", "FREE")

                return {
                    "color": get_congestion_color(category),
                    "weight": 7,
                    "opacity": 1.0
                }

            road_type_value = normalize_value(
                properties.get("highway", "other")
            )

            return {
                "color": ROAD_COLORS.get(
                    road_type_value,
                    "#8a9aaa"
                ),
                "weight": 2,
                "opacity": 0.75
            }

        folium.GeoJson(
            roads.to_json(),
            name="All Bhopal Roads",
            style_function=traffic_road_style,
            highlight_function=lambda feature: {
                "weight": 5,
                "opacity": 1.0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "display_name",
                    "highway",
                    "demo_category"
                ],
                aliases=[
                    "Road:",
                    "Road Type:",
                    "Demo Category:"
                ],
                sticky=True
            )
        ).add_to(traffic_map)

        minx, miny, maxx, maxy = roads.total_bounds

        traffic_map.fit_bounds([
            [miny, minx],
            [maxy, maxx]
        ])

        map_result = st_folium(
            traffic_map,
            width=None,
            height=600,
            key="bhopal_traffic_estimator_map",
            returned_objects=["last_clicked"]
        )

        # Map click: find nearest road segment
        clicked = map_result.get("last_clicked") if map_result else None

        if clicked and "lat" in clicked and "lng" in clicked:

            click_signature = (
                round(float(clicked["lat"]), 5),
                round(float(clicked["lng"]), 5)
            )

            previous_signature = st.session_state.get(
                "bhopal_last_processed_click"
            )

            if click_signature != previous_signature:

                try:

                    clicked_point = gpd.GeoSeries(
                        [
                            gpd.points_from_xy(
                                [float(clicked["lng"])],
                                [float(clicked["lat"])]
                            )[0]
                        ],
                        crs="EPSG:4326"
                    ).to_crs(epsg=3857).iloc[0]

                    projected_roads = roads.to_crs(epsg=3857)

                    distances = projected_roads.geometry.distance(
                        clicked_point
                    )

                    nearest_index = distances.idxmin()

                    nearest_distance = float(
                        distances.loc[nearest_index]
                    )

                    if nearest_distance <= 200:

                        new_road_id = roads.loc[
                            nearest_index, "road_id"
                        ]

                        st.session_state[
                            "bhopal_selected_road_id"
                        ] = new_road_id

                        st.session_state[
                            "bhopal_last_processed_click"
                        ] = click_signature

                        st.rerun()

                    else:

                        st.session_state[
                            "bhopal_last_processed_click"
                        ] = click_signature

                        st.warning(
                            "Road se door click hua hai. "
                            "Zoom karke road line par click karo."
                        )

                except Exception as click_error:

                    st.warning(
                        f"Road selection error: {click_error}"
                    )

        st.caption(
            "Selected road congestion color mein highlight hogi. "
            "Baaki roads apne road type ke color mein dikhengi."
        )

        # Distribution
        st.divider()
        st.subheader("📈 Estimated Road Category Distribution")

        category_order = [
            "FREE", "LIGHT", "MODERATE", "HEAVY"
        ]

        distribution = (
            roads["demo_category"]
            .value_counts()
            .reindex(category_order, fill_value=0)
        )

        st.bar_chart(distribution)

        total_roads = len(roads)
        metric_cols = st.columns(4)

        for metric_col, category in zip(
            metric_cols, category_order
        ):

            count = int(distribution[category])

            percentage = (
                count / total_roads * 100
                if total_roads else 0
            )

            metric_col.metric(
                f"{category} roads",
                f"{count:,}",
                f"{percentage:.1f}%"
            )

        # Top 10 demo roads
        st.subheader("🚦 Top 10 Highest Demo Congestion Roads")

        roads["demo_score"] = roads["demo_category"].map({
            "FREE": 1,
            "LIGHT": 2,
            "MODERATE": 3,
            "HEAVY": 4
        })

        top_roads = roads.nlargest(
            10, ["demo_score", "length"]
        )[
            ["display_name", "highway", "demo_category", "length"]
        ].copy()

        top_roads = top_roads.rename(columns={
            "display_name": "Road",
            "highway": "Road Type",
            "demo_category": "Demo Category",
            "length": "Length (m)"
        })

        top_roads["Length (m)"] = (
            top_roads["Length (m)"].round(1)
        )

        st.dataframe(
            top_roads.reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These rankings are based on illustrative rules, "
            "not observed or live traffic data."
        )

    except Exception as e:

        st.error(f"Bhopal traffic estimator error: {e}")
        st.code(str(e))


# =====================================================
# 17. ACCIDENT SEVERITY
# =====================================================

elif page == "Accident Severity":

    st.subheader("🚨 Accident Severity Classification")

    st.warning(
        "This model was trained on US accident records. "
        "Results are not validated Bhopal forecasts."
    )

    if accident_model is None or accident_data.empty:
        st.error("Accident model or dataset not found.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:

        temperature = st.number_input(
            "Temperature (°F)",
            value=70.0
        )

        humidity = st.slider(
            "Humidity (%)",
            0,
            100,
            50
        )

        visibility = st.number_input(
            "Visibility (miles)",
            min_value=0.0,
            value=10.0
        )

        wind_speed = st.number_input(
            "Wind Speed (mph)",
            min_value=0.0,
            value=5.0
        )

    with c2:

        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Cloudy",
                "Rain",
                "Snow",
                "Fog",
                "Other"
            ]
        )

        time_of_day = st.selectbox(
            "Time of Day",
            ["Day", "Night"]
        )

        hour = st.slider(
            "Hour",
            0,
            23,
            12
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13))
        )

    with c3:

        latitude = st.number_input(
            "Latitude",
            value=23.2599,
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            value=77.4126,
            format="%.6f"
        )

        day_of_week = st.selectbox(
            "Day of Week",
            list(range(7))
        )

    if st.button("Classify Severity", type="primary"):

        input_row = {}

        for column in accident_data.columns:

            if column == "Severity":
                continue

            if pd.api.types.is_numeric_dtype(
                accident_data[column]
            ):

                value = accident_data[column].median()

                input_row[column] = (
                    0 if pd.isna(value) else value
                )

            else:

                modes = accident_data[column].mode()

                input_row[column] = (
                    modes.iloc[0]
                    if len(modes)
                    else "Unknown"
                )

        user_values = {
            "Temperature(F)": temperature,
            "Humidity(%)": humidity,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Weather_Condition": weather,
            "Sunrise_Sunset": time_of_day,
            "Start_Lat": latitude,
            "Start_Lng": longitude,
            "Start_Time_hour": hour,
            "Start_Time_month": month,
            "Start_Time_dayofweek": day_of_week
        }

        for column, value in user_values.items():

            if column in input_row:
                input_row[column] = value

        input_data = pd.DataFrame([input_row])

        try:

            prediction = accident_model.predict(input_data)[0]

            probabilities = accident_model.predict_proba(input_data)[0]

            classes = accident_model.classes_

            labels = {
                1: "Low Severity",
                2: "Moderate Severity",
                3: "High Severity",
                4: "Very High Severity"
            }

            result = labels.get(
                int(prediction),
                f"Severity {prediction}"
            )

            st.divider()
            st.subheader("Classification Result")

            st.warning(f"Predicted Category: {result}")

            chart_data = pd.DataFrame({
                "Severity": [
                    labels.get(int(x), str(x))
                    for x in classes
                ],
                "Probability (%)": probabilities * 100
            })

            st.bar_chart(
                chart_data.set_index("Severity")
            )

            st.caption(
                "Model trained on US accident data. "
                "It classifies severity conditional on an accident "
                "scenario; it does not predict accident occurrence "
                "or provide verified Bhopal forecasts."
            )

        except Exception as e:
            st.error(f"Classification error: {e}")


# =====================================================
# ACCIDENT SEVERITY PREDICTOR
# =====================================================

elif page == "Accident Severity Predictor":

    st.title("🚨 Accident Severity Predictor")
    st.markdown(
        "Predict the possible severity of an accident scenario "
        "using weather, road and time-related conditions."
    )

    st.warning(
        "This model is trained on US accident data. "
        "It is an educational prediction demo and has not been "
        "validated for real-world Bhopal conditions."
    )

    if accident_model is None or accident_data is None:
        st.error("Accident model or training data could not be loaded.")
        st.stop()

    # Get exact model feature names
    if hasattr(accident_model, "feature_names_in_"):
        model_features = list(accident_model.feature_names_in_)
    else:
        model_features = [
            col for col in accident_data.columns
            if col != "severity"
        ]

    missing_columns = [
        col for col in model_features
        if col not in accident_data.columns
    ]

    if missing_columns:
        st.error(
            f"Required training columns are missing: {missing_columns}"
        )
        st.stop()

    st.subheader("📍 Accident Location")

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude",
            value=35.838308,
            format="%.6f"
        )

    with col2:
        longitude = st.number_input(
            "Longitude",
            value=-87.766426,
            format="%.6f"
        )

    st.caption(
        "Default coordinates are based on the US training sample. "
        "You can enter other coordinates for experimentation."
    )

    st.divider()

    st.subheader("🕒 Date and Time")

    col1, col2 = st.columns(2)

    with col1:
        accident_date = st.date_input(
            "Accident Date",
            value=date.today()
        )

    with col2:
        accident_hour = st.slider(
            "Accident Hour",
            min_value=0,
            max_value=23,
            value=17
        )

    day_night = st.selectbox(
        "Day or Night",
        ["Day", "Night"]
    )

    st.divider()

    st.subheader("🌦️ Weather Conditions")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature (°F)",
            value=64.0
        )

        wind_chill = st.number_input(
            "Wind Chill (°F)",
            value=64.0
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=50
        )

        pressure = st.number_input(
            "Pressure (in)",
            value=29.90
        )

    with col2:
        visibility = st.number_input(
            "Visibility (miles)",
            min_value=0.0,
            value=10.0
        )

        wind_speed = st.number_input(
            "Wind Speed (mph)",
            min_value=0.0,
            value=5.0
        )

        precipitation = st.number_input(
            "Precipitation (in)",
            min_value=0.0,
            value=0.0
        )

        weather_condition = st.selectbox(
            "Weather Condition",
            [
                "Fair",
                "Clear",
                "Cloudy",
                "Mostly Cloudy",
                "Partly Cloudy",
                "Overcast",
                "Light Rain",
                "Rain",
                "Heavy Rain",
                "Fog",
                "Snow"
            ]
        )

    wind_direction = st.selectbox(
        "Wind Direction",
        [
            "CALM", "N", "NNE", "NE", "ENE",
            "E", "ESE", "SE", "SSE", "S",
            "SSW", "SW", "WSW", "W", "WNW",
            "NW", "NNW", "VAR"
        ]
    )

    st.divider()

    st.subheader("🛣️ Road Infrastructure")

    col1, col2 = st.columns(2)

    infrastructure_features = [
        "amenity",
        "bump",
        "crossing",
        "give_way",
        "junction",
        "no_exit",
        "railway",
        "roundabout",
        "station",
        "stop",
        "traffic_calming",
        "traffic_signal",
        "turning_loop"
    ]

    infrastructure_values = {}

    with col1:
        for feature in infrastructure_features[:7]:
            infrastructure_values[feature] = st.checkbox(
                feature.replace("_", " ").title(),
                value=False
            )

    with col2:
        for feature in infrastructure_features[7:]:
            infrastructure_values[feature] = st.checkbox(
                feature.replace("_", " ").title(),
                value=False
            )

    st.divider()

    st.subheader("🌅 Twilight Conditions")

    col1, col2 = st.columns(2)

    with col1:
        civil_twilight = st.selectbox(
            "Civil Twilight",
            ["Day", "Night"]
        )

        nautical_twilight = st.selectbox(
            "Nautical Twilight",
            ["Day", "Night"]
        )

    with col2:
        astronomical_twilight = st.selectbox(
            "Astronomical Twilight",
            ["Day", "Night"]
        )

    st.divider()

    # Create a row using training data defaults
    input_row = {}

    for feature in model_features:

        if pd.api.types.is_numeric_dtype(accident_data[feature]):
            input_row[feature] = accident_data[feature].median()

        else:
            mode_values = accident_data[feature].mode()

            if len(mode_values) > 0:
                input_row[feature] = mode_values.iloc[0]
            else:
                input_row[feature] = "Unknown"

    # Apply user inputs
    user_values = {
        "start_lat": latitude,
        "start_lng": longitude,
        "temperature(f)": temperature,
        "wind_chill(f)": wind_chill,
        "humidity(%)": humidity,
        "pressure(in)": pressure,
        "visibility(mi)": visibility,
        "wind_speed(mph)": wind_speed,
        "precipitation(in)": precipitation,
        "weather_condition": weather_condition,
        "wind_direction": wind_direction,
        "sunrise_sunset": day_night,
        "start_time_hour": accident_hour,
        "start_time_day": accident_date.day,
        "start_time_month": accident_date.month,
        "start_time_weekday": accident_date.weekday(),
        "start_time_weekend": int(accident_date.weekday() >= 5),
        "start_time_rush_hour": int(
            accident_hour in [7, 8, 9, 16, 17, 18, 19]
        ),
        "start_time_hour_sin": np.sin(
            2 * np.pi * accident_hour / 24
        ),
        "start_time_hour_cos": np.cos(
            2 * np.pi * accident_hour / 24
        ),
        "civil_twilight": civil_twilight,
        "nautical_twilight": nautical_twilight,
        "astronomical_twilight": astronomical_twilight
    }

    user_values.update(infrastructure_values)

    for feature, value in user_values.items():
        if feature in input_row:
            input_row[feature] = value

    # Arrange columns in the exact model order
    input_df = pd.DataFrame(
        [input_row],
        columns=model_features
    )

    st.subheader("🔍 Prediction")

    if st.button(
        "🚨 Predict Accident Severity",
        type="primary",
        use_container_width=True
    ):

        try:
            prediction = accident_model.predict(input_df)[0]

            probabilities = accident_model.predict_proba(input_df)[0]

            classes = accident_model.classes_

            probability_df = pd.DataFrame({
                "Severity": classes,
                "Probability (%)": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                "Probability (%)",
                ascending=False
            )
            st.success("Prediction completed successfully!")

            severity_details = {
                1: {
                    "name": "Minor Severity",
                    "risk": "LOW",
                    "description": "Lowest severity category in the training dataset."
                },
                2: {
                    "name": "Moderate Severity",
                    "risk": "MODERATE",
                    "description": "Moderate severity category in the training dataset."
                },
                3: {
                    "name": "High Severity",
                    "risk": "HIGH",
                    "description": "Higher severity category in the training dataset."
                },
                4: {
                    "name": "Very High Severity",
                    "risk": "VERY HIGH",
                    "description": "Highest severity category in the training dataset."
                }
            }

            predicted_class = int(prediction)
            details = severity_details.get(
                predicted_class,
                {
                    "name": "Unknown",
                    "risk": "UNKNOWN",
                    "description": "No description available."
                }
            )

            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Severity",
                    f"Level {predicted_class}"
                )

            with col2:
                st.metric(
                    "Risk Category",
                    details["risk"]
                )

            st.info(
                f"**{details['name']}**\n\n"
                f"{details['description']}"
            )

            st.divider()

            st.subheader("📈 Class Probability Analysis")

            probability_df = pd.DataFrame({
                "Severity": [f"Level {int(c)}" for c in classes],
                "Probability (%)": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                "Probability (%)",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Severity")
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("📥 Download Prediction Report")

            report = pd.DataFrame([{
                "Latitude": latitude,
                "Longitude": longitude,
                "Date": str(accident_date),
                "Hour": accident_hour,
                "Weather": weather_condition,
                "Temperature_F": temperature,
                "Humidity": humidity,
                "Predicted_Severity": predicted_class,
                "Severity_Name": details["name"],
                "Risk_Category": details["risk"],
                "Highest_Probability_Percent": round(
                    float(np.max(probabilities) * 100), 2
                )
            }])

            csv_report = report.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download CSV Report",
                data=csv_report,
                file_name="accident_prediction_report.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.caption(
                "Note: These are model estimates based on US accident "
                "training data. They are not verified real-world "
                "accident risks or predictions for Bhopal."
            
            
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")  

            # =====================================================
# ACCIDENT ANALYTICS DASHBOARD
# =====================================================

elif page == "Accident Analytics":

    st.title("🚨 Accident Analytics Dashboard")

    st.markdown(
        "Explore accident patterns, severity distribution, "
        "weather conditions and time-based trends."
    )

    if accident_data is None or accident_data.empty:
        st.error("Accident dataset could not be loaded.")
        st.stop()

    df = accident_data.copy()

    if "severity" not in df.columns:
        st.error("Severity column is missing.")
        st.stop()

    df["severity"] = pd.to_numeric(
        df["severity"], errors="coerce"
    )
    df = df.dropna(subset=["severity"])

    # SIDEBAR FILTERS
    st.sidebar.subheader("🔎 Accident Analytics Filters")

    severity_options = sorted(
        df["severity"].unique().tolist()
    )

    selected_severity = st.sidebar.multiselect(
        "Select Severity",
        severity_options,
        default=severity_options,
        key="analytics_severity"
    )

    if not selected_severity:
        st.warning("Please select at least one severity.")
        st.stop()

    filtered_df = df[
        df["severity"].isin(selected_severity)
    ].copy()

    if "weather_condition" in filtered_df.columns:

        weather_options = sorted(
            filtered_df["weather_condition"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_weather = st.sidebar.multiselect(
            "Select Weather",
            weather_options,
            default=weather_options,
            key="analytics_weather"
        )

        filtered_df = filtered_df[
            filtered_df["weather_condition"]
            .astype(str)
            .isin(selected_weather)
        ]

    # KPI CARDS
    st.subheader("📊 Key Accident Indicators")

    total = len(filtered_df)

    average_severity = (
        filtered_df["severity"].mean() if total else 0
    )

    high_severity = filtered_df[
        filtered_df["severity"].isin([3, 4])
    ].shape[0]

    high_percentage = (
        high_severity / total * 100 if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{total:,}")
    c2.metric("Average Severity", f"{average_severity:.2f}")
    c3.metric("Severity 3 & 4", f"{high_severity:,}")
    c4.metric("Severity 3 & 4 %", f"{high_percentage:.2f}%")

    st.divider()

    # SEVERITY DISTRIBUTION
    st.subheader("1. Severity Distribution")

    severity_counts = (
        filtered_df["severity"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(severity_counts)

    st.divider()

    # WEATHER ANALYSIS
    st.subheader("2. Weather-wise Accident Records")

    if "weather_condition" in filtered_df.columns:

        weather_counts = (
            filtered_df["weather_condition"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(weather_counts)

    else:
        st.info("Weather data unavailable.")

    st.divider()

    # HOURLY ANALYSIS
    st.subheader("3. Hour-wise Accident Trends")

    if "start_time_hour" in filtered_df.columns:

        hours = pd.to_numeric(
            filtered_df["start_time_hour"],
            errors="coerce"
        )

        hourly_counts = (
            hours.dropna()
            .value_counts()
            .reindex(range(24), fill_value=0)
            .sort_index()
        )

        st.line_chart(hourly_counts)

        if hourly_counts.sum() > 0:
            peak_hour = hourly_counts.idxmax()

            st.info(
                f"Highest record count in this sample: "
                f"{peak_hour}:00 ({hourly_counts.max():,} records)."
            )

    else:
        st.info("Hourly data unavailable.")

    st.divider()

    # DAY VS NIGHT
    st.subheader("4. Day vs Night Analysis")

    if "sunrise_sunset" in filtered_df.columns:

        day_night = filtered_df[
            "sunrise_sunset"
        ].value_counts()

        st.bar_chart(day_night)

    else:
        st.info("Day/Night data unavailable.")

    st.divider()

    # SEVERITY BY WEATHER
    st.subheader("5. Severity by Weather")

    if "weather_condition" in filtered_df.columns:

        weather_severity = pd.crosstab(
            filtered_df["weather_condition"],
            filtered_df["severity"]
        )

        st.dataframe(
            weather_severity,
            use_container_width=True
        )

    st.divider()

    # WEEKDAY ANALYSIS
    st.subheader("6. Weekday Accident Analysis")

    if "start_time_weekday" in filtered_df.columns:

        weekdays = pd.to_numeric(
            filtered_df["start_time_weekday"],
            errors="coerce"
        )

        weekday_counts = (
            weekdays.dropna()
            .value_counts()
            .reindex(range(7), fill_value=0)
            .sort_index()
        )

        weekday_counts.index = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]

        st.bar_chart(weekday_counts)

    else:
        st.info("Weekday data unavailable.")

    st.divider()

    # DATA PREVIEW
    st.subheader("7. Accident Dataset Preview")

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )

    # DOWNLOAD
    st.subheader("📥 Download Analytics Data")

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Filtered Accident Data",
        data=csv_data,
        file_name="traffic_accident_analytics.csv",
        mime="text/csv"
    )

    st.caption(
        "Analysis uses a sampled US accident dataset. "
        "It does not represent verified Bhopal accident statistics "
        "or establish causes of accidents."
    ) 

# =====================================================
# ACCIDENT HOTSPOT MAP
# =====================================================

elif page == "Accident Hotspot Map":

    st.title("🗺️ Accident Hotspot Map")

    st.write(
        "Interactive geographical visualization of accident "
        "records using latitude and longitude."
    )

    st.warning(
        "This map uses a sampled US accident dataset. "
        "It is not a verified Bhopal accident map or a live traffic map."
    )

    if accident_data is None or accident_data.empty:
        st.error("Accident dataset is unavailable.")
        st.stop()

    map_df = accident_data.copy()

    required = ["start_lat", "start_lng", "severity"]

    missing = [
        col for col in required
        if col not in map_df.columns
    ]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Clean coordinates
    map_df["start_lat"] = pd.to_numeric(
        map_df["start_lat"], errors="coerce"
    )

    map_df["start_lng"] = pd.to_numeric(
        map_df["start_lng"], errors="coerce"
    )

    map_df["severity"] = pd.to_numeric(
        map_df["severity"], errors="coerce"
    )

    map_df = map_df.dropna(
        subset=["start_lat", "start_lng", "severity"]
    )

    # Keep valid geographical coordinates
    map_df = map_df[
        map_df["start_lat"].between(-90, 90)
        & map_df["start_lng"].between(-180, 180)
    ].copy()

    if map_df.empty:
        st.error("No valid geographical records found.")
        st.stop()

    # Filters
    st.sidebar.subheader("🗺️ Hotspot Map Filters")

    severity_values = sorted(
        map_df["severity"].unique().tolist()
    )

    selected_levels = st.sidebar.multiselect(
        "Select Severity Levels",
        severity_values,
        default=severity_values,
        key="hotspot_severity"
    )

    if not selected_levels:
        st.warning("Select at least one severity level.")
        st.stop()

    map_df = map_df[
        map_df["severity"].isin(selected_levels)
    ].copy()

    # Limit map records for performance
    max_points = st.select_slider(
        "Number of map records",
        options=[500, 1000, 2000, 3000, 5000],
        value=2000
    )

    if len(map_df) > max_points:
        map_df = map_df.sample(
            n=max_points,
            random_state=42
        )

    # Metrics
    st.subheader("📊 Map Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Mapped Records", f"{len(map_df):,}")

    c2.metric(
        "Severity 3 & 4",
        f"{map_df['severity'].isin([3, 4]).sum():,}"
    )

    c3.metric(
        "Severity Levels",
        f"{map_df['severity'].nunique()}"
    )

    st.divider()

    # Create map
    center_lat = map_df["start_lat"].median()
    center_lng = map_df["start_lng"].median()

    accident_map = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=5,
        tiles="OpenStreetMap"
    )

    # Cluster markers
    from folium.plugins import MarkerCluster

    marker_cluster = MarkerCluster().add_to(accident_map)

    severity_colors = {
        1: "green",
        2: "blue",
        3: "orange",
        4: "red"
    }

    for _, row in map_df.iterrows():

        level = int(row["severity"])

        color = severity_colors.get(level, "gray")

        folium.CircleMarker(
            location=[
                row["start_lat"],
                row["start_lng"]
            ],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=f"Severity Level: {level}"
        ).add_to(marker_cluster)

    # Display map
    st.subheader("📍 Interactive Accident Map")

    st.components.v1.html(
        accident_map._repr_html_(),
        height=600,
        scrolling=False
    )

    st.divider()

    # Legend
    st.subheader("🎨 Severity Legend")

    legend_data = pd.DataFrame({
        "Severity": [
            "Level 1",
            "Level 2",
            "Level 3",
            "Level 4"
        ],
        "Map Colour": [
            "Green",
            "Blue",
            "Orange",
            "Red"
        ]
    })

    st.dataframe(
        legend_data,
        use_container_width=True,
        hide_index=True
    )

    # Download
    st.subheader("📥 Download Map Data")

    csv_data = map_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Mapped Accident Records",
        data=csv_data,
        file_name="accident_hotspot_records.csv",
        mime="text/csv"
    )

    st.caption(
        "Markers show sampled accident records, not statistically "
        "verified accident hotspots. Cluster size depends on the "
        "displayed sample and map zoom."
    )   


# =====================================================
# AI TRAFFIC INTELLIGENCE & INSIGHTS
# =====================================================

elif page == "AI Traffic Intelligence":

    st.title("🚦 AI Traffic Intelligence")
    st.markdown(
        "Data-driven traffic patterns, accident analysis "
        "and traffic management recommendations."
    )

    st.info(
        "Insights are generated from available training datasets. "
        "They are not live traffic measurements or verified "
        "Bhopal accident statistics."
    )

    # ==========================================
    # TRAFFIC DATA ANALYSIS
    # ==========================================

    st.header("🚗 Traffic Congestion Intelligence")

    if traffic_data is not None and not traffic_data.empty:

        tdf = traffic_data.copy()

        if "congestion" in tdf.columns:

            congestion_counts = (
                tdf["congestion"]
                .astype(str)
                .value_counts()
            )

            total_traffic = len(tdf)

            heavy_count = int(
                tdf["congestion"].astype(str).str.upper()
                .eq("HEAVY").sum()
            )

            heavy_percentage = (
                heavy_count / total_traffic * 100
                if total_traffic else 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Traffic Records", f"{total_traffic:,}")
            c2.metric("Heavy Records", f"{heavy_count:,}")
            c3.metric("Heavy Record Share", f"{heavy_percentage:.2f}%")

            st.subheader("Congestion Distribution")
            st.bar_chart(congestion_counts)

            if "hour" in tdf.columns:

                tdf["hour"] = pd.to_numeric(
                    tdf["hour"], errors="coerce"
                )

                hourly = (
                    tdf.dropna(subset=["hour"])
                    .groupby("hour")
                    .size()
                    .sort_index()
                )

                if not hourly.empty:

                    st.subheader("Traffic Records by Hour")
                    st.line_chart(hourly)

                    busiest_hour = hourly.idxmax()

                    st.info(
                        f"Most records in this dataset occur "
                        f"at hour {int(busiest_hour)}:00."
                    )

            if "delay" in tdf.columns:

                tdf["delay"] = pd.to_numeric(
                    tdf["delay"], errors="coerce"
                )

                avg_delay = tdf["delay"].mean()

                if pd.notna(avg_delay):
                    st.metric(
                        "Average Recorded Delay",
                        f"{avg_delay:.2f}"
                    )

            if "road_id" in tdf.columns:

                st.subheader("Road-wise Traffic Records")

                road_counts = (
                    tdf["road_id"].astype(str)
                    .value_counts()
                    .head(10)
                )

                st.bar_chart(road_counts)

        else:
            st.warning("Congestion column is unavailable.")

    else:
        st.warning("Traffic dataset is unavailable.")

    st.divider()

    # ==========================================
    # ACCIDENT INTELLIGENCE
    # ==========================================

    st.header("🚨 Accident Pattern Intelligence")

    if accident_data is not None and not accident_data.empty:

        adf = accident_data.copy()

        if "severity" in adf.columns:

            adf["severity"] = pd.to_numeric(
                adf["severity"], errors="coerce"
            )

            adf = adf.dropna(subset=["severity"])

            total_accidents = len(adf)

            severe_records = adf[
                adf["severity"].isin([3, 4])
            ]

            severe_percentage = (
                len(severe_records) / total_accidents * 100
                if total_accidents else 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Accident Sample Records",
                f"{total_accidents:,}"
            )

            c2.metric(
                "Severity 3 & 4 Records",
                f"{len(severe_records):,}"
            )

            c3.metric(
                "Severity 3 & 4 Share",
                f"{severe_percentage:.2f}%"
            )

            st.subheader("Severity Distribution")

            severity_counts = (
                adf["severity"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(severity_counts)

            # Weather analysis
            if "weather_condition" in adf.columns:

                st.subheader("Weather-wise Record Distribution")

                weather_counts = (
                    adf["weather_condition"]
                    .value_counts()
                    .head(10)
                )

                st.bar_chart(weather_counts)

                if not weather_counts.empty:

                    top_weather = weather_counts.index[0]

                    st.info(
                        f"Most frequent weather category in "
                        f"this sample: {top_weather}."
                    )

            # Hour analysis
            if "start_time_hour" in adf.columns:

                st.subheader("Accident Records by Hour")

                adf["start_time_hour"] = pd.to_numeric(
                    adf["start_time_hour"],
                    errors="coerce"
                )

                hour_counts = (
                    adf.groupby("start_time_hour")
                    .size()
                    .reindex(range(24), fill_value=0)
                )

                st.line_chart(hour_counts)

                if hour_counts.sum() > 0:

                    peak_hour = hour_counts.idxmax()

                    st.info(
                        f"Highest accident record count in "
                        f"this sample: {int(peak_hour)}:00."
                    )

        else:
            st.warning("Accident severity column is unavailable.")

    else:
        st.warning("Accident dataset is unavailable.")

    st.divider()

    # ==========================================
    # AUTOMATIC INSIGHTS
    # ==========================================

    st.header("🧠 Automatic Data Insights")

    insights = []

    if (
        traffic_data is not None
        and not traffic_data.empty
        and "congestion" in traffic_data.columns
    ):

        congestion_values = (
            traffic_data["congestion"]
            .astype(str)
            .str.upper()
        )

        heavy_total = int(
            congestion_values.eq("HEAVY").sum()
        )

        if heavy_total > 0:
            insights.append(
                f"Traffic dataset contains {heavy_total} "
                "HEAVY congestion records."
            )
        else:
            insights.append(
                "No HEAVY congestion records found in this traffic sample."
            )

    if (
        accident_data is not None
        and not accident_data.empty
        and "severity" in accident_data.columns
    ):

        severity_values = pd.to_numeric(
            accident_data["severity"],
            errors="coerce"
        )

        severe_total = int(
            severity_values.isin([3, 4]).sum()
        )

        insights.append(
            f"Accident sample contains {severe_total} "
            "records in severity classes 3 and 4."
        )

    if insights:

        for i, insight in enumerate(insights, start=1):
            st.write(f"**{i}.** {insight}")

    else:
        st.info("Not enough data available to generate insights.")

    # ==========================================
    # RECOMMENDATIONS
    # ==========================================

    st.header("💡 Traffic Management Recommendations")

    recommendations = [
        "Monitor road segments with repeated heavy congestion records.",
        "Review traffic signal timing using verified local traffic studies.",
        "Improve road signage and visibility where local assessments identify a need.",
        "Use verified local accident records before identifying accident-prone locations.",
        "Collect more Bhopal-specific traffic and accident data for reliable local analysis."
    ]

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    st.divider()

    st.caption(
        "Traffic congestion model uses a small Indore training dataset. "
        "Accident analysis uses sampled US records. "
        "Neither dataset establishes verified live Bhopal conditions."
    )


# ============================================================
# BHOPAL SMART ROUTE ANALYZER
# ============================================================


elif page == "Bhopal Live Traffic":

    st.title("Bhopal Live Traffic")
    st.caption(
        "Live traffic flow from TomTom for the selected map point. "
        "This represents the nearest road segment, not the whole city."
    )

    import streamlit as st
    from traffic_api import get_tomtom_traffic

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude",
            value=23.2599,
            format="%.6f"
        )

    with col2:
        longitude = st.number_input(
            "Longitude",
            value=77.4126,
            format="%.6f"
        )

    if st.button("Fetch Live Traffic", type="primary"):
        try:
            api_key = st.secrets.get("TOMTOM_API_KEY", "")
            if not api_key:
                st.session_state["tomtom_live_result"] = {
                    "success": False,
                    "message": "TomTom API key is missing from secrets.toml."
                }
            else:
                with st.spinner("Fetching TomTom live traffic..."):
                    result = get_tomtom_traffic(
                        latitude,
                        longitude,
                        api_key
                    )
                st.session_state["tomtom_live_result"] = result
                st.session_state["tomtom_live_point"] = (
                    latitude,
                    longitude
                )
        except Exception as error:
            st.session_state["tomtom_live_result"] = {
                "success": False,
                "message": str(error)
            }

    result = st.session_state.get("tomtom_live_result")

    if result:
        if result.get("success"):
            st.success("TomTom live traffic data received.")

            current_speed = result.get("current_speed")
            free_speed = result.get("free_flow_speed")
            current_time = result.get("current_travel_time")
            free_time = result.get("free_flow_travel_time")
            confidence = result.get("confidence")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Current Speed",
                f"{current_speed} km/h" if current_speed is not None else "N/A"
            )

            m2.metric(
                "Free-flow Speed",
                f"{free_speed} km/h" if free_speed is not None else "N/A"
            )

            delay = None
            if current_time is not None and free_time is not None:
                delay = current_time - free_time

            m3.metric(
                "Travel Time Difference",
                f"{delay} sec" if delay is not None else "N/A"
            )

            m4, m5 = st.columns(2)

            m4.metric(
                "API Confidence",
                f"{confidence * 100:.1f}%"
                if confidence is not None else "N/A"
            )

            m5.metric(
                "Road Closure",
                "Yes" if result.get("road_closure") else "No"
            )

            if delay is not None:
                if delay > 0:
                    st.info(
                        f"Travel time is {delay} seconds above free-flow time."
                    )
                else:
                    st.info("No additional travel-time delay reported.")

            st.caption(
                "Coordinates: "
                f"{st.session_state.get('tomtom_live_point', (latitude, longitude))}"
            )

        else:
            st.error(result.get("message", "TomTom request failed."))

    else:
        st.info("Click Fetch Live Traffic to request current traffic data.")

    st.warning(
        "This is a single-point road-segment reading. "
        "It is not a city-wide traffic map or a verified route-wide estimate."
    )


elif page == "Bhopal Smart Route Analyzer":

    st.title("🛣️ Bhopal Smart Route Analyzer")

    st.caption(
        "Shortest route analysis using OpenStreetMap road data. "
        "Travel time is an estimate, not live traffic information."
    )

    import json
    import networkx as nx
    from pathlib import Path

    road_file = Path(BHOPAL_MAP_FILE)

    if not road_file.exists():
        st.error(f"Road GeoJSON file not found: {road_file}")

    else:
        try:
            with open(road_file, "r", encoding="utf-8") as file:
                road_geojson = json.load(file)

            graph = nx.MultiGraph()
            edge_lookup = {}

            # LOAD ROAD NETWORK
            for index, feature in enumerate(
                road_geojson.get("features", [])
            ):
                props = feature.get("properties") or {}
                geometry = feature.get("geometry") or {}

                u = props.get("u")
                v = props.get("v")
                length = props.get("length")

                try:
                    u = int(u)
                    v = int(v)
                    length = float(length)
                except (TypeError, ValueError):
                    continue

                if u == v or length <= 0:
                    continue

                geometry_type = geometry.get("type")
                coords = geometry.get("coordinates", [])

                if geometry_type == "LineString":
                    lines = [coords]

                elif geometry_type == "MultiLineString":
                    lines = coords

                else:
                    continue

                valid_lines = []

                for line in lines:
                    if not isinstance(line, list) or len(line) < 2:
                        continue

                    valid_points = []

                    for point in line:
                        if (
                            isinstance(point, (list, tuple))
                            and len(point) >= 2
                        ):
                            try:
                                lon = float(point[0])
                                lat = float(point[1])

                                if (
                                    -180 <= lon <= 180
                                    and -90 <= lat <= 90
                                ):
                                    valid_points.append([lon, lat])
                            except (TypeError, ValueError):
                                continue

                    if len(valid_points) >= 2:
                        valid_lines.append(valid_points)

                if not valid_lines:
                    continue

                road_name = props.get("name")

                if isinstance(road_name, list):
                    road_name = ", ".join(map(str, road_name))

                if not road_name:
                    road_name = f"Unnamed Road {index + 1}"

                highway = props.get("highway", "other")

                if isinstance(highway, list):
                    highway = ", ".join(map(str, highway))

                edge_id = f"edge_{index}"

                                # ROAD-TYPE-BASED ILLUSTRATIVE TRAFFIC PENALTY
                # This is an estimate, NOT live traffic data.

                highway_type = str(highway).lower()

                if "motorway" in highway_type:
                    traffic_score = 15
                elif "trunk" in highway_type:
                    traffic_score = 20
                elif "primary" in highway_type:
                    traffic_score = 30
                elif "secondary" in highway_type:
                    traffic_score = 40
                elif "tertiary" in highway_type:
                    traffic_score = 50
                elif "residential" in highway_type:
                    traffic_score = 60
                elif "living_street" in highway_type:
                    traffic_score = 65
                elif "service" in highway_type:
                    traffic_score = 70
                else:
                    traffic_score = 45
               

                graph.add_edge(
                    u,
                    v,
                    key=edge_id,
                    weight=length,
                    edge_id=edge_id,
                    traffic_score=traffic_score
                )


                edge_lookup[edge_id] = {
                    "u": u,
                    "v": v,
                    "length": length,
                    "name": str(road_name),
                    "highway": str(highway),
                    "lines": valid_lines
                }

            if graph.number_of_edges() == 0:
                st.error("No usable road connections found.")

            else:
                st.success(
                    f"Loaded {graph.number_of_edges():,} road connections "
                    f"and {graph.number_of_nodes():,} junctions."
                )

                
                # CREATE EASY-TO-UNDERSTAND JUNCTION NAMES
                node_options = sorted(graph.nodes())

                node_names = {}

                for node in node_options:
                    nearby_roads = []

                    for neighbour in graph.neighbors(node):
                        edge_data = graph.get_edge_data(node, neighbour)

                        if edge_data:
                            for edge in edge_data.values():
                                road_name = edge_lookup[
                                    edge["edge_id"]
                                ]["name"]

                                if road_name not in nearby_roads:
                                    nearby_roads.append(road_name)

                    if nearby_roads:
                        display_name = " / ".join(nearby_roads[:2])
                    else:
                        display_name = "Unnamed Junction"

                    node_names[node] = (
                        f"{display_name} (Junction {node})"
                    )

                col1, col2 = st.columns(2)

                with col1:
                    start_node = st.selectbox(
                        "📍 Starting Road / Junction",
                        node_options,
                        format_func=lambda x: node_names[x],
                        key="route_start_node"
                    )

                with col2:
                    end_options = [
                        node for node in node_options
                        if node != start_node
                    ]

                    end_node = st.selectbox(
                        "🏁 Destination Road / Junction",
                        end_options,
                        format_func=lambda x: node_names[x],
                        key="route_end_node"
                    )


                speed_factor = st.slider(
                    "Illustrative Average Speed (km/h)",
                    min_value=10,
                    max_value=60,
                    value=30,
                    step=5,
                    key="route_speed"
                )

                traffic_level = st.selectbox(
                    "🚦 Select Traffic Condition",
                    ["Low Traffic", "Moderate Traffic", "High Traffic"],
                    key="route_traffic_level"
                )

                traffic_multipliers = {
                    "Low Traffic": 1.0,
                    "Moderate Traffic": 1.5,
                    "High Traffic": 2.5
                }

                traffic_multiplier = traffic_multipliers[traffic_level]
                
                if st.button(
                    "🚗 Find Shortest Route",
                    type="primary",
                    key="find_bhopal_route"
                ):
                    try:

                        
                        # ==========================================
                        # SHORTEST DISTANCE + TRAFFIC AWARE ROUTING
                        # ==========================================

                        distance_graph = nx.Graph()
                        traffic_graph = nx.Graph()

                        for u, v, data in graph.edges(data=True):

                            road_length = data["weight"]
                            traffic_score = data.get("traffic_score", 20)

                            traffic_cost = road_length * (
                                1 + (traffic_score / 100) * (traffic_multiplier - 1)
                            )

                            # Shortest distance graph
                            if (
                                not distance_graph.has_edge(u, v)
                                or road_length < distance_graph[u][v]["weight"]
                            ):
                                distance_graph.add_edge(
                                    u, v, weight=road_length
                                )

                            # Fastest traffic-aware graph
                            if (
                                not traffic_graph.has_edge(u, v)
                                or traffic_cost < traffic_graph[u][v]["weight"]
                            ):
                                traffic_graph.add_edge(
                                    u, v, weight=traffic_cost
                                )

                        # Find shortest distance path
                        shortest_path = nx.shortest_path(
                            distance_graph,
                            source=start_node,
                            target=end_node,
                            weight="weight"
                        )

                        # Find traffic-aware paths
                        route_generator = nx.shortest_simple_paths(
                            traffic_graph,
                            source=start_node,
                            target=end_node,
                            weight="weight"
                        )

                        candidate_paths = []

                        for path in route_generator:
                            candidate_paths.append(path)

                            if len(candidate_paths) >= 10:
                                break

                        # Fastest route
                        fastest_path = candidate_paths[0]

                        # Build route details
                        def build_route(path, route_mode):

                            path_edges = []
                            path_distance = 0.0
                            total_traffic_cost = 0.0

                            for i in range(len(path) - 1):

                                u = path[i]
                                v = path[i + 1]

                                parallel_edges = graph.get_edge_data(u, v)

                                if route_mode == "distance":

                                    selected_edge = min(
                                        parallel_edges.values(),
                                        key=lambda item: item["weight"]
                                    )

                                else:

                                    selected_edge = min(
                                        parallel_edges.values(),
                                        key=lambda item: item["weight"] * (
                                            1 + (
                                                item.get("traffic_score", 20) / 100
                                            ) * (traffic_multiplier - 1)
                                        )
                                    )

                                edge_info = edge_lookup[
                                    selected_edge["edge_id"]
                                ]

                                path_edges.append(edge_info)

                                path_distance += selected_edge["weight"]

                                total_traffic_cost += selected_edge["weight"] * (
                                    1 + (
                                        selected_edge.get("traffic_score", 20) / 100
                                    ) * (traffic_multiplier - 1)
                                )

                            return {
                                "nodes": path,
                                "edges": path_edges,
                                "distance": path_distance,
                                "traffic_cost": total_traffic_cost
                            }

                        # Calculate shortest and fastest routes
                        shortest_distance_route = build_route(
                            shortest_path, "distance"
                        )

                        fastest_route = build_route(
                            fastest_path, "traffic"
                        )

                        # Find a different alternative route
                        alternative_path = None

                        for path in candidate_paths:

                            if (
                                path != shortest_path
                                and path != fastest_path
                            ):
                                alternative_path = path
                                break

                        # Save routes in display order
                        alternative_routes = [
                            shortest_distance_route,
                            fastest_route
                        ]

                        if alternative_path is not None:

                            alternative_route = build_route(
                                alternative_path, "traffic"
                            )

                            alternative_routes.append(alternative_route)

                        shortest = shortest_distance_route

                        st.session_state["bhopal_route_result"] = {
                            "nodes": shortest["nodes"],
                            "edges": shortest["edges"],
                            "distance": shortest["distance"],
                            "speed": speed_factor,
                            "start": start_node,
                            "end": end_node,
                            "traffic_level": traffic_level,
                            "traffic_multiplier": traffic_multiplier,
                            "alternatives": alternative_routes
                        }



                    except nx.NetworkXNoPath:
                        st.session_state.pop(
                            "bhopal_route_result", None
                        )
                        st.error(
                            "No connected route exists between these junctions."
                        )

                    except Exception as e:
                        st.session_state.pop(
                            "bhopal_route_result", None
                        )
                        st.error(f"Route calculation error: {e}")

                result = st.session_state.get(
                    "bhopal_route_result"
                )

                if result is not None:

                    if (
                        result["start"] != start_node
                        or result["end"] != end_node
                        or result["speed"] != speed_factor
                        or result.get("traffic_level") != traffic_level
                    ):
                        st.info(
                            "Inputs changed. Click Find Shortest Route "
                            "to calculate the updated route."
                        )

                    else:
                        route_nodes = result["nodes"]
                        route_edges = result["edges"]

                        distance_km = result["distance"] / 1000

                        normal_travel_minutes = (
                            distance_km / result["speed"]
                        ) * 60

                        traffic_factor = (
                            result["alternatives"][0]["traffic_cost"]
                            / result["alternatives"][0]["distance"]
                            if result["alternatives"][0]["distance"] > 0
                            else 1.0
                        )

                        travel_minutes = (
                            normal_travel_minutes * traffic_factor
                        )

                        st.divider()
                        st.subheader("📊 Route Results")

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Route Distance",
                            f"{distance_km:.2f} km"
                        )

                        c2.metric(
                            "Estimated Travel Time",
                            f"{travel_minutes:.1f} min"
                        )

                        c3.metric(
                            "Road Segments",
                            len(route_edges)
                        )

                        traffic_delay = max(
                            0,
                            travel_minutes - normal_travel_minutes
                        )

                        st.metric(
                            "🚦 Estimated Extra Traffic Delay",
                            f"+{traffic_delay:.1f} min"
                        )

                        st.info(
                            f"🚦 Selected Traffic Condition: {result.get('traffic_level', 'Demo Traffic')}"
                        )
                        
                        # ALTERNATIVE ROUTE COMPARISON
                        st.divider()
                        st.subheader("🔀 Alternative Route Comparison")

                        comparison_rows = []
                        route_options = []

                        for index, route in enumerate(
                            result.get("alternatives", [])
                        ):
                            route_distance = route["distance"] / 1000

                            route_distance_km = route["distance"] / 1000

                            route_normal_time = (
                                route_distance_km / result["speed"]
                            ) * 60

                            route_traffic_factor = (
                                route["traffic_cost"] / route["distance"]
                                if route["distance"] > 0
                                else 1.0
                            )

                            route_time = (
                                route_normal_time * route_traffic_factor
                            )

                            if index == 0:
                                route_name = "🔵 Shortest Distance Route"

                            elif index == 1:
                                route_name = "🟢 Fastest Traffic-Aware Route"

                            else:
                                route_name = f"🟠 Alternative Route {index - 1}"

                            route_options.append(route_name)

                            comparison_rows.append({
                                "Route": route_name,
                                "Distance (km)": round(route_distance, 2),
                                "Estimated Time (min)": round(route_time, 1),
                                "Road Segments": len(route["edges"])
                            })

                        comparison_df = pd.DataFrame(comparison_rows)

                        st.dataframe(
                            comparison_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # ROUTE SELECTION
                        st.divider()
                        st.subheader("🎯 Select Your Preferred Route")

                        selected_route_name = st.selectbox(
                            "Choose a route",
                            route_options,
                            key="selected_bhopal_route"
                        )

                        selected_index = route_options.index(
                            selected_route_name
                        )

                        selected_route = result["alternatives"][
                            selected_index
                        ]

                        selected_distance = (
                            selected_route["distance"] / 1000
                        )

                        selected_normal_time = (
                            selected_distance / result["speed"]
                        ) * 60

                        selected_traffic_factor = (
                            selected_route["traffic_cost"] / selected_route["distance"]
                            if selected_route["distance"] > 0
                            else 1.0
                        )

                        selected_time = (
                            selected_normal_time * selected_traffic_factor
                        )

                        r1, r2, r3 = st.columns(3)

                        r1.metric(
                            "Selected Route",
                            selected_route_name
                        )

                        r2.metric(
                            "Distance",
                            f"{selected_distance:.2f} km"
                        )

                        r3.metric(
                            "Estimated Time",
                            f"{selected_time:.1f} min"
                        )

                        # SIMULATED TRAFFIC IMPACT INDICATOR
                        if selected_route["distance"] > 0:

                            traffic_impact_factor = (
                                selected_route["traffic_cost"]
                                / selected_route["distance"]
                            )

                            if traffic_impact_factor < 1.20:
                                impact_level = "LOW"
                                impact_message = "Lower simulated traffic impact"

                            elif traffic_impact_factor < 1.50:
                                impact_level = "MODERATE"
                                impact_message = "Moderate simulated traffic impact"

                            else:
                                impact_level = "HIGH"
                                impact_message = "Higher simulated traffic impact"

                            st.divider()
                            st.subheader("🚦 Route Traffic Impact")

                            impact_col1, impact_col2 = st.columns(2)

                            impact_col1.metric(
                                "Traffic Impact Level",
                                impact_level
                            )

                            impact_col2.metric(
                                "Traffic Impact Factor",
                                f"{traffic_impact_factor:.2f}"
                            )

                            st.info(impact_message)

                            st.caption(
                                "This indicator uses simulated traffic scores "
                                "for demonstration. It is not a live traffic "
                                "or road safety assessment."
                            )


                        # TOMTOM LIVE TRAFFIC - SELECTED ROUTE SAMPLING
                        st.divider()
                        st.subheader("Live Traffic on Selected Route")
                        st.caption(
                            "Maximum 5 road segments are sampled near their "
                            "midpoints using TomTom. This does not represent "
                            "complete road coverage."
                        )

                        if st.session_state.get("live_route_name") != selected_route_name:
                            st.session_state["live_route_name"] = selected_route_name
                            st.session_state["live_route_results"] = {}

                        if "live_route_results" not in st.session_state:
                            st.session_state["live_route_results"] = {}

                        if st.button(
                            "Fetch Live Traffic for Selected Route",
                            key="fetch_selected_route_live_traffic"
                        ):
                            live_results = {}
                            try:
                                from traffic_api import get_tomtom_traffic
                                tomtom_key = st.secrets.get("TOMTOM_API_KEY", "")
                            except Exception as error:
                                tomtom_key = ""
                                st.warning(
                                    f"TomTom configuration could not be read: {error}"
                                )

                            if not tomtom_key:
                                st.warning(
                                    "TomTom API key not found. Check "
                                    ".streamlit/secrets.toml."
                                )
                            else:
                                for segment_index, edge in enumerate(
                                    selected_route["edges"][:5]
                                ):
                                    try:
                                        lines = edge.get("lines", [])
                                        if not lines or not lines[0]:
                                            live_results[segment_index] = {
                                                "success": False,
                                                "message": "Road geometry unavailable"
                                            }
                                            continue

                                        polyline = lines[0]
                                        midpoint = polyline[len(polyline) // 2]

                                        if len(midpoint) < 2:
                                            raise ValueError("Invalid midpoint")

                                        longitude = float(midpoint[0])
                                        latitude = float(midpoint[1])

                                        traffic_data = get_tomtom_traffic(
                                            latitude,
                                            longitude,
                                            tomtom_key
                                        )
                                        live_results[segment_index] = traffic_data

                                    except Exception as error:
                                        live_results[segment_index] = {
                                            "success": False,
                                            "message": str(error)
                                        }

                                st.session_state["live_route_results"] = live_results

                                successful_samples = sum(
                                    1 for item in live_results.values()
                                    if item.get("success")
                                )
                                st.success(
                                    f"Live traffic request completed. "
                                    f"Successful samples: {successful_samples}/"
                                    f"{len(live_results)}"
                                )

                        live_route_results = st.session_state.get(
                            "live_route_results", {}
                        )

                        live_road_rows = []

                        for segment_index, edge in enumerate(
                            selected_route["edges"]
                        ):
                            segment_distance = edge["length"] / 1000
                            fallback_time = (
                                segment_distance / result["speed"] * 60
                                if result.get("speed", 0) > 0
                                else None
                            )

                            live_data = live_route_results.get(segment_index, {})
                            current_speed = live_data.get("current_speed")
                            free_flow_speed = live_data.get("free_flow_speed")

                            if (
                                live_data.get("success")
                                and current_speed is not None
                                and float(current_speed) > 0
                            ):
                                live_time = (
                                    segment_distance / float(current_speed) * 60
                                )
                                status = "Live sample"
                            else:
                                live_time = fallback_time
                                status = (
                                    "Not sampled"
                                    if segment_index >= 5
                                    else live_data.get("message", "Not fetched")
                                )

                            live_road_rows.append({
                                "Road": edge.get("name") or "Unnamed Road",
                                "Distance (km)": round(segment_distance, 3),
                                "Current Speed (km/h)": (
                                    round(float(current_speed), 1)
                                    if current_speed is not None
                                    else "N/A"
                                ),
                                "Free-flow Speed (km/h)": (
                                    round(float(free_flow_speed), 1)
                                    if free_flow_speed is not None
                                    else "N/A"
                                ),
                                "Estimated Segment Time (min)": (
                                    round(live_time, 2)
                                    if live_time is not None
                                    else "N/A"
                                ),
                                "Data Status": status
                            })

                        st.dataframe(
                            pd.DataFrame(live_road_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.caption(
                            "Live segment time uses distance/current speed. "
                            "When live speed is unavailable, the existing "
                            "estimated time is shown instead. Route ranking "
                            "still uses the existing route calculation."
                        )


                        st.write("**Roads included in selected route:**")

                        st.caption(
                            "Live readings are requested for up to five "
                            "road-segment midpoints. Other segments retain "
                            "the existing estimated travel time."
                        )

                        route_live_key = (
                            f"{start_node}-{end_node}-{selected_route_name}"
                        )

                        if st.button(
                            "Fetch Live Traffic for Route Segments",
                            key="fetch_route_segment_live"
                        ):
                            try:
                                from traffic_api import get_tomtom_traffic

                                api_key = st.secrets.get(
                                    "TOMTOM_API_KEY", ""
                                )

                                live_results = {}

                                if not api_key:
                                    st.warning(
                                        "TomTom API key is missing from secrets."
                                    )
                                else:
                                    with st.spinner(
                                        "Checking up to five route segments..."
                                    ):
                                        for segment_number, edge in enumerate(
                                            selected_route["edges"][:5],
                                            start=1
                                        ):
                                            lines = edge.get("lines", [])

                                            if not lines or not lines[0]:
                                                continue

                                            coordinates = lines[0]
                                            midpoint = coordinates[
                                                len(coordinates) // 2
                                            ]

                                            longitude = float(midpoint[0])
                                            latitude = float(midpoint[1])

                                            response = get_tomtom_traffic(
                                                latitude,
                                                longitude,
                                                api_key
                                            )

                                            if response.get("success"):
                                                live_results[segment_number] = {
                                                    "current_speed": response.get(
                                                        "current_speed"
                                                    ),
                                                    "free_flow_speed": response.get(
                                                        "free_flow_speed"
                                                    ),
                                                    "confidence": response.get(
                                                        "confidence"
                                                    ),
                                                }

                                st.session_state[
                                    "bhopal_route_live_results"
                                ] = live_results

                                st.session_state[
                                    "bhopal_route_live_key"
                                ] = route_live_key

                                if live_results:
                                    st.success(
                                        f"Live readings received for "
                                        f"{len(live_results)} route segment(s)."
                                    )
                                else:
                                    st.warning(
                                        "No live segment readings received. "
                                        "Estimated times will be used."
                                    )

                            except Exception as error:
                                st.error(
                                    f"Live route traffic error: {error}"
                                )

                        live_results = st.session_state.get(
                            "bhopal_route_live_results", {}
                        )

                        if st.session_state.get(
                            "bhopal_route_live_key"
                        ) != route_live_key:
                            live_results = {}

                        road_detail_rows = []
                        cumulative_distance = 0.0
                        cumulative_time = 0.0
                        live_segments_used = 0

                        for segment_number, edge in enumerate(
                            selected_route["edges"], start=1
                        ):
                            segment_distance = edge["length"] / 1000
                            cumulative_distance += segment_distance

                            estimated_time = (
                                segment_distance / result["speed"]
                            ) * 60

                            segment_data = live_results.get(
                                segment_number, {}
                            )

                            current_speed = segment_data.get(
                                "current_speed"
                            )
                            free_flow_speed = segment_data.get(
                                "free_flow_speed"
                            )
                            confidence = segment_data.get("confidence")

                            if (
                                current_speed is not None
                                and current_speed > 0
                            ):
                                segment_time = (
                                    segment_distance / current_speed
                                ) * 60
                                traffic_source = "TomTom live speed"
                                live_segments_used += 1
                            else:
                                segment_time = estimated_time
                                traffic_source = "Existing estimate"

                            cumulative_time += segment_time

                            road_detail_rows.append({
                                "Segment": segment_number,
                                "Road Name": edge["name"] or "Unnamed Road",
                                "Road Type": edge["highway"],
                                "Distance (km)": round(
                                    segment_distance, 3
                                ),
                                "Current Speed (km/h)": (
                                    current_speed
                                    if current_speed is not None else "N/A"
                                ),
                                "Free-flow Speed (km/h)": (
                                    free_flow_speed
                                    if free_flow_speed is not None else "N/A"
                                ),
                                "API Confidence": (
                                    round(confidence * 100, 1)
                                    if confidence is not None else "N/A"
                                ),
                                "Travel Time (min)": round(
                                    segment_time, 2
                                ),
                                "Cumulative Time (min)": round(
                                    cumulative_time, 2
                                ),
                                "Time Source": traffic_source
                            })

                        if live_segments_used:
                            st.success(
                                f"{live_segments_used} segment(s) use "
                                "TomTom live speed. Remaining segments "
                                "use existing estimates."
                            )
                        else:
                            st.info(
                                "No live readings currently applied. "
                                "All segment times use existing estimates."
                            )

                        selected_road_table = pd.DataFrame(
                            road_detail_rows
                        )
                        
                        st.dataframe(
                            selected_road_table,
                            use_container_width=True,
                            hide_index=True
                        )

                        # SMART ROUTE RECOMMENDATION
                        st.divider()
                        st.subheader("🤖 Smart Route Recommendation")

                        alternatives = result.get("alternatives", [])

                        if alternatives:

                            shortest_index = min(
                                range(len(alternatives)),
                                key=lambda i: alternatives[i]["distance"]
                            )

                            fastest_index = min(
                                range(len(alternatives)),
                                key=lambda i: alternatives[i]["traffic_cost"]

                                )
                            

                            shortest_route = alternatives[shortest_index]
                            fastest_route = alternatives[fastest_index]

                            shortest_km = shortest_route["distance"] / 1000
                            fastest_km = fastest_route["distance"] / 1000

                            fastest_route = alternatives[fastest_index]

                            fastest_route_distance_km = (
                                fastest_route["distance"] / 1000
                            )

                            fastest_normal_minutes = (
                                fastest_route_distance_km / result["speed"]
                            ) * 60

                            fastest_traffic_factor = (
                                fastest_route["traffic_cost"]
                                / fastest_route["distance"]
                                if fastest_route["distance"] > 0
                                else 1.0
                            )

                            fastest_minutes = (
                                fastest_normal_minutes
                                * fastest_traffic_factor
                            )
                            
                            st.success(
                                f"Recommended by shortest distance: "
                                f"{route_options[shortest_index]}"
                            )

                            st.info(
                                f"Fastest estimated route: "
                                f"{route_options[fastest_index]} "
                                f"({fastest_minutes:.1f} minutes)"
                            )

                            if shortest_index == fastest_index:
                                st.write(
                                    "Both recommendations point to the "
                                    "same route based on the selected "
                                    "average speed."
                                )
                            else:
                                st.write(
                                    "The shortest-distance and "
                                    "fastest-estimated routes are different."
                                )

                            st.caption(
                                "Shortest route is ranked by road distance. "
                                "Fastest traffic-aware route is ranked by "
                                "simulated traffic cost and selected average "
                                "speed. No live traffic data is used."
                            )



                        
                        # BUILD ALL ROUTE COORDINATES
                        st.divider()
                        st.subheader("🗺️ Route Comparison Map")

                        route_colors = ["blue", "green", "orange"]

                        route_labels = [
                            "🔵 Shortest Distance Route",
                            "🟢 Fastest Traffic-Aware Route",
                            "🟠 Alternative Route"
                        ]

                        routes_to_show = result.get(
                            "alternatives",
                            [{
                                "nodes": route_nodes,
                                "edges": route_edges,
                                "distance": result["distance"]
                            }]
                        )

                        route_map = folium.Map(
                            location=[23.2599, 77.4126],
                            zoom_start=12,
                            tiles="OpenStreetMap",
                            control_scale=True
                        )

                        all_map_coordinates = []
                        route_coordinates_list = []

                        for route_index, route in enumerate(routes_to_show):
                            selected_junctions = []
                            current_coordinates = []
                            current_nodes = route["nodes"]

                            for i, edge in enumerate(route["edges"]):

                                lines = edge["lines"]

                                if edge["u"] == current_nodes[i]:
                                    ordered_lines = lines
                                else:
                                    ordered_lines = [
                                        list(reversed(line))
                                        for line in reversed(lines)
                                    ]

                            if route_index == selected_index and ordered_lines:
                                if ordered_lines[-1]:
                                    end_point = ordered_lines[-1][-1]

                                    junction_coordinate = (
                                        float(end_point[1]),
                                        float(end_point[0])
                                    )

                                    if i < len(route["edges"]) - 1:
                                        selected_junctions.append(
                                            junction_coordinate
                                        )    

                                for line in ordered_lines:
                                    for point in line:

                                        coordinate = (
                                            float(point[1]),
                                            float(point[0])
                                        )

                                        if (
                                            not current_coordinates
                                            or current_coordinates[-1] != coordinate
                                        ):
                                            current_coordinates.append(
                                                coordinate
                                            )

                            route_coordinates_list.append(
                                current_coordinates
                            )

                            if len(current_coordinates) >= 2:

                                color = route_colors[
                                    min(route_index, 2)
                                ]

                                label = route_labels[
                                    min(route_index, 2)
                                ]

                                is_selected = (
                                    route_index == selected_index
                                )

                                folium.PolyLine(
                                    current_coordinates,
                                    color="red" if is_selected else color,
                                    weight=8 if is_selected else 3,
                                    opacity=1.0 if is_selected else 0.35,
                                    tooltip=(
                                        "SELECTED ROUTE - " + label
                                        if is_selected else label
                                    )
                                ).add_to(route_map)

                                all_map_coordinates.extend(
                                    current_coordinates
                                )

                        if (
                            len(route_coordinates_list) > selected_index
                            and len(route_coordinates_list[selected_index]) >= 2
                        ):

                            folium.Marker(
                                route_coordinates_list[selected_index][0],
                                tooltip="Starting Junction",
                                icon=folium.Icon(color="green")
                            ).add_to(route_map)

                            folium.Marker(
                                route_coordinates_list[selected_index][-1],
                                tooltip="Destination Junction",
                                icon=folium.Icon(color="red")
                            ).add_to(route_map)

                                                # SELECTED ROUTE INTERMEDIATE JUNCTIONS

                        for junction_number, junction_coordinate in enumerate(
                            selected_junctions, start=1
                        ):
                            folium.CircleMarker(
                                location=junction_coordinate,
                                radius=5,
                                color="purple",
                                fill=True,
                                fill_color="purple",
                                fill_opacity=1,
                                tooltip=f"Intermediate Junction {junction_number}"
                            ).add_to(route_map)
                        
                        if all_map_coordinates:

                            route_map.fit_bounds(all_map_coordinates)

                            st.components.v1.html(
                                route_map._repr_html_(),
                                height=600,
                                scrolling=False
                            )

                        else:
                            st.warning(
                                "Route coordinates could not be displayed."
                            )

                            st.markdown(
                            """
                            **🗺️ Route Map Legend**

                            🔴 **Red Thick Line:** Selected Route

                            🔵 **Blue Line:** Shortest Route

                            🟢 **Green Line:** Fastest Route

                            🟠 **Orange Line:** Alternative Route

                            🟣 **Purple Circles:** Intermediate Junctions

                            🟢 **Green Marker:** Starting Junction

                            🔴 **Red Marker:** Destination Junction
                            """
                        )

                        st.divider()
                        st.subheader("🛣️ Route Road Details")

                                               # PROFESSIONAL ROAD-WISE ROUTE DETAILS

                        road_detail_rows = []
                        cumulative_distance = 0.0
                        cumulative_time = 0.0

                        for segment_number, edge in enumerate(
                            selected_route["edges"], start=1
                        ):
                            segment_distance = edge["length"] / 1000

                            segment_time = (
                                segment_distance / result["speed"]
                            ) * 60

                            cumulative_distance += segment_distance
                            cumulative_time += segment_time

                            road_detail_rows.append({
                                "Segment": segment_number,
                                "Road": edge["name"] or "Unnamed Road",
                                "Road Type": edge["highway"],
                                "Distance (km)": round(
                                    segment_distance, 3
                                ),
                                "Cumulative Distance (km)": round(
                                    cumulative_distance, 3
                                ),
                                "Estimated Base Time (min)": round(
                                    segment_time, 2
                                ),
                                "Cumulative Base Time (min)": round(
                                    cumulative_time, 2
                                )
                            })

                        route_table = pd.DataFrame(road_detail_rows)

                        st.dataframe(
                            route_table,
                            use_container_width=True,
                            hide_index=True
                        )

                        # PROFESSIONAL ROUTE REPORT

                        report_header = f"""
                        TRAFFICSENSE AI - ROUTE REPORT

                        Selected Route: {selected_route_name}
                        Starting Junction: {start_node}
                        Destination Junction: {end_node}
                        Traffic Condition: {traffic_level}

                        Total Distance: {selected_distance:.2f} km
                        Estimated Travel Time: {selected_time:.1f} minutes

                        Note: Traffic scores are simulated for demonstration.
                        This report does not use live traffic data.
                        """

                        report_data = (
                           report_header
                           + "\n\nROAD-WISE DETAILS\n"
                           + route_table.to_csv(index=False)
                       )

                        st.download_button(
                            label="⬇️ Download Professional Route Report",
                            data=report_data.encode("utf-8"),
                            file_name="TrafficSense_AI_Route_Report.csv",
                            mime="text/csv",
                            key="download_bhopal_route"
                        )

                        st.caption(
                            "Route distance is calculated from the road "
                            "network. Travel time uses simulated traffic "
                            "scores and selected average speed; it is not live."
                        )

        except Exception as e:
            st.error(f"Road network loading error: {e}")


# =====================================================
# 18. BHOPAL ROAD NETWORK
# =====================================================

elif page == "Bhopal Dataset Quality Monitor":

    import os
    import pandas as pd

    st.title("?? Bhopal Dataset Quality Monitor")
    st.caption("TrafficSense AI | Dataset Quality & ML Readiness")

    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "bhopal_traffic_observations.csv"
    )

    if not os.path.exists(dataset_path):
        st.warning("Bhopal traffic dataset has not been created yet.")
    else:
        try:
            df = pd.read_csv(dataset_path)

            required = [
                "record_id", "timestamp", "road_name",
                "congestion", "source", "notes"
            ]

            for column in required:
                if column not in df.columns:
                    df[column] = ""

            df = df[required]
            df["timestamp_parsed"] = pd.to_datetime(
                df["timestamp"], errors="coerce"
            )

            total = len(df)
            roads = df["road_name"].nunique()
            classes = df["congestion"].nunique()

            st.subheader("Dataset Overview")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Observations", total)
            c2.metric("Unique Roads", roads)
            c3.metric("Traffic Classes", classes)

            st.divider()
            st.subheader("Traffic Class Distribution")

            if total:
                st.bar_chart(df["congestion"].value_counts())
            else:
                st.info("No observations available.")

            st.subheader("Road-wise Distribution")

            if total:
                st.bar_chart(df["road_name"].value_counts())

            st.subheader("Time-wise Analysis")

            valid_times = df.dropna(subset=["timestamp_parsed"]).copy()

            if len(valid_times):
                hours = valid_times["timestamp_parsed"].dt.hour

                valid_times["Time Period"] = hours.apply(
                    lambda h:
                        "Morning (5-11)" if 5 <= h < 11
                        else "Afternoon (11-17)" if 11 <= h < 17
                        else "Evening (17-22)" if 17 <= h < 22
                        else "Night (22-5)"
                )

                st.bar_chart(
                    valid_times["Time Period"].value_counts()
                )
            else:
                st.info("Valid timestamps are not available yet.")

            st.divider()
            st.subheader("Data Quality Checks")

            missing = df[required].isna().sum().sum()

            duplicate_count = df.duplicated(
                subset=[
                    "timestamp", "road_name",
                    "congestion", "source", "notes"
                ]
            ).sum()

            invalid_timestamps = df["timestamp_parsed"].isna().sum()

            q1, q2, q3 = st.columns(3)
            q1.metric("Missing Cells", int(missing))
            q2.metric("Duplicate Observations", int(duplicate_count))
            q3.metric("Invalid Timestamps", int(invalid_timestamps))

            st.divider()
            st.subheader("ML Training Readiness")

            class_counts = df["congestion"].value_counts()
            expected_classes = ["FREE", "LIGHT", "MODERATE", "HEAVY"]

            checks = {
                "At least 100 observations": total >= 100,
                "At least 5 unique roads": roads >= 5,
                "All 4 traffic classes present": all(
                    class_counts.get(label, 0) > 0
                    for label in expected_classes
                ),
                "At least 10 records per class": all(
                    class_counts.get(label, 0) >= 10
                    for label in expected_classes
                ),
                "At least 95% valid timestamps": (
                    total > 0 and
                    (total - invalid_timestamps) / total >= 0.95
                ),
                "No duplicate observations": duplicate_count == 0
            }

            for label, passed in checks.items():
                if passed:
                    st.success("PASS ? " + label)
                else:
                    st.warning("NOT READY ? " + label)

            ready = all(checks.values())

            if ready:
                st.success(
                    "Initial dataset checks passed. "
                    "Next: perform leakage-safe model validation."
                )
            else:
                st.info(
                    "Dataset is still being collected. "
                    "Do not train or deploy a Bhopal model yet."
                )

            st.caption(
                "These are initial project quality thresholds, "
                "not a guarantee of model accuracy or sufficient "
                "statistical representativeness."
            )

            st.divider()
            st.subheader("Observation Records")

            st.dataframe(
                df[required],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Bhopal Dataset CSV",
                data=df[required].to_csv(index=False).encode("utf-8-sig"),
                file_name="bhopal_traffic_observations.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Dataset monitor error: {error}")



elif page == "Bhopal Traffic Data Collector":

    import os
    import uuid
    import pandas as pd
    from datetime import datetime, date

    st.title("Bhopal Traffic Data Collector")
    st.caption("TrafficSense AI | Verified Road Observation System")

    st.info(
        "Sirf genuine traffic observations ya verifiable sources ka data "
        "record karein. Ye form automatically live traffic detect nahi karta."
    )

    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
    os.makedirs(data_dir, exist_ok=True)

    data_file = os.path.join(
        data_dir, "bhopal_traffic_observations.csv"
    )

    columns = [
        "record_id", "timestamp", "road_name",
        "congestion", "source", "notes"
    ]
    classes = ["FREE", "LIGHT", "MODERATE", "HEAVY"]

    def load_bhopal_observations():
        if not os.path.exists(data_file):
            return pd.DataFrame(columns=columns)

        data = pd.read_csv(data_file)
        for column in columns:
            if column not in data.columns:
                data[column] = ""
        return data[columns]

    try:
        observations = load_bhopal_observations()
    except Exception as error:
        st.error(f"Dataset loading error: {error}")
        st.stop()

    total = len(observations)
    unique_roads = (
        observations["road_name"].fillna("").astype(str).str.strip().nunique()
    )
    class_counts = observations["congestion"].value_counts()
    represented_classes = sum(class_counts.get(c, 0) > 0 for c in classes)

    st.subheader("Dataset Collection Progress")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Observations", total, "Target: 100")
    c2.metric("Unique Roads", unique_roads, "Target: 5")
    c3.metric("Traffic Classes", represented_classes, "Target: 4")

    st.progress(min(total / 100, 1.0), text=f"{total}/100 observations collected")

    st.markdown("**Class-wise collection (target: at least 10 each)**")
    progress_columns = st.columns(4)

    for column, traffic_class in zip(progress_columns, classes):
        count = int(class_counts.get(traffic_class, 0))
        column.metric(traffic_class, f"{count}/10")
        column.progress(min(count / 10, 1.0))

    st.divider()
    st.subheader("Add New Traffic Observation")

    with st.form("bhopal_observation_form_v2"):
        road = st.text_input(
            "Road / Junction Name",
            placeholder="Example: MP Nagar Zone 1"
        )

        col1, col2 = st.columns(2)

        with col1:
            obs_date = st.date_input(
                "Observation Date", value=date.today()
            )

        with col2:
            obs_time = st.time_input(
                "Observation Time",
                value=datetime.now().time().replace(
                    second=0, microsecond=0
                )
            )

        congestion = st.selectbox(
            "Observed Traffic Condition", classes
        )

        source = st.text_input(
            "Observation Source",
            placeholder="Personal observation / verifiable source"
        )

        notes = st.text_area("Additional Notes (Optional)")

        submitted = st.form_submit_button(
            "Save Observation", type="primary"
        )

        if submitted:
            clean_road = " ".join(road.strip().split())
            clean_source = source.strip()
            observation_time = datetime.combine(obs_date, obs_time)

            if not clean_road:
                st.error("Road name is required.")

            elif not clean_source:
                st.error("Observation source is required.")

            elif observation_time > datetime.now():
                st.error("Future date/time observations cannot be saved.")

            else:
                existing_times = pd.to_datetime(
                    observations["timestamp"], errors="coerce"
                )

                normalized_roads = (
                    observations["road_name"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.casefold()
                )

                duplicate_mask = (
                    normalized_roads.eq(clean_road.casefold())
                    & existing_times.eq(observation_time)
                    & observations["congestion"].astype(str).eq(congestion)
                )

                if duplicate_mask.any():
                    st.error(
                        "Duplicate observation detected: same road, "
                        "time and traffic category already exist."
                    )
                else:
                    record = {
                        "record_id": str(uuid.uuid4()),
                        "timestamp": observation_time.isoformat(),
                        "road_name": clean_road,
                        "congestion": congestion,
                        "source": clean_source,
                        "notes": notes.strip()
                    }

                    updated = pd.concat(
                        [observations, pd.DataFrame([record])],
                        ignore_index=True
                    )

                    temp_file = data_file + ".tmp"

                    try:
                        updated[columns].to_csv(
                            temp_file,
                            index=False,
                            encoding="utf-8-sig"
                        )
                        os.replace(temp_file, data_file)
                        st.success("Observation saved successfully.")
                        st.rerun()

                    except Exception as error:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        st.error(f"Could not save observation: {error}")

    st.divider()
    st.subheader("Traffic Analytics")

    observations = load_bhopal_observations()

    if not observations.empty:
        left, right = st.columns(2)

        with left:
            st.markdown("**Traffic Category Distribution**")
            st.bar_chart(
                observations["congestion"].value_counts().reindex(
                    classes, fill_value=0
                )
            )

        with right:
            st.markdown("**Road-wise Observations**")
            st.bar_chart(observations["road_name"].value_counts())

        st.subheader("Observation History")

        st.dataframe(
            observations.sort_values(
                "timestamp", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Bhopal Traffic Dataset",
            data=observations.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="bhopal_traffic_observations.csv",
            mime="text/csv"
        )

    else:
        st.info("No observations recorded yet. Add your first observation.")

    st.caption(
        "TrafficSense AI | Observation-based dataset, not live traffic."
    )


elif page == "Bhopal Road Network":

    st.subheader("🗺️ Bhopal Interactive Road Network")

    st.write(
        "Explore Bhopal's road infrastructure "
        "using OpenStreetMap data."
    )

    if not os.path.exists(BHOPAL_MAP_FILE):
        st.error("Bhopal GeoJSON file not found.")
        st.stop()

    try:

        roads = load_bhopal_roads()

        if roads.empty:
            st.warning("No road geometries available.")
            st.stop()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Road Segments",
            f"{len(roads):,}"
        )

        col2.metric(
            "Road Categories",
            roads["highway"].nunique()
        )

        col3.metric(
            "Named Roads",
            int((roads["name"].str.strip() != "").sum())
        )

        st.divider()

        center_lat = 23.2599
        center_lon = 77.4126

        bhopal_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=None,
            control_scale=True
        )

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="Street Map",
            control=True,
            show=True
        ).add_to(bhopal_map)

        folium.TileLayer(
            tiles="CartoDB positron",
            name="Light Map",
            control=True,
            show=False
        ).add_to(bhopal_map)

        folium.Marker(
            location=[center_lat, center_lon],
            tooltip="Bhopal, Madhya Pradesh",
            popup=folium.Popup(
                "<b>Bhopal</b><br>Madhya Pradesh, India",
                max_width=250
            ),
            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )
        ).add_to(bhopal_map)

        def road_style(feature):

            road_type = normalize_value(
                feature.get(
                    "properties", {}
                ).get(
                    "highway",
                    "other"
                )
            )

            return {
                "color": ROAD_COLORS.get(
                    road_type,
                    "#adb5bd"
                ),
                "weight": 2,
                "opacity": 0.8
            }

        folium.GeoJson(
            roads.to_json(),
            name="Bhopal Road Segments",
            style_function=road_style,
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "display_name",
                    "highway"
                ],
                aliases=[
                    "Road Name / Type:",
                    "Road Category:"
                ],
                sticky=True
            )
        ).add_to(bhopal_map)

        folium.LayerControl(
            collapsed=False
        ).add_to(bhopal_map)

        minx, miny, maxx, maxy = roads.total_bounds

        bhopal_map.fit_bounds([
            [miny, minx],
            [maxy, maxx]
        ])

        st.subheader("📍 Bhopal, Madhya Pradesh")

        st.caption(
            "Hover over any road to view its available name "
            "and road category. Zoom in for detailed exploration."
        )

        st_folium(
            bhopal_map,
            width=None,
            height=650,
            key="bhopal_interactive_map",
            returned_objects=[]
        )

        st.divider()
        st.subheader("📊 Road Type Distribution")

        road_counts = (
            roads["highway"]
            .value_counts()
            .head(12)
        )

        st.bar_chart(road_counts)

        st.caption(
            "Road geometry and map labels are provided through "
            "OpenStreetMap. Missing road names are displayed "
            "using their road category, not an invented street name. "
            "This map does not show live traffic."
        )

    except Exception as e:

        st.error(f"Unable to load Bhopal map: {e}")
        st.code(str(e))




# =====================================================
# 19. PROJECT EVALUATION & DOCUMENTATION
# =====================================================

elif page == "Project Evaluation & Documentation":

    st.title("📘 Project Evaluation & Documentation")
    st.markdown(
        "System health, dataset quality, model availability, "
        "project methodology and downloadable documentation."
    )

    st.info(
        "This page reports what is available in the local project files. "
        "A model being present does not by itself prove that it is accurate "
        "or suitable for real-world deployment."
    )

    # ------------------------------
    # SYSTEM HEALTH
    # ------------------------------
    st.header("1. System Health")

    def check_path(path):
        return os.path.exists(path)

    status_items = [
        ("Traffic model", TRAFFIC_MODEL_FILE),
        ("Accident severity model", ACCIDENT_MODEL_FILE),
        ("Traffic dataset", TRAFFIC_DATA_FILE),
        ("Accident dataset", ACCIDENT_DATA_FILE),
        ("Bhopal road GeoJSON", BHOPAL_MAP_FILE),
    ]

    health_rows = []
    for item_name, item_path in status_items:
        exists = check_path(item_path)
        try:
            file_size = (
                f"{os.path.getsize(item_path) / (1024 * 1024):.2f} MB"
                if exists and os.path.isfile(item_path)
                else ("Available" if exists else "Missing")
            )
        except OSError:
            file_size = "Unable to read"

        health_rows.append({
            "Component": item_name,
            "Status": "Available" if exists else "Missing",
            "File Size": file_size,
            "Path": os.path.relpath(item_path, BASE_DIR)
        })

    health_df = pd.DataFrame(health_rows)
    available_count = int(
        health_df["Status"].eq("Available").sum()
    )

    h1, h2, h3 = st.columns(3)
    h1.metric("Available Components", f"{available_count}/5")
    h2.metric(
        "Loaded ML Models",
        int(traffic_model is not None) + int(accident_model is not None)
    )
    h3.metric(
        "Loaded Datasets",
        int(not traffic_data.empty) + int(not accident_data.empty)
    )

    st.dataframe(
        health_df.drop(columns=["Path"]),
        use_container_width=True,
        hide_index=True
    )

    with st.expander("View local project file paths"):
        st.dataframe(
            health_df[["Component", "Path"]],
            use_container_width=True,
            hide_index=True
        )

    # ------------------------------
    # DATA QUALITY
    # ------------------------------
    st.divider()
    st.header("2. Dataset Quality Overview")

    def dataset_quality_report(frame, dataset_name):
        if frame is None or frame.empty:
            return {
                "Dataset": dataset_name,
                "Rows": 0,
                "Columns": 0,
                "Duplicate Rows": 0,
                "Missing Cells": 0,
                "Missing Cells (%)": 0.0
            }

        total_cells = int(frame.shape[0] * frame.shape[1])
        missing_cells = int(frame.isna().sum().sum())

        return {
            "Dataset": dataset_name,
            "Rows": int(frame.shape[0]),
            "Columns": int(frame.shape[1]),
            "Duplicate Rows": int(frame.duplicated().sum()),
            "Missing Cells": missing_cells,
            "Missing Cells (%)": (
                round(missing_cells / total_cells * 100, 2)
                if total_cells else 0.0
            )
        }

    quality_df = pd.DataFrame([
        dataset_quality_report(traffic_data, "Traffic dataset"),
        dataset_quality_report(accident_data, "Accident dataset")
    ])

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:
        st.markdown("**Traffic dataset columns**")
        if traffic_data is not None and not traffic_data.empty:
            st.write(", ".join(map(str, traffic_data.columns)))
        else:
            st.caption("Traffic dataset not loaded.")

    with quality_col2:
        st.markdown("**Accident dataset columns**")
        if accident_data is not None and not accident_data.empty:
            st.write(", ".join(map(str, accident_data.columns)))
        else:
            st.caption("Accident dataset not loaded.")

    # ------------------------------
    # MODEL AND DATA LIMITATIONS
    # ------------------------------
    st.divider()
    st.header("3. Model Scope & Limitations")

    st.markdown("""
    | Component | Current scope | Important limitation |
    |---|---|---|
    | Traffic congestion ML | Trained using the available Indore road/time dataset | Not validated as a Bhopal traffic forecast |
    | Bhopal traffic estimator | Rule-based demo using road category, hour and day | Simulated estimate, not measured or live traffic |
    | Accident severity model | Trained using sampled US accident records | Not validated for Bhopal or Indian road conditions |
    | Accident severity output | Classifies severity for a supplied scenario | Does not predict whether an accident will occur |
    | Accident hotspot map | Visualizes coordinates in the accident sample | The sample is not a Bhopal accident map |
    | Smart route analyzer | Uses the extracted Bhopal road graph and simulated edge traffic scores | Route cost and travel time are illustrative, not live navigation |
    """)

    st.warning(
        "Do not use these prototype outputs for emergency response, "
        "official traffic control, route safety decisions or real-world "
        "risk assessment."
    )

    # ------------------------------
    # PROJECT WORKFLOW
    # ------------------------------
    st.divider()
    st.header("4. Project Workflow")

    workflow = [
        "Collect and prepare traffic, accident and OpenStreetMap road data.",
        "Perform data cleaning, feature preparation and exploratory analysis.",
        "Train machine-learning models for traffic congestion and accident severity.",
        "Load trained models and datasets into the Streamlit application.",
        "Explore traffic patterns, accident records and road infrastructure.",
        "Compare road-network routes using distance and simulated traffic costs.",
        "Review model scope, data quality and limitations before any deployment."
    ]

    for step_number, step_text in enumerate(workflow, start=1):
        st.markdown(f"**Step {step_number}.** {step_text}")

    # ------------------------------
    # DOWNLOADABLE PROJECT REPORT
    # ------------------------------
    st.divider()
    st.header("5. Download Project Documentation")

    project_report = f"""# TrafficSense AI
## Smart Traffic & Accident Intelligence Platform

### Project Overview
TrafficSense AI is a Streamlit research prototype for exploring traffic congestion,
accident severity records, accident patterns, Bhopal road infrastructure and
road-network route comparisons.

### Technology Stack
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn models loaded through Joblib
- GeoPandas
- Folium
- OpenStreetMap road data

### Local Project Resources
| Resource | Status |
|---|---|
| Traffic congestion model | {"Available" if os.path.exists(TRAFFIC_MODEL_FILE) else "Missing"} |
| Accident severity model | {"Available" if os.path.exists(ACCIDENT_MODEL_FILE) else "Missing"} |
| Traffic dataset | {"Available" if os.path.exists(TRAFFIC_DATA_FILE) else "Missing"} |
| Accident dataset | {"Available" if os.path.exists(ACCIDENT_DATA_FILE) else "Missing"} |
| Bhopal road GeoJSON | {"Available" if os.path.exists(BHOPAL_MAP_FILE) else "Missing"} |

### Main Application Modules
1. Dashboard
2. Traffic Congestion
3. Bhopal Traffic Estimator
4. Accident Severity
5. Accident Severity Predictor
6. Accident Analytics
7. Accident Hotspot Map
8. AI Traffic Intelligence
9. Bhopal Smart Route Analyzer
10. Bhopal Road Network
11. Project Evaluation & Documentation

### Methodology
The application loads prepared datasets and trained machine-learning models.
It provides exploratory data analysis, scenario-based classification,
geospatial visualization and route comparison over a road graph.

### Important Limitations
- The traffic ML model uses Indore training data and is not validated for Bhopal.
- The Bhopal traffic estimator uses illustrative rules, not live traffic measurements.
- The accident model uses sampled US accident data and is not validated for India.
- Accident severity classification does not predict accident occurrence.
- The hotspot map reflects the geography of the supplied accident sample.
- Route traffic scores and route travel-time estimates are simulated.
- This application is a research/educational prototype, not a production traffic,
  emergency-response or road-safety system.

### Responsible Use
Validate all models with representative local data, document model metrics,
review class imbalance and obtain appropriate domain evaluation before deployment.

Generated from the local TrafficSense AI application.
"""

    st.download_button(
        label="⬇️ Download Project Report (Markdown)",
        data=project_report.encode("utf-8"),
        file_name="TrafficSense_AI_Project_Report.md",
        mime="text/markdown",
        use_container_width=True
    )

    if traffic_data is not None and not traffic_data.empty:
        st.download_button(
            label="⬇️ Download Traffic Dataset Copy",
            data=traffic_data.to_csv(index=False).encode("utf-8"),
            file_name="traffic_dataset_export.csv",
            mime="text/csv"
        )

    if accident_data is not None and not accident_data.empty:
        st.download_button(
            label="⬇️ Download Accident Dataset Copy",
            data=accident_data.to_csv(index=False).encode("utf-8"),
            file_name="accident_dataset_export.csv",
            mime="text/csv"
        )

    st.caption(
        "The report is generated from the application's configured resources "
        "and documented project scope. It does not certify model accuracy."
    )


# =====================================================
# 20. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#627d98;
        padding:10px;
    ">
        <b>🚦 TrafficSense AI</b><br>
        Smart Traffic & Accident Analysis Platform<br>
        Python | Machine Learning | Streamlit |
        GeoPandas | Folium | OpenStreetMap
    </div>
    """,
    unsafe_allow_html=True
)