
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "accident_severity_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "accident_training_sample.csv"

st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# FEATURES
# --------------------------------------------------
FEATURES = [
    "start_lat", "start_lng",
    "temperature(f)", "wind_chill(f)", "humidity(%)",
    "pressure(in)", "visibility(mi)", "wind_direction",
    "wind_speed(mph)", "precipitation(in)", "weather_condition",
    "amenity", "bump", "crossing", "give_way", "junction",
    "no_exit", "railway", "roundabout", "station", "stop",
    "traffic_calming", "traffic_signal", "turning_loop",
    "sunrise_sunset", "civil_twilight", "nautical_twilight",
    "astronomical_twilight", "start_time_hour", "start_time_day",
    "start_time_month", "start_time_weekday", "start_time_weekend",
    "start_time_rush_hour", "start_time_hour_sin",
    "start_time_hour_cos"
]

NUMERIC_FEATURES = [
    "start_lat", "start_lng", "temperature(f)",
    "wind_chill(f)", "humidity(%)", "pressure(in)",
    "visibility(mi)", "wind_speed(mph)", "precipitation(in)"
]

BOOLEAN_FEATURES = [
    "amenity", "bump", "crossing", "give_way", "junction",
    "no_exit", "railway", "roundabout", "station", "stop",
    "traffic_calming", "traffic_signal", "turning_loop"
]

CATEGORICAL_FEATURES = [
    "wind_direction", "weather_condition", "sunrise_sunset",
    "civil_twilight", "nautical_twilight",
    "astronomical_twilight"
]

# --------------------------------------------------
# LOAD MODEL AND SAMPLE DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sample():
    return pd.read_csv(DATA_PATH)


try:
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    if not DATA_PATH.exists():
        st.error(f"Training sample not found: {DATA_PATH}")
        st.stop()

    model = load_model()
    sample = load_sample()

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def numeric_default(column):
    values = pd.to_numeric(sample[column], errors="coerce")
    value = values.median()
    return float(value) if pd.notna(value) else 0.0


def category_options(column):
    values = sample[column].dropna().astype(str)
    options = values.value_counts().index.tolist()

    if not options:
        return ["Unknown"]

    return options


def default_category(column):
    options = category_options(column)
    return options[0]


def severity_label(value):
    labels = {
        1: "Severity 1 - Lowest category",
        2: "Severity 2",
        3: "Severity 3",
        4: "Severity 4 - Highest category"
    }
    try:
        return labels.get(int(value), f"Severity {value}")
    except (ValueError, TypeError):
        return str(value)


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
st.title("🚦 AI Accident Severity Predictor")

st.markdown(
    "Predict an experimental accident severity category "
    "using weather, road infrastructure and time conditions."
)

st.warning(
    "Research demo only: This model is trained on US accident data. "
    "It is not validated for Bhopal or India. It estimates severity "
    "conditional on an accident scenario; it does not predict whether "
    "an accident will occur."
)

st.divider()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
with st.form("accident_prediction_form"):

    st.subheader("1. Location")

    st.caption(
        "Coordinates are latitude and longitude. "
        "The model was trained on US records."
    )

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude",
            value=numeric_default("start_lat"),
            format="%.6f"
        )

    with col2:
        longitude = st.number_input(
            "Longitude",
            value=numeric_default("start_lng"),
            format="%.6f"
        )

    st.divider()
    st.subheader("2. Weather Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°F)",
            value=numeric_default("temperature(f)")
        )

        wind_chill = st.number_input(
            "Wind Chill (°F)",
            value=numeric_default("wind_chill(f)")
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(np.clip(numeric_default("humidity(%)"), 0, 100))
        )

    with col2:
        pressure = st.number_input(
            "Pressure (in)",
            min_value=0.0,
            value=max(0.0, numeric_default("pressure(in)"))
        )

        visibility = st.number_input(
            "Visibility (miles)",
            min_value=0.0,
            value=max(0.0, numeric_default("visibility(mi)"))
        )

        wind_speed = st.number_input(
            "Wind Speed (mph)",
            min_value=0.0,
            value=max(0.0, numeric_default("wind_speed(mph)"))
        )

    with col3:
        precipitation = st.number_input(
            "Precipitation (in)",
            min_value=0.0,
            value=max(0.0, numeric_default("precipitation(in)"))
        )

        wind_direction = st.selectbox(
            "Wind Direction",
            category_options("wind_direction")
        )

        weather = st.selectbox(
            "Weather Condition",
            category_options("weather_condition")
        )

    st.divider()
    st.subheader("3. Road Infrastructure")

    st.caption("Select the road features present in the scenario.")

    boolean_values = {}

    cols = st.columns(3)

    for i, feature in enumerate(BOOLEAN_FEATURES):
        with cols[i % 3]:
            boolean_values[feature] = st.checkbox(
                feature.replace("_", " ").title(),
                value=False
            )

    st.divider()
    st.subheader("4. Time Conditions")

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input("Date")

    with col2:
        selected_time = st.time_input("Time")

    col1, col2, col3 = st.columns(3)

    with col1:
        sunrise_sunset = st.selectbox(
            "Sunrise / Sunset",
            category_options("sunrise_sunset")
        )

    with col2:
        civil_twilight = st.selectbox(
            "Civil Twilight",
            category_options("civil_twilight")
        )

    with col3:
        nautical_twilight = st.selectbox(
            "Nautical Twilight",
            category_options("nautical_twilight")
        )

    astronomical_twilight = st.selectbox(
        "Astronomical Twilight",
        category_options("astronomical_twilight")
    )

    submitted = st.form_submit_button(
        "🚦 Predict Accident Severity",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:

    hour = selected_time.hour
    day = selected_date.day
    month = selected_date.month
    weekday = selected_date.weekday()

    is_weekend = int(weekday >= 5)
    is_rush_hour = int(
        (7 <= hour <= 10) or (16 <= hour <= 19)
    )

    row = {
        "start_lat": latitude,
        "start_lng": longitude,
        "temperature(f)": temperature,
        "wind_chill(f)": wind_chill,
        "humidity(%)": humidity,
        "pressure(in)": pressure,
        "visibility(mi)": visibility,
        "wind_direction": wind_direction,
        "wind_speed(mph)": wind_speed,
        "precipitation(in)": precipitation,
        "weather_condition": weather,
        **boolean_values,
        "sunrise_sunset": sunrise_sunset,
        "civil_twilight": civil_twilight,
        "nautical_twilight": nautical_twilight,
        "astronomical_twilight": astronomical_twilight,
        "start_time_hour": hour,
        "start_time_day": day,
        "start_time_month": month,
        "start_time_weekday": weekday,
        "start_time_weekend": is_weekend,
        "start_time_rush_hour": is_rush_hour,
        "start_time_hour_sin": np.sin(2 * np.pi * hour / 24),
        "start_time_hour_cos": np.cos(2 * np.pi * hour / 24)
    }

    input_df = pd.DataFrame([row], columns=FEATURES)

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        classifier = model.named_steps["classifier"]
        classes = classifier.classes_

        st.divider()
        st.subheader("5. Prediction Result")

        st.success(
            f"Predicted category: {severity_label(prediction)}"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Predicted Severity", str(prediction))

        with result_col2:
            st.metric(
                "Prediction confidence",
                f"{max(probabilities) * 100:.2f}%"
            )

        st.subheader("Class Probability")

        probability_df = pd.DataFrame({
            "Severity": [
                severity_label(c) for c in classes
            ],
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Severity")
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "These probabilities are model outputs, not verified "
            "real-world accident risks. A high probability does not "
            "mean an accident will happen."
        )

        with st.expander("View model input data"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "TrafficSense AI | Accident Severity Prediction | "
    "Experimental ML demonstration"
)
