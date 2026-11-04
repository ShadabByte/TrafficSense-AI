import os
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "bhopal_traffic_observations.csv")

os.makedirs(DATA_DIR, exist_ok=True)

CLASSES = ["FREE", "LIGHT", "MODERATE", "HEAVY"]
COLUMNS = ["record_id", "timestamp", "road_name", "congestion", "source", "notes"]

st.set_page_config(page_title="Bhopal Traffic Collector", page_icon="🚦", layout="wide")

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)

st.title("🚦 Bhopal Traffic Data Collector")
st.caption("TrafficSense AI | Bhopal Road Observation System")

st.info("Sirf actual observed traffic conditions ya verifiable sources se mili information enter karein. Ye module khud live traffic detect nahi karta.")

df = load_data()

a, b, c = st.columns(3)
a.metric("Total Observations", len(df))
b.metric("Unique Roads", df["road_name"].nunique() if len(df) else 0)
c.metric("Traffic Categories", df["congestion"].nunique() if len(df) else 0)

st.divider()
st.subheader("Add New Traffic Observation")

with st.form("observation_form"):
    road = st.text_input("Road / Junction Name", placeholder="Example: MP Nagar Zone 1")
    col1, col2 = st.columns(2)
    with col1:
        obs_date = st.date_input("Observation Date", value=date.today())
    with col2:
        obs_time = st.time_input("Observation Time", value=datetime.now().time().replace(second=0, microsecond=0))

    congestion = st.selectbox("Traffic Condition", CLASSES)
    source = st.text_input("Observation Source", placeholder="Personal observation / official source")
    notes = st.text_area("Additional Notes (Optional)")

    submitted = st.form_submit_button("Save Observation", type="primary")

    if submitted:
        if not road.strip():
            st.error("Road name is required.")
        elif not source.strip():
            st.error("Observation source is required.")
        else:
            record = {
                "record_id": str(uuid.uuid4()),
                "timestamp": datetime.combine(obs_date, obs_time).isoformat(),
                "road_name": road.strip(),
                "congestion": congestion,
                "source": source.strip(),
                "notes": notes.strip()
            }
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
            st.success("Observation saved successfully.")
            st.rerun()

st.divider()
st.subheader("Traffic Analytics")

df = load_data()

if not df.empty:
    left, right = st.columns(2)
    with left:
        st.markdown("**Traffic Category Distribution**")
        st.bar_chart(df["congestion"].value_counts())
    with right:
        st.markdown("**Road-wise Observations**")
        st.bar_chart(df["road_name"].value_counts())

    st.subheader("Observation History")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Bhopal Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bhopal_traffic_observations.csv",
        mime="text/csv"
    )
else:
    st.info("Abhi koi observation save nahi hui hai. Form se pehli observation add karein.")

st.caption("TrafficSense AI | Observation-based dataset. Not live traffic data.")
