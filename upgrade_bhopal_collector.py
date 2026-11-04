from pathlib import Path
from datetime import datetime, date
import shutil
import sys

app = Path("app.py")
source = app.read_text(encoding="utf-8-sig")

start = 'elif page == "Bhopal Traffic Data Collector":'
end = 'elif page == "Bhopal Road Network":'

if source.count(start) != 1 or source.count(end) != 1:
    raise SystemExit("ERROR: Collector section anchors match nahi hue. No changes made.")

start_index = source.index(start)
end_index = source.index(end, start_index)

if end_index <= start_index:
    raise SystemExit("ERROR: Collector section boundaries invalid.")

new_block = r'''
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


'''

backup = Path("app_before_collector_upgrade_backup.py")
if backup.exists():
    from datetime import datetime
    backup = Path(
        "app_before_collector_upgrade_" +
        datetime.now().strftime("%Y%m%d_%H%M%S") + ".py"
    )

shutil.copy2(app, backup)

updated_source = (
    source[:start_index] +
    new_block +
    source[end_index:]
)

try:
    compile(updated_source, str(app), "exec")
    app.write_text(updated_source, encoding="utf-8")
    compile(app.read_text(encoding="utf-8-sig"), str(app), "exec")

    print("SUCCESS: Bhopal Collector upgraded.")
    print("Duplicate protection: Added")
    print("Future timestamp validation: Added")
    print("Dataset progress tracking: Added")
    print("Existing CSV data: Preserved")
    print("Syntax check: PASSED")
    print(f"Backup created: {backup}")

except Exception as error:
    shutil.copy2(backup, app)
    print("FAILED: Original app.py restored.")
    print(f"Reason: {error}")
    sys.exit(1)
