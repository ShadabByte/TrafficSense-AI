from pathlib import Path
from datetime import datetime
import shutil
import sys

app = Path("app.py")
if not app.exists():
    raise SystemExit("ERROR: app.py nahi mili.")

source = app.read_text(encoding="utf-8")

anchor = '                        st.write("**Roads included in selected route:**")'

if source.count(anchor) != 1:
    raise SystemExit(
        "ERROR: Expected route-table location exactly once nahi mila. "
        "app.py mein koi change nahi kiya."
    )

insert = r'''
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

'''

backup = Path(
    "app_before_route_live_patch_" +
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".py"
)
shutil.copy2(app, backup)

updated = source.replace(anchor, insert + "\n" + anchor, 1)

try:
    compile(updated, str(app), "exec")
    app.write_text(updated, encoding="utf-8")
    compile(app.read_text(encoding="utf-8"), str(app), "exec")
    print("SUCCESS: Live traffic integration added.")
    print(f"Backup: {backup}")
    print("Python syntax check: PASSED")
except Exception as error:
    shutil.copy2(backup, app)
    print("PATCH FAILED. Original app.py restored.")
    print(f"Reason: {error}")
    sys.exit(1)
