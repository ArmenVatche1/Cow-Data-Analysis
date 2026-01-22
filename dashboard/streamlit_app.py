import os
import requests
import pandas as pd
import streamlit as st
from datetime import date

API_BASE_DEFAULT = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Cow Milk Monitor", layout="wide")
st.title("🐄 Cow Milk Monitor (Prototype v4)")
st.caption("Scenario-based demo data + Herd View + ML evaluation + Docker-ready.")

with st.sidebar:
    st.header("API")
    api_base = st.text_input("API Base URL", API_BASE_DEFAULT)
    st.caption("If using Docker Compose, this should already be set.")

def get_json(url: str):
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json()

def post_json(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# Load cows
try:
    cows = get_json(f"{api_base}/cows")
    cow_ids = [c["cow_id"] for c in cows]
    cow_map = {c["cow_id"]: c for c in cows}
except Exception as e:
    st.error(f"Could not fetch cows from API: {e}")
    st.stop()

tabs = st.tabs(["Herd View", "Scan", "Add Cow", "Add Daily Record", "Alerts", "ML", "Recent Scans"])

# --------------------
# Herd View
# --------------------
with tabs[0]:
    st.subheader("Herd View")
    st.caption("Rank cows by risk, filter, then drill down into a cow.")

    colF1, colF2, colF3 = st.columns(3)
    with colF1:
        status_filter = st.selectbox("Status filter", ["(all)", "FLAG", "WARNING", "OK", "UNKNOWN"], index=0)
    with colF2:
        min_risk = st.slider("Min risk score", 0, 100, 0)
    with colF3:
        breed_filter = st.selectbox("Breed filter", ["(all)"] + sorted(set([c["breed"] for c in cows])))

    try:
        url = f"{api_base}/alerts"
        if status_filter != "(all)":
            url += f"?status={status_filter}"
        alerts = get_json(url)
        df = pd.DataFrame(alerts)
        if df.empty:
            st.info("No alerts returned.")
        else:
            if breed_filter != "(all)":
                df = df[df["breed"] == breed_filter]
            df = df[df["risk_score"].fillna(0) >= min_risk]

            df = df.sort_values(by=["risk_score", "difference_pct"], ascending=[False, True])
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download herd table CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="herd_view.csv",
                mime="text/csv",
            )

            st.divider()
            st.subheader("Drill-down")
            choice = st.selectbox("Select cow to drill down", options=df["cow_id"].tolist() if not df.empty else cow_ids)

            if choice:
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.markdown("### Scan result")
                    try:
                        scan = get_json(f"{api_base}/scan/{choice}?include_forecast=true")
                        st.success(f"Status: {scan['status']} | Risk: {scan.get('risk_score')}")
                        st.write("Reasons:", scan.get("reasons", []))
                        recs = scan.get("recommendations", [])
                        if recs:
                            st.markdown("#### Recommendations")
                            for r in recs:
                                st.write(f"- {r}")
                        if scan.get("ml_forecast_7d"):
                            df_fc = pd.DataFrame(scan["ml_forecast_7d"])
                            df_fc["date"] = pd.to_datetime(df_fc["date"])
                            st.markdown("#### Forecast (7d)")
                            st.line_chart(df_fc.set_index("date")[["predicted_milk_liters"]])
                    except Exception as e:
                        st.error(f"Scan failed: {e}")

                with c2:
                    st.markdown("### Last 30 days signals")
                    try:
                        records = get_json(f"{api_base}/cows/{choice}/records?days=30")
                        dfR = pd.DataFrame(records)
                        if dfR.empty:
                            st.info("No records to plot.")
                        else:
                            dfR["date"] = pd.to_datetime(dfR["date"])
                            dfR = dfR.sort_values("date")
                            k1, k2 = st.columns(2)
                            with k1:
                                st.markdown("#### Milk (L)")
                                st.line_chart(dfR.set_index("date")[["milk_liters"]])
                                st.markdown("#### Feed (kg)")
                                st.line_chart(dfR.set_index("date")[["feed_intake_kg"]])
                            with k2:
                                st.markdown("#### Temp (°C)")
                                st.line_chart(dfR.set_index("date")[["body_temp_c"]])
                                st.markdown("#### Rumination + Activity")
                                st.line_chart(dfR.set_index("date")[["rumination_min", "activity_index"]])
                            st.markdown("#### Environment")
                            st.line_chart(dfR.set_index("date")[["ambient_temp_c", "humidity_pct"]])
                    except Exception as e:
                        st.warning(f"Could not load records: {e}")

    except Exception as e:
        st.error(f"Could not load herd alerts: {e}")

# --------------------
# Scan
# --------------------
with tabs[1]:
    st.subheader("Scan Cow")
    selected = st.selectbox("Cow ID", options=cow_ids, index=0)
    include_forecast = st.checkbox("Include ML forecast (if trained)", value=True)

    if st.button("Scan"):
        try:
            data = get_json(f"{api_base}/scan/{selected}?include_forecast={'true' if include_forecast else 'false'}")
            st.success(f"Status: {data['status']} | Risk score: {data.get('risk_score', 'N/A')}")
            st.write("Reasons:", data.get("reasons", []))

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Actual milk (L)", data.get("actual_liters"))
            m2.metric("Expected milk (L)", data.get("expected_liters"))
            m3.metric("Δ %", data.get("difference_pct"))
            m4.metric("Feed (kg)", data.get("feed_intake_kg"))
            m5.metric("Temp (°C)", data.get("body_temp_c"))

            recs = data.get("recommendations", [])
            if recs:
                st.markdown("### Recommendations")
                for r in recs:
                    st.write(f"- {r}")

            fc = data.get("ml_forecast_7d")
            if fc:
                df_fc = pd.DataFrame(fc)
                df_fc["date"] = pd.to_datetime(df_fc["date"])
                st.markdown("### ML Forecast (next 7 days)")
                st.line_chart(df_fc.set_index("date")[["predicted_milk_liters"]])

            st.markdown("### Raw JSON")
            st.json(data)
        except Exception as e:
            st.error(f"Scan failed: {e}")

# --------------------
# Add Cow
# --------------------
with tabs[2]:
    st.subheader("Register New Cow")
    with st.form("cow_form"):
        cow_id = st.text_input("Cow ID (unique)", value="")
        breed = st.text_input("Breed", value="Holstein")
        birth_date = st.date_input("Birth date", value=date(2020, 1, 1))
        last_calving_date = st.date_input("Last calving date", value=date.today())
        parity = st.number_input("Parity (calvings)", min_value=0, max_value=20, value=1)
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=550.0)
        submitted = st.form_submit_button("Create cow")

    if submitted:
        try:
            payload = {
                "cow_id": cow_id.strip(),
                "breed": breed.strip(),
                "birth_date": birth_date.isoformat(),
                "last_calving_date": last_calving_date.isoformat(),
                "parity": int(parity),
                "weight_kg": float(weight_kg),
            }
            resp = post_json(f"{api_base}/cows", payload)
            st.success(resp)
            st.info("Refresh the page to see the cow in dropdowns.")
        except Exception as e:
            st.error(f"Create cow failed: {e}")

# --------------------
# Add Daily Record
# --------------------
with tabs[3]:
    st.subheader("Add Daily Record")
    cow_for_record = st.selectbox("Cow", options=cow_ids, index=0, key="cow_for_record")

    with st.form("record_form"):
        rec_date = st.date_input("Record date", value=date.today())
        milk = st.number_input("Milk liters", min_value=0.0, value=28.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            feed = st.number_input("Feed intake (kg)", min_value=0.0, value=20.0)
            rum = st.number_input("Rumination (min)", min_value=0.0, value=420.0)
            eat = st.number_input("Eating (min)", min_value=0.0, value=300.0)
        with c2:
            temp = st.number_input("Body temp (°C)", min_value=30.0, max_value=45.0, value=38.6)
            act = st.number_input("Activity index", min_value=0.0, value=55.0)
        with c3:
            amb = st.number_input("Ambient temp (°C)", value=18.0)
            hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

        submitted = st.form_submit_button("Create record")

    if submitted:
        try:
            payload = {
                "cow_id": cow_for_record,
                "date": rec_date.isoformat(),
                "milk_liters": float(milk),
                "feed_intake_kg": float(feed),
                "body_temp_c": float(temp),
                "rumination_min": float(rum),
                "eating_min": float(eat),
                "activity_index": float(act),
                "ambient_temp_c": float(amb),
                "humidity_pct": float(hum),
            }
            resp = post_json(f"{api_base}/records", payload)
            st.success(resp)
        except Exception as e:
            st.error(f"Create record failed: {e}")

# --------------------
# Alerts
# --------------------
with tabs[4]:
    st.subheader("Alerts")
    status_filter = st.selectbox("Filter", ["(all)", "FLAG", "WARNING", "OK", "UNKNOWN"], index=0, key="alerts_status")
    if st.button("Load alerts"):
        try:
            url = f"{api_base}/alerts"
            if status_filter != "(all)":
                url += f"?status={status_filter}"
            data = get_json(url)
            dfA = pd.DataFrame(data)
            if dfA.empty:
                st.info("No results.")
            else:
                st.dataframe(dfA, use_container_width=True)
                st.download_button(
                    "Download alerts CSV",
                    dfA.to_csv(index=False).encode("utf-8"),
                    file_name="alerts.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Load alerts failed: {e}")

# --------------------
# ML
# --------------------
with tabs[5]:
    st.subheader("ML Model")
    st.caption("Train, evaluate, then forecast.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Train model (POST /ml/train)"):
            try:
                resp = post_json(f"{api_base}/ml/train", {})
                st.success(resp)
            except Exception as e:
                st.error(f"Train failed: {e}")

    with col2:
        if st.button("Evaluate model (GET /ml/evaluate)"):
            try:
                metrics = get_json(f"{api_base}/ml/evaluate?test_size=0.2")
                st.success(metrics)
            except Exception as e:
                st.error(f"Evaluate failed: {e}")

    with col3:
        cow_ml = st.selectbox("Cow for forecast", options=cow_ids, index=0, key="cow_ml")
        days = st.slider("Days", min_value=1, max_value=30, value=7)

    if st.button("Get forecast"):
        try:
            resp = get_json(f"{api_base}/predict/{cow_ml}?days={days}")
            df = pd.DataFrame(resp["forecast"])
            df["date"] = pd.to_datetime(df["date"])
            st.line_chart(df.set_index("date")[["predicted_milk_liters"]])
            st.json(resp)
        except Exception as e:
            st.error(f"Forecast failed: {e}")

    st.divider()
    st.markdown("### Demo cows to try")
    st.write("- DEMO-A-HEALTHY (should look good)")
    st.write("- DEMO-B-HEAT (heat stress causes decline near the end)")
    st.write("- DEMO-C-FEVER (fever/appetite drop then milk drop)")
    st.write("- DEMO-D-FEED (appetite issue)")
    st.write("- DEMO-E-WARN (mild underperformance)")

# --------------------
# Recent scans
# --------------------
with tabs[6]:
    st.subheader("Recent scan audit trail")
    try:
        scans = get_json(f"{api_base}/scans/recent?limit=50")
        dfS = pd.DataFrame(scans)
        if dfS.empty:
            st.info("No scans yet. Use Scan tab.")
        else:
            st.dataframe(dfS, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load recent scans: {e}")
