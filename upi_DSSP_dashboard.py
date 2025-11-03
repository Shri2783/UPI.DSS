from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import random

# ---------- Page config & styles ----------
st.set_page_config(page_title="UPI DSS ", layout="wide", page_icon="💳")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2.0rem;}
      .kpi-card {
        border-radius: 16px; padding: 16px 18px; border: 1px solid rgba(120,120,120,0.15);
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
      }
      .kpi-label {font-size: 0.85rem; opacity: 0.8; margin-bottom: 6px;}
      .kpi-value {font-size: 1.6rem; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💳 UPI Transaction Decision Support System (Public)")

# ---------- Demo dataset (so the public can try without uploading) ----------
def make_demo_rows(n_users=50, days=30, seed=42):
    rng = np.random.default_rng(seed)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)

    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata"]
    devices = ["and_12", "and_13", "ios_16", "ios_17"]
    users = [f"U{1000+i}" for i in range(n_users)]

    rows = []
    txn_id = 1
    for day in range(days):
        for u in users:
            # Poisson-ish daily count
            k = rng.poisson(2.2)
            for _ in range(k):
                ts = start + timedelta(days=day, hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
                amount = max(1, rng.normal(1500, 900))
                # inject some high amounts occasionally
                if rng.random() < 0.03:
                    amount = rng.normal(20000, 5000)

                city = random.choice(cities)
                device = random.choice(devices)
                hour = ts.hour

                rows.append({
                    "txn_id": f"T{txn_id:07d}",
                    "user_id": u,
                    "timestamp": ts.strftime("%d-%m-%Y %H:%M"),
                    "amount": round(float(amount), 2),
                    "device_id": device,
                    "city": city,
                    "hour": hour,
                    "device_change": int(rng.random() < 0.05),
                    "geo_change": int(rng.random() < 0.04),
                    "if_score": np.nan,          # placeholder (we can compute)
                    "if_label": 0,               # 0/1; precomputed optional
                    "flag": 0,
                })
                txn_id += 1
    demo = pd.DataFrame(rows)
    return demo

# ---------- Sidebar: data source ----------
with st.sidebar:
    st.header("📁 Data Source")
    st.write(
        "Upload a CSV with these columns:\n\n"
        "`txn_id, user_id, timestamp, amount, device_id, city, hour, "
        "device_change, geo_change, if_score, if_label, flag`"
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_demo = st.toggle("Use demo dataset (ignore upload)", value=True)

# ---------- Load data ----------
@st.cache_data(show_spinner=True)
def load_df(file_like_or_buffer):
    df = pd.read_csv(file_like_or_buffer)
    df.columns = [c.strip() for c in df.columns]
    return df

if use_demo:
    df = make_demo_rows()
    src_name = "demo_generated.csv"
else:
    if uploaded is None:
        st.info("Upload a CSV or enable the demo dataset in the sidebar.")
        st.stop()
    df = load_df(uploaded)
    src_name = getattr(uploaded, "name", "uploaded.csv")

st.success(f"✅ Loaded: **{src_name}**")

# ---------- Validate schema ----------
expected = [
    "txn_id", "user_id", "timestamp", "amount", "device_id", "city", "hour",
    "device_change", "geo_change", "if_score", "if_label", "flag",
]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# ---------- Parse & normalize ----------
def parse_timestamp_col(s):
    s = s.astype(str).str.strip()
    fmts = ["%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]
    for fmt in fmts:
        try:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

df["timestamp_parsed"] = parse_timestamp_col(df["timestamp"])
n_total = len(df)
df = df.dropna(subset=["timestamp_parsed"]).reset_index(drop=True)
n_after = len(df)
if n_after < n_total:
    st.warning(f"Dropped {n_total - n_after} rows with invalid timestamps; remaining {n_after} rows.")

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.dropna(subset=["amount"]).reset_index(drop=True)

df["date"] = df["timestamp_parsed"].dt.date
df["year"] = df["timestamp_parsed"].dt.year
df["month"] = df["timestamp_parsed"].dt.month_name()
df["month_num"] = df["timestamp_parsed"].dt.month
df["hour_parsed"] = (
    pd.to_numeric(df["hour"], errors="coerce")
      .fillna(df["timestamp_parsed"].dt.hour)
      .round()
      .clip(0, 23)
      .astype(int)
)

df["device_change_flag"] = pd.to_numeric(df["device_change"], errors="coerce").fillna(0).astype(int)
df["geo_change_flag"] = pd.to_numeric(df["geo_change"], errors="coerce").fillna(0).astype(int)
df["_if_pre"] = pd.to_numeric(df["if_label"], errors="coerce").fillna(0).astype(int)
df["_if_score_pre"] = pd.to_numeric(df["if_score"], errors="coerce")

# ---------- Sidebar: controls ----------
with st.sidebar:
    st.header("🎛️ Controls")
    dark = st.toggle("Dark mode charts", value=False)
    PLT_TPL = "plotly_dark" if dark else "simple_white"

    users = ["All"] + sorted(df["user_id"].astype(str).unique().tolist())
    cities = ["All"] + sorted(df["city"].astype(str).unique().tolist())
    years = ["All"] + sorted(df["year"].dropna().astype(int).unique().tolist())
    months = ["All"] + sorted(df["month"].dropna().unique().tolist())

    sel_user = st.selectbox("User", users)
    sel_city = st.selectbox("City", cities)
    sel_year = st.selectbox("Year", years)
    sel_month = st.selectbox("Month", months)

    amt_min = float(df["amount"].min())
    amt_max = float(df["amount"].max())
    sel_amt = st.slider("Amount range", amt_min, amt_max, (amt_min, amt_max), step=max(1.0, (amt_max-amt_min)/200))

    st.divider()
    st.subheader("📝Isolation Forest")
    use_precomputed_if = st.checkbox("Prefer precomputed if_label", value=True)
    recompute_if = st.checkbox("Allow recomputing on filtered data", value=False)
    contamination = st.slider("Contamination", 0.001, 0.2, 0.05, step=0.001)

    st.subheader("🧷 Rule Flags")
    high_k = st.slider("High-value threshold = mean + k·std", 1.0, 5.0, 3.0, step=0.1)
    odd_start = st.number_input("Odd-hour start (0–23)", min_value=0, max_value=23, value=23)
    odd_end = st.number_input("Odd-hour end (0–23)", min_value=0, max_value=23, value=5)

    run_button = st.button(" 📥 Apply & Update")

# ---------- Filters ----------
view = df.copy()
view = view[(view["amount"] >= sel_amt[0]) & (view["amount"] <= sel_amt[1])]
if sel_user != "All":
    view = view[view["user_id"].astype(str) == sel_user]
if sel_city != "All":
    view = view[view["city"].astype(str) == sel_city]
if sel_year != "All":
    view = view[view["year"] == int(sel_year)]
if sel_month != "All":
    view = view[view["month"] == sel_month]

if view.empty:
    st.warning("No records match selected filters. Adjust filters.")
    st.stop()

# ---------- Isolation Forest ----------
if (view["_if_pre"].sum() > 0) and use_precomputed_if and (not recompute_if):
    view["_if_used"] = view["_if_pre"]
    view["_if_score"] = view["_if_score_pre"]
else:
    features = ["amount", "hour_parsed"]
    X = view[features].fillna(view[features].median())
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)
        preds = model.predict(X)  # -1 anomaly, 1 normal
        view["_if_used"] = np.where(preds == -1, 1, 0)
        view["_if_score"] = pd.Series(model.decision_function(X), index=view.index)
    except Exception as e:
        st.error(f"IsolationForest failed: {e}")
        view["_if_used"] = 0
        view["_if_score"] = np.nan

# ---------- Rule flags ----------
mean_amt = view["amount"].mean()
std_amt = view["amount"].std() if not np.isnan(view["amount"].std()) else 0.0
high_threshold = mean_amt + high_k * std_amt
view["flag_high_value"] = (view["amount"] >= high_threshold).astype(int)

def odd_flag(h):
    try:
        h = float(h)
    except Exception:
        return 0
    if odd_start <= odd_end:
        return 1 if (h >= odd_start and h <= odd_end) else 0
    else:
        return 1 if (h >= odd_start or h <= odd_end) else 0

view["flag_odd_hour"] = view["hour_parsed"].apply(odd_flag).astype(int)
view["flag_device_change"] = view["device_change_flag"].astype(int)
view["flag_geo_change"] = view["geo_change_flag"].astype(int)

# burst: user >5 txns same date
view["flag_burst"] = 0
for u, g in view.groupby("user_id"):
    heavy_dates = g.groupby("date").size()
    heavy_dates = heavy_dates[heavy_dates > 5].index
    if len(heavy_dates):
        idxs = g[g["date"].isin(heavy_dates)].index
        view.loc[idxs, "flag_burst"] = 1

# ---------- Severity ----------
w_if, w_high, w_odd, w_device, w_geo, w_burst = 3, 2, 1, 1, 1, 2
view["severity_score"] = (
    w_if * view["_if_used"].fillna(0).astype(int)
    + w_high * view["flag_high_value"]
    + w_odd * view["flag_odd_hour"]
    + w_device * view["flag_device_change"]
    + w_geo * view["flag_geo_change"]
    + w_burst * view["flag_burst"]
)

def severity_label(s):
    if s >= 8: return "High"
    if s >= 4: return "Medium"
    if s >= 1: return "Low"
    return "None"

view["severity"] = view["severity_score"].apply(severity_label)

def suggestion_text(r):
    reasons = []
    if int(r["_if_used"]) == 1: reasons.append("AI anomaly")
    if int(r["flag_high_value"]) == 1: reasons.append("High value")
    if int(r["flag_odd_hour"]) == 1: reasons.append("Odd hour")
    if int(r["flag_burst"]) == 1: reasons.append("Burst")
    if int(r["flag_device_change"]) == 1: reasons.append("Device change")
    if int(r["flag_geo_change"]) == 1: reasons.append("Geo change")
    if not reasons: return "No immediate concern"

    adv = []
    if "High value" in reasons: adv.append("Verify beneficiary & purpose; consider hold.")
    if "AI anomaly" in reasons: adv.append("Prioritize manual review; check beneficiary chain.")
    if "Odd hour" in reasons: adv.append("Contact user; check login activity.")
    if "Burst" in reasons: adv.append("Investigate repeated transfers; consider temporary limits.")
    if "Device change" in reasons: adv.append("Review device binding & OTP/IP logs.")
    if "Geo change" in reasons: adv.append("Verify IP/location history & recent travel.")
    return " / ".join(adv)

view["suggestion"] = view.apply(suggestion_text, axis=1)

# ---------- Require button ----------
if not run_button:
    st.info("Adjust filters and press ** 📥 Apply & Update**.")
    st.stop()

# ---------- KPI Row ----------
st.header(" Summary (Filtered View)")
k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Transactions</div><div class="kpi-value">{len(view):,}</div></div>""", unsafe_allow_html=True)
k2.markdown(f"""<div class="kpi-card"><div class="kpi-label">AI Anomalies</div><div class="kpi-value">{int(view["_if_used"].sum()):,}</div></div>""", unsafe_allow_html=True)
k3.markdown(f"""<div class="kpi-card"><div class="kpi-label">High Severity</div><div class="kpi-value">{int((view["severity"]=="High").sum()):,}</div></div>""", unsafe_allow_html=True)
k4.markdown(f"""<div class="kpi-card"><div class="kpi-label">High-Value Threshold</div><div class="kpi-value">{high_threshold:.2f}</div></div>""", unsafe_allow_html=True)
k5.markdown(f"""<div class="kpi-card"><div class="kpi-label">Average Amount</div><div class="kpi-value">{view['amount'].mean():.2f}</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🕒 Time Profiles", "🌍 City & Users", "🧩 Distribution & Correlation"])

with tab1:
    st.subheader("Severity Distribution")
    sev = (view["severity"].value_counts()
           .reindex(["High", "Medium", "Low", "None"])
           .fillna(0).reset_index())
    sev.columns = ["severity", "count"]
    fig_sev = px.pie(
        sev, names="severity", values="count", hole=0.45,
        color="severity",
        template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white"),
        color_discrete_map={"High":"#b30000","Medium":"#ff7f0e","Low":"#ffcc00","None":"#d3d3d3"},
    )
    fig_sev.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_sev, use_container_width=True)

    st.subheader("Daily Transactions & Anomalies")
    daily = view.groupby("date").agg(total=("txn_id","count"), anomalies=("_if_used","sum")).reset_index().sort_values("date")
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(x=daily["date"], y=daily["total"], name="Total"))
    fig_daily.add_trace(go.Scatter(x=daily["date"], y=daily["anomalies"], name="Anomalies", yaxis="y2", mode="lines+markers"))
    fig_daily.update_layout(template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white"),
                            yaxis2=dict(overlaying="y", side="right", title="Anomalies"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(t=30))
    st.plotly_chart(fig_daily, use_container_width=True)

with tab2:
    st.subheader("Monthly & Yearly Aggregates")
    colm, coly = st.columns(2)
    month_agg = view.groupby("month").agg(txns=("txn_id","count"), anomalies=("_if_used","sum")).reset_index()
    year_agg = view.groupby("year").agg(txns=("txn_id","count"), anomalies=("_if_used","sum")).reset_index()

    colm.plotly_chart(px.bar(month_agg, x="month", y="txns", title="Transactions by Month", template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white")), use_container_width=True)
    coly.plotly_chart(px.bar(year_agg, x="year", y="txns", title="Transactions by Year", template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white")), use_container_width=True)

    st.subheader("Hourly Profile (Transactions vs Anomalies)")
    hourly = (view.groupby("hour_parsed")
              .agg(txns=("txn_id","count"), anomalies=("_if_used","sum"))
              .reset_index().sort_values("hour_parsed"))
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(x=hourly["hour_parsed"], y=hourly["txns"], name="Transactions"))
    fig_h.add_trace(go.Scatter(x=hourly["hour_parsed"], y=hourly["anomalies"], name="Anomalies", yaxis="y2", mode="lines+markers"))
    fig_h.update_layout(template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white"),
                        xaxis_title="Hour of Day",
                        yaxis2=dict(overlaying="y", side="right", title="Anomalies"),
                        margin=dict(t=30))
    st.plotly_chart(fig_h, use_container_width=True)

with tab3:
    st.subheader("City-level Analysis")
    city_agg = (view.groupby("city")
                .agg(txns=("txn_id","count"), anomalies=("_if_used","sum"), avg_amt=("amount","mean"))
                .reset_index().sort_values("txns", ascending=False))
    st.plotly_chart(px.bar(city_agg.head(20), x="city", y="txns", color="anomalies",
                           title="Top Cities by Transactions (color = anomalies)",
                           template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white")),
                    use_container_width=True)
    st.plotly_chart(px.scatter(city_agg, x="txns", y="avg_amt", hover_data=["city"],
                               title="City: Transactions vs Avg Amount",
                               template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white")),
                    use_container_width=True)

    st.subheader("Top Suspicious Users")
    user_agg = (view.groupby("user_id")
                .agg(txns=("txn_id","count"),
                     anomalies=("_if_used","sum"),
                     highsev=("severity", lambda s: (s=="High").sum()))
                .reset_index()
                .sort_values("anomalies", ascending=False)
                .head(25))
    fig_users = px.bar(user_agg, x="user_id", y="anomalies", color="highsev",
                       title="Top Users by Anomalies (color = # High severity)",
                       template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white"))
    st.plotly_chart(fig_users, use_container_width=True)

with tab4:
    st.subheader("Amount Distribution by Severity")
    st.plotly_chart(px.histogram(view, x="amount", color="severity", nbins=50,
                                 title="Amount Distribution (by severity)",
                                 template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white")),
                    use_container_width=True)

    st.subheader("Numeric Correlation")
    numcols = ["amount", "hour_parsed", "_if_score"]
    existing_numcols = [c for c in numcols if c in view.columns]
    if len(existing_numcols) >= 2:
        corr = view[existing_numcols].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix",
                             template=("plotly_dark" if st.session_state.get("dark_mode", False) else "simple_white"))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

# ---------- Flagged table & download ----------
st.subheader("🚩 Flagged Transactions (severity ≠ None)")
flagged = view[view["severity"] != "None"].sort_values("severity_score", ascending=False)
if not flagged.empty:
    cols_show = [
        "txn_id", "user_id", "timestamp", "amount", "city", "hour_parsed", "_if_used",
        "flag_high_value", "flag_odd_hour", "flag_burst", "flag_device_change", "flag_geo_change",
        "severity", "severity_score", "suggestion",
    ]
    cols_show = [c for c in cols_show if c in flagged.columns]
    st.dataframe(flagged[cols_show].head(1000), use_container_width=True)
    st.download_button(
        "💾 Download flagged CSV",
        flagged.to_csv(index=False).encode("utf-8"),
        "flagged_upi.csv",
        "text/csv"
    )
else:
    st.info("No flagged transactions found for current filters.")

# ---------- Recommendations ----------
st.subheader(" 👩🏻‍💻 Recommendations")
recs = []
if int((view["severity"] == "High").sum()) > 0:
    recs.append(f"Investigate **{int((view['severity']=='High').sum())}** high-severity transactions immediately.")
if view["flag_high_value"].sum() > 0:
    recs.append("High-value transactions detected → verify beneficiaries and consider temporary holds.")
if view["flag_burst"].sum() > 0:
    recs.append("Burst activity detected → review recent activity and consider per-user limits.")
if view["flag_device_change"].sum() > 0:
    recs.append("Device-change flags present → check device history and authentication logs.")
if view["flag_geo_change"].sum() > 0:
    recs.append("Geo-change flags present → verify IP/location history and user travel.")
if not recs:
    recs.append("No urgent actions required for current filters.")

for r in recs:
    st.markdown(f"- {r}")

st.markdown("---")
st.caption("Made by 22MIY0014|22MIY0015|22MIY0070 ")

