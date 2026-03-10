import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except Exception:
    SARIMAX_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Riyadh Airport Analytics Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #0b1728 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081221 0%, #0d1b2a 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #0ea5e9 100%);
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: 0 18px 35px rgba(2, 6, 23, 0.35);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.3px;
    }

    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1rem;
        margin-top: 0.55rem;
        line-height: 1.7;
    }

    .mini-note {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }

    .kpi-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.22);
        min-height: 112px;
    }

    .kpi-title {
        color: #cbd5e1;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .kpi-value {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.05;
    }

    .kpi-sub {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }

    .section-head {
        color: white;
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 0.45rem;
        margin-bottom: 0.65rem;
    }

    .glass {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.16);
    }

    .insight {
        background: rgba(255,255,255,0.04);
        border-left: 4px solid #38bdf8;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        color: #e2e8f0;
    }

    .insight strong {
        color: #ffffff;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.86rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA SOURCE
# =========================================================
DATA_URL = "https://media.githubusercontent.com/media/FO7S/Riyadh-Airport-Flight-Analysis/refs/heads/main/flights_RUH.csv"


# =========================================================
# HELPERS
# =========================================================
def fmt_num(x):
    if pd.isna(x):
        return "-"
    return f"{int(x):,}"


def mode_safe(series):
    s = series.dropna()
    if s.empty:
        return None
    return s.mode().iloc[0]


def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def insight_box(title, text):
    st.markdown(
        f"""
        <div class="insight">
            <strong>{title}</strong><br>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def build_destination_map_data(filtered_df):
    airport_coords = {
        "Riyadh": (24.7136, 46.6753),
        "Jeddah": (21.5433, 39.1728),
        "Dammam": (26.4207, 50.0888),
        "Medina": (24.5247, 39.5692),
        "Abha": (18.2164, 42.5053),
        "Tabuk": (28.3838, 36.5662),
        "Taif": (21.2703, 40.4158),
        "Jazan": (16.8892, 42.5511),
        "Yanbu": (24.0895, 38.0618),
        "Al Ula": (26.6084, 37.9230),
        "Neom Bay Airport": (28.8337, 34.7871),
        "Arar": (30.9753, 41.0381),
        "Bisha": (19.9844, 42.6052),
        "Rafha": (29.6264, 43.4906),
        "Sharura": (17.4669, 47.1214),
        "Najran": (17.5650, 44.2289),
        "Hail": (27.5114, 41.7208),
        "Qassim": (26.3028, 43.7744),
        "Dubai": (25.2048, 55.2708),
        "Cairo": (30.0444, 31.2357),
        "Istanbul": (41.0082, 28.9784),
        "Doha": (25.2854, 51.5310),
        "Kuwait": (29.3759, 47.9774),
        "Amman": (31.9539, 35.9106),
        "Bahrain": (26.0667, 50.5577),
        "Abu Dhabi": (24.4539, 54.3773),
        "Sharjah": (25.3463, 55.4209),
        "Muscat": (23.5880, 58.3829),
        "Manama": (26.2235, 50.5876),
        "Beirut": (33.8938, 35.5018),
        "Baghdad": (33.3152, 44.3661),
        "Alexandria": (31.2001, 29.9187),
        "Sohag": (26.5560, 31.6948),
        "Asmara": (15.3229, 38.9251),
        "Khartoum": (15.5007, 32.5599),
        "Addis Ababa": (8.9806, 38.7578),
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Karachi": (24.8607, 67.0011),
        "Lahore": (31.5204, 74.3587),
        "London": (51.5072, -0.1276),
        "Paris": (48.8566, 2.3522),
        "Frankfurt": (50.1109, 8.6821),
        "Rome": (41.9028, 12.4964),
        "Vienna": (48.2082, 16.3738),
        "Madrid": (40.4168, -3.7038),
        "Athens": (37.9838, 23.7275),
        "Tunis": (36.8065, 10.1815),
        "Casablanca": (33.5731, -7.5898),
        "New York": (40.7128, -74.0060),
        "Washington": (38.9072, -77.0369),
        "Toronto": (43.6532, -79.3832),
        "Kuala Lumpur": (3.1390, 101.6869),
        "Jakarta": (-6.2088, 106.8456),
        "Singapore": (1.3521, 103.8198),
        "Bangkok": (13.7563, 100.5018),
        "Manila": (14.5995, 120.9842)
    }

    if "destination_airport_name" not in filtered_df.columns:
        return pd.DataFrame()

    map_df = (
        filtered_df.groupby("destination_airport_name")
        .size()
        .reset_index(name="Number of Flights")
        .rename(columns={"destination_airport_name": "Destination Airport"})
    )

    map_df["lat"] = map_df["Destination Airport"].map(lambda x: airport_coords.get(x, (None, None))[0])
    map_df["lon"] = map_df["Destination Airport"].map(lambda x: airport_coords.get(x, (None, None))[1])
    map_df = map_df.dropna(subset=["lat", "lon"]).copy()

    origin_df = pd.DataFrame({
        "Destination Airport": ["Riyadh"],
        "Number of Flights": [map_df["Number of Flights"].sum() if not map_df.empty else 0],
        "lat": [24.7136],
        "lon": [46.6753],
        "kind": ["origin"]
    })

    if not map_df.empty:
        map_df["kind"] = "destination"

    final_map_df = pd.concat([map_df, origin_df], ignore_index=True)
    return final_map_df


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_URL)


# =========================================================
# DATA PREPROCESSING
# =========================================================
@st.cache_data(show_spinner=False)
def preprocess_data(df):
    df = df.copy()

    df["movement.scheduledTime.local"] = pd.to_datetime(
        df["movement.scheduledTime.local"], errors="coerce"
    )

    df["movement.scheduledTime.utc"] = pd.to_datetime(
        df["movement.scheduledTime.utc"], errors="coerce"
    )

    try:
        if getattr(df["movement.scheduledTime.local"].dt, "tz", None) is not None:
            df["movement.scheduledTime.local"] = df["movement.scheduledTime.local"].dt.tz_localize(None)
    except Exception:
        pass

    try:
        if getattr(df["movement.scheduledTime.utc"].dt, "tz", None) is not None:
            df["movement.scheduledTime.utc"] = df["movement.scheduledTime.utc"].dt.tz_localize(None)
    except Exception:
        pass

    cols_to_drop = [
        "aircraft.reg",
        "aircraft.modeS",
        "callSign",
        "movement.airport.timeZone",
        "status",
        "aircraft.model",
        "codeshareStatus",
        "isCargo"
    ]
    existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    fill_map = {
        "destination_airport_iata": "Unknown",
        "destination_airport_icao": "Unknown",
        "airline.iata": "Unknown",
        "airline.icao": "Unknown"
    }
    for col, value in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    if "movement.terminal" in df.columns:
        df = df.dropna(subset=["movement.terminal"])

    if "destination_airport_name" in df.columns:
        df = df.dropna(subset=["destination_airport_name"])

    df = df.dropna(subset=["movement.scheduledTime.local"]).copy()

    df["date"] = df["movement.scheduledTime.local"].dt.floor("D")
    df["year"] = df["movement.scheduledTime.local"].dt.year
    df["month"] = df["movement.scheduledTime.local"].dt.month
    df["month_name"] = df["movement.scheduledTime.local"].dt.month_name()
    df["day"] = df["movement.scheduledTime.local"].dt.day
    df["day_of_week"] = df["movement.scheduledTime.local"].dt.day_name()
    df["hour"] = df["movement.scheduledTime.local"].dt.hour

    saudi_cities = {
        "Abha", "Ad Dammam", "Al Qaysumah", "Al Ula", "Al-Ula", "Al-Bakha",
        "Al-Jawf", "Arar", "Bisha", "Burayda", "Dammam", "Dawadmi", "Gerayat",
        "Jazan", "Jeddah", "Khail", "Kaysuma", "Medina", "Neyran",
        "Neom Bay Airport", "Rafha", "Sharura", "Tabuk", "Taif",
        "Turayf", "Vadi-ed-Davasir", "Yanbu"
    }

    df["destination_airport_name_clean"] = (
        df["destination_airport_name"]
        .astype(str)
        .str.strip()
    )

    df["route_type"] = df["destination_airport_name_clean"].apply(
        lambda x: "Domestic" if x in saudi_cities else "International"
    )

    df = df.sort_values("movement.scheduledTime.local")
    df = df.reset_index(drop=True)

    return df


def daily_series(df):
    d = (
        df.groupby("date")
        .size()
        .reset_index(name="Number of Flights")
        .sort_values("date")
    )
    d["7-Day Rolling Average"] = d["Number of Flights"].rolling(7).mean()
    return d


@st.cache_data(show_spinner=False)
def get_column_dictionary(df):
    descriptions = {
        "flight_number": "Unique identifier assigned to each flight departure record.",
        "airline.name": "Name of the airline operating the departure flight.",
        "airline.iata": "IATA airline code.",
        "airline.icao": "ICAO airline code.",
        "flight_type": "Flight classification provided in the source dataset.",
        "origin_airport_name": "Name of the origin airport.",
        "origin_airport_icao": "ICAO code of the origin airport.",
        "origin_airport_iata": "IATA code of the origin airport.",
        "destination_airport_name": "Name of the destination airport.",
        "destination_airport_icao": "ICAO code of the destination airport.",
        "destination_airport_iata": "IATA code of the destination airport.",
        "movement.terminal": "Airport terminal associated with the departure flight.",
        "movement.quality": "Quality or reliability indicator of the movement record.",
        "movement.scheduledTime.utc": "Scheduled departure time in UTC.",
        "movement.scheduledTime.local": "Scheduled departure time in local Riyadh time.",
        "date": "Calendar date extracted from the scheduled local departure time.",
        "year": "Year extracted from the scheduled local departure time.",
        "month": "Month number extracted from the scheduled local departure time.",
        "month_name": "Month name extracted from the scheduled local departure time.",
        "day": "Day of month extracted from the scheduled local departure time.",
        "day_of_week": "Weekday name extracted from the scheduled local departure time.",
        "hour": "Hour of day extracted from the scheduled local departure time.",
        "destination_airport_name_clean": "Cleaned destination airport name used for route classification.",
        "route_type": "Derived route type classified as Domestic or International based on destination airport name."
    }

    rows = []
    for col in df.columns:
        rows.append([
            col,
            str(df[col].dtype),
            descriptions.get(col, "No description added yet.")
        ])

    return pd.DataFrame(rows, columns=["Column Name", "Data Type", "Description"])


@st.cache_data(show_spinner=False)
def run_forecasting(daily_df):
    result = {"ready": False}

    if len(daily_df) < 30:
        return result

    ts = daily_df.copy().sort_values("date").reset_index(drop=True)

    if len(ts) > 2 and ts.loc[len(ts) - 1, "Number of Flights"] < ts["Number of Flights"].median() * 0.65:
        ts = ts.iloc[:-1].copy().reset_index(drop=True)

    ts["time_index"] = np.arange(len(ts))
    ts["day_of_week_num"] = ts["date"].dt.dayofweek
    ts["month_num"] = ts["date"].dt.month
    ts["is_weekend"] = ts["day_of_week_num"].isin([4, 5]).astype(int)

    split_idx = int(len(ts) * 0.8)
    train = ts.iloc[:split_idx].copy()
    test = ts.iloc[split_idx:].copy()

    feature_cols = ["time_index", "day_of_week_num", "month_num", "is_weekend"]

    lr = LinearRegression()
    lr.fit(train[feature_cols], train["Number of Flights"])
    test["Linear Regression Prediction"] = lr.predict(test[feature_cols])

    fitted_model = None
    if SARIMAX_AVAILABLE and len(train) >= 20:
        try:
            model = SARIMAX(
                train["Number of Flights"],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            test["SARIMAX Prediction"] = fitted_model.forecast(steps=len(test)).values
        except Exception:
            test["SARIMAX Prediction"] = np.nan
    else:
        test["SARIMAX Prediction"] = np.nan

    rows = []
    for model_name, col in [
        ("Linear Regression", "Linear Regression Prediction"),
        ("SARIMAX", "SARIMAX Prediction")
    ]:
        if test[col].notna().sum() > 0:
            mae = mean_absolute_error(test["Number of Flights"], test[col])
            rmse = np.sqrt(mean_squared_error(test["Number of Flights"], test[col]))
            rows.append([model_name, mae, rmse])

    results_df = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE"])

    future_days = 14
    last_date = ts["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq="D")

    future_lr = pd.DataFrame({
        "date": future_dates,
        "time_index": np.arange(len(ts), len(ts) + future_days)
    })
    future_lr["day_of_week_num"] = future_lr["date"].dt.dayofweek
    future_lr["month_num"] = future_lr["date"].dt.month
    future_lr["is_weekend"] = future_lr["day_of_week_num"].isin([4, 5]).astype(int)

    lr_future_pred = lr.predict(future_lr[feature_cols])

    if fitted_model is not None:
        try:
            sarimax_future_pred = fitted_model.forecast(steps=future_days).values
        except Exception:
            sarimax_future_pred = [np.nan] * future_days
    else:
        sarimax_future_pred = [np.nan] * future_days

    future_df = pd.DataFrame({
        "Forecast Date": future_dates,
        "Linear Regression Forecast": lr_future_pred,
        "SARIMAX Forecast": sarimax_future_pred
    })

    result.update({
        "ready": True,
        "ts": ts,
        "train": train,
        "test": test,
        "results": results_df,
        "future": future_df
    })
    return result


# =========================================================
# READ + PREPROCESS
# =========================================================
try:
    raw_df = load_data()
    df = preprocess_data(raw_df)
    column_dictionary_df = get_column_dictionary(df)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Dataset Description")
    st.markdown("""
    This dashboard analyzes **Riyadh Airport departure flights** and includes:
    - data preprocessing
    - airline analysis
    - destination analysis
    - domestic and international route analysis
    - traffic pattern exploration
    - short-term forecasting
    """)

    st.markdown("---")
    st.markdown("## Interactive Filters")

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    top_n = st.selectbox("Select Top N Records", [5, 10, 15, 20], index=1)

    hour_range = st.slider(
        "Select Hour Range",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )

    if "movement.terminal" in df.columns:
        terminal_options = sorted(df["movement.terminal"].dropna().astype(str).unique())
        terminal_selected = st.multiselect(
            "Select Terminal",
            terminal_options,
            default=terminal_options
        )
    else:
        terminal_selected = None

    if "airline.name" in df.columns:
        airline_options = sorted(df["airline.name"].dropna().astype(str).unique())
        airline_selected = st.multiselect(
            "Select Airlines",
            airline_options,
            default=airline_options
        )
    else:
        airline_selected = None

    search_text = st.text_input("Search by Destination or Airline", "")


# =========================================================
# APPLY FILTERS
# =========================================================
mask = (
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1])) &
    (df["hour"].between(hour_range[0], hour_range[1]))
)

if "movement.terminal" in df.columns and terminal_selected:
    mask &= df["movement.terminal"].astype(str).isin(terminal_selected)

if "airline.name" in df.columns and airline_selected:
    mask &= df["airline.name"].astype(str).isin(airline_selected)

filtered = df.loc[mask].copy()

if search_text.strip():
    q = search_text.strip().lower()
    search_mask = pd.Series(False, index=filtered.index)

    if "destination_airport_name" in filtered.columns:
        search_mask |= filtered["destination_airport_name"].astype(str).str.lower().str.contains(q, na=False)

    if "airline.name" in filtered.columns:
        search_mask |= filtered["airline.name"].astype(str).str.lower().str.contains(q, na=False)

    filtered = filtered.loc[search_mask].copy()

if filtered.empty:
    st.error("No rows match the selected filters. Please adjust the filters and try again.")
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>Riyadh Airport Analytics Dashboard</h1>
    <p>
        Interactive dashboard for analyzing Riyadh Airport departure traffic, airline activity,
        destination demand, operational distribution, and short-term forecasting.
    </p>
    <div class="mini-note">Prepared by: Faisal Al-Sulami</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# KPI SECTION
# =========================================================
total_flights = len(filtered)
unique_airlines = filtered["airline.name"].nunique() if "airline.name" in filtered.columns else np.nan
unique_destinations = filtered["destination_airport_name"].nunique() if "destination_airport_name" in filtered.columns else np.nan
busiest_terminal = mode_safe(filtered["movement.terminal"]) if "movement.terminal" in filtered.columns else "-"
dominant_route = mode_safe(filtered["route_type"]) if "route_type" in filtered.columns else "-"

hour_df = (
    filtered.groupby("hour")
    .size()
    .reset_index(name="Number of Flights")
    .sort_values("hour")
)

peak_hour = int(hour_df.loc[hour_df["Number of Flights"].idxmax(), "hour"]) if not hour_df.empty else None

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Total Flights", fmt_num(total_flights), "Flights after filtering")
with c2:
    kpi_card("Active Airlines", fmt_num(unique_airlines), "Distinct operating airlines")
with c3:
    kpi_card("Unique Destinations", fmt_num(unique_destinations), "Distinct destination airports")
with c4:
    kpi_card("Busiest Terminal", str(busiest_terminal), "Highest traffic concentration")
with c5:
    kpi_card("Peak Hour", f"{peak_hour}:00" if peak_hour is not None else "-", "Hour with highest activity")


# =========================================================
# DATA PREVIEW + SUMMARY
# =========================================================
left_preview, right_preview = st.columns([1.15, 0.85])

with left_preview:
    st.markdown('<div class="section-head">Filtered Data Preview</div>', unsafe_allow_html=True)

    preview_cols = [
        col for col in [
            "movement.scheduledTime.local",
            "airline.name",
            "route_type",
            "destination_airport_name",
            "movement.terminal",
            "hour",
            "day_of_week",
            "month_name"
        ] if col in filtered.columns
    ]

    st.dataframe(filtered[preview_cols].head(15), use_container_width=True, hide_index=True)

with right_preview:
    st.markdown('<div class="section-head">Summary Statistics</div>', unsafe_allow_html=True)

    top_airline = mode_safe(filtered["airline.name"]) if "airline.name" in filtered.columns else "-"
    top_destination = mode_safe(filtered["destination_airport_name"]) if "destination_airport_name" in filtered.columns else "-"

    summary = pd.DataFrame({
        "Metric": [
            "Rows after preprocessing and filtering",
            "Earliest date",
            "Latest date",
            "Dominant route type",
            "Most active airline",
            "Top destination"
        ],
        "Value": [
            len(filtered),
            str(filtered["date"].min().date()),
            str(filtered["date"].max().date()),
            dominant_route,
            top_airline,
            top_destination
        ]
    })
    st.table(summary)


# =========================================================
# DAILY SERIES
# =========================================================
daily_df = daily_series(filtered)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Traffic Patterns", "Destinations", "Forecasting", "Insights", "Data Overview"]
)


# =========================================================
# TAB 1 OVERVIEW
# =========================================================
with tab1:
    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown('<div class="section-head">Daily Flight Trend</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_df["date"],
            y=daily_df["Number of Flights"],
            mode="lines",
            name="Daily Flights",
            line=dict(width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=daily_df["date"],
            y=daily_df["7-Day Rolling Average"],
            mode="lines",
            name="7-Day Rolling Average",
            line=dict(width=3)
        ))

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Date",
            yaxis_title="Number of Flights",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-head">Executive Snapshot</div>', unsafe_allow_html=True)

        avg_daily = daily_df["Number of Flights"].mean()
        median_daily = daily_df["Number of Flights"].median()

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="small-muted">
        The selected data shows an average of <b>{avg_daily:,.0f}</b> flights per day and a median of
        <b>{median_daily:,.0f}</b>. This gives a concise view of daily traffic intensity and helps identify
        whether operations are concentrated around a stable level or affected by fluctuations.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2 TRAFFIC PATTERNS
# =========================================================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-head">Flights by Hour of Day</div>', unsafe_allow_html=True)
        fig_hour = px.area(
            hour_df,
            x="hour",
            y="Number of Flights",
            markers=True
        )
        fig_hour.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Hour of Day",
            yaxis_title="Number of Flights",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with c2:
        st.markdown('<div class="section-head">Flights by Day of Week</div>', unsafe_allow_html=True)

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df = filtered.groupby("day_of_week").size().reset_index(name="Number of Flights")
        dow_df["day_of_week"] = pd.Categorical(dow_df["day_of_week"], categories=day_order, ordered=True)
        dow_df = dow_df.sort_values("day_of_week")

        fig_dow = px.bar(
            dow_df,
            x="day_of_week",
            y="Number of Flights",
            text_auto=True
        )
        fig_dow.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Day of Week",
            yaxis_title="Number of Flights",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown('<div class="section-head">Flights by Month</div>', unsafe_allow_html=True)

    month_df = (
        filtered.groupby(["month", "month_name"])
        .size()
        .reset_index(name="Number of Flights")
        .sort_values("month")
    )

    fig_month = px.line(
        month_df,
        x="month_name",
        y="Number of Flights",
        markers=True
    )
    fig_month.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Month",
        yaxis_title="Number of Flights",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_month, use_container_width=True)


# =========================================================
# TAB 3 DESTINATIONS
# =========================================================
with tab3:
    d1, d2 = st.columns(2)

    if "destination_airport_name" in filtered.columns and "route_type" in filtered.columns:
        top_international = (
            filtered[filtered["route_type"] == "International"]["destination_airport_name"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )

        if not top_international.empty:
            top_international.columns = ["Destination Airport", "Number of Flights"]

            with d1:
                st.markdown('<div class="section-head">Top 10 International Destinations from Riyadh</div>', unsafe_allow_html=True)
                fig_int = px.bar(
                    top_international,
                    x="Destination Airport",
                    y="Number of Flights",
                    text="Number of Flights"
                )
                fig_int.update_layout(
                    height=430,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis_title="International Destination Airport",
                    yaxis_title="Number of Flights",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig_int.update_traces(textposition="outside")
                st.plotly_chart(fig_int, use_container_width=True)
        else:
            with d1:
                st.markdown('<div class="section-head">Top 10 International Destinations from Riyadh</div>', unsafe_allow_html=True)
                st.info("No international destination data is available for the selected filters.")

        top_domestic = (
            filtered[filtered["route_type"] == "Domestic"]["destination_airport_name"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )

        if not top_domestic.empty:
            top_domestic.columns = ["Destination Airport", "Number of Flights"]

            with d2:
                st.markdown('<div class="section-head">Top 10 Domestic Destinations from Riyadh</div>', unsafe_allow_html=True)
                fig_dom = px.bar(
                    top_domestic,
                    x="Destination Airport",
                    y="Number of Flights",
                    text="Number of Flights"
                )
                fig_dom.update_layout(
                    height=430,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis_title="Domestic Destination Airport",
                    yaxis_title="Number of Flights",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig_dom.update_traces(textposition="outside")
                st.plotly_chart(fig_dom, use_container_width=True)
        else:
            with d2:
                st.markdown('<div class="section-head">Top 10 Domestic Destinations from Riyadh</div>', unsafe_allow_html=True)
                st.info("No domestic destination data is available for the selected filters.")

    st.markdown('<div class="section-head">Geographic Destination Heatmap</div>', unsafe_allow_html=True)

    map_df = build_destination_map_data(filtered)

    if not map_df.empty:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="Number of Flights",
            color="kind",
            hover_name="Destination Airport",
            hover_data={
                "lat": False,
                "lon": False,
                "Number of Flights": True,
                "kind": True
            },
            zoom=3.2,
            height=620,
            size_max=38,
            mapbox_style="carto-darkmatter"
        )

        fig_map.update_traces(
            marker=dict(opacity=0.82, line=dict(width=1.2, color="white"))
        )

        fig_map.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Location Type"
        )

        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("""
        <div class="glass">
        This geographic view highlights Riyadh as the central origin point and displays destination airports
        according to their flight volume. Larger circles indicate higher traffic concentration, helping users
        quickly identify the strongest regional and international connections.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No mapped destination coordinates are currently available for the selected filters.")

    st.markdown('<div class="section-head">Top Airlines by Number of Flights</div>', unsafe_allow_html=True)

    if "airline.name" in filtered.columns:
        airline_df = (
            filtered.groupby("airline.name")
            .size()
            .reset_index(name="Number of Flights")
            .sort_values("Number of Flights", ascending=False)
            .head(top_n)
        )
        airline_df = airline_df.rename(columns={"airline.name": "Airline Name"})

        fig_air = px.bar(
            airline_df,
            x="Airline Name",
            y="Number of Flights",
            text="Number of Flights"
        )
        fig_air.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Airline",
            yaxis_title="Number of Flights",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        fig_air.update_traces(textposition="outside")
        st.plotly_chart(fig_air, use_container_width=True)

    if "movement.terminal" in filtered.columns:
        st.markdown('<div class="section-head">Flight Distribution by Terminal</div>', unsafe_allow_html=True)

        terminal_df = (
            filtered.groupby("movement.terminal")
            .size()
            .reset_index(name="Number of Flights")
            .sort_values("Number of Flights", ascending=False)
        )

        terminal_df = terminal_df.rename(columns={"movement.terminal": "Terminal"})

        fig_terminal = px.bar(
            terminal_df,
            x="Terminal",
            y="Number of Flights",
            text="Number of Flights"
        )

        fig_terminal.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Airport Terminal",
            yaxis_title="Number of Flights",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        fig_terminal.update_traces(textposition="outside")
        st.plotly_chart(fig_terminal, use_container_width=True)


# =========================================================
# TAB 4 FORECASTING
# =========================================================
with tab4:
    st.markdown('<div class="section-head">Short-Term Forecasting</div>', unsafe_allow_html=True)

    fc = run_forecasting(daily_df)

    if not fc["ready"]:
        st.info("There are not enough daily observations to build the forecasting models.")
    else:
        f1, f2 = st.columns([1.15, 0.85])

        with f1:
            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(
                x=fc["train"]["date"],
                y=fc["train"]["Number of Flights"],
                mode="lines",
                name="Training Data"
            ))
            fig_split.add_trace(go.Scatter(
                x=fc["test"]["date"],
                y=fc["test"]["Number of Flights"],
                mode="lines",
                name="Actual Test Data"
            ))
            fig_split.add_trace(go.Scatter(
                x=fc["test"]["date"],
                y=fc["test"]["Linear Regression Prediction"],
                mode="lines",
                name="Linear Regression Prediction",
                line=dict(dash="dash")
            ))

            if fc["test"]["SARIMAX Prediction"].notna().sum() > 0:
                fig_split.add_trace(go.Scatter(
                    x=fc["test"]["date"],
                    y=fc["test"]["SARIMAX Prediction"],
                    mode="lines",
                    name="SARIMAX Prediction",
                    line=dict(dash="dash")
                ))

            fig_split.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis_title="Date",
                yaxis_title="Number of Flights",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_split, use_container_width=True)

        with f2:
            st.markdown("#### Forecasting Model Error Comparison")
            st.dataframe(fc["results"], use_container_width=True, hide_index=True)

            if not fc["results"].empty:
                fig_err = go.Figure()
                fig_err.add_trace(go.Bar(
                    x=fc["results"]["Model"],
                    y=fc["results"]["MAE"],
                    name="MAE"
                ))
                fig_err.add_trace(go.Bar(
                    x=fc["results"]["Model"],
                    y=fc["results"]["RMSE"],
                    name="RMSE"
                ))
                fig_err.update_layout(
                    barmode="group",
                    height=330,
                    margin=dict(l=10, r=10, t=15, b=10),
                    xaxis_title="Forecasting Model",
                    yaxis_title="Error Value",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("#### Next 14-Day Forecast")
        fig_future = go.Figure()

        fig_future.add_trace(go.Scatter(
            x=fc["ts"]["date"],
            y=fc["ts"]["Number of Flights"],
            mode="lines",
            name="Historical Flights",
            line=dict(width=3)
        ))
        fig_future.add_trace(go.Scatter(
            x=fc["future"]["Forecast Date"],
            y=fc["future"]["Linear Regression Forecast"],
            mode="lines+markers",
            name="Linear Regression Forecast",
            line=dict(dash="dash")
        ))

        if fc["future"]["SARIMAX Forecast"].notna().sum() > 0:
            fig_future.add_trace(go.Scatter(
                x=fc["future"]["Forecast Date"],
                y=fc["future"]["SARIMAX Forecast"],
                mode="lines+markers",
                name="SARIMAX Forecast",
                line=dict(dash="dash")
            ))

        fig_future.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Forecast Date",
            yaxis_title="Number of Flights",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_future, use_container_width=True)


# =========================================================
# TAB 5 INSIGHTS
# =========================================================
with tab5:
    st.markdown('<div class="section-head">Automated Insight Summary</div>', unsafe_allow_html=True)

    if "destination_airport_name" in filtered.columns and filtered["destination_airport_name"].notna().sum() > 0:
        top_destination = filtered["destination_airport_name"].value_counts().idxmax()
        top_destination_count = int(filtered["destination_airport_name"].value_counts().max())
        insight_box(
            "Top Destination Overall",
            f"<b>{top_destination}</b> is the most frequent destination in the filtered dataset with <b>{top_destination_count:,}</b> flights."
        )

    if "airline.name" in filtered.columns and filtered["airline.name"].notna().sum() > 0:
        top_airline = filtered["airline.name"].value_counts().idxmax()
        top_airline_count = int(filtered["airline.name"].value_counts().max())
        insight_box(
            "Most Active Airline",
            f"<b>{top_airline}</b> records the highest number of departures in the selected period with <b>{top_airline_count:,}</b> flights."
        )

    if "movement.terminal" in filtered.columns and filtered["movement.terminal"].notna().sum() > 0:
        top_terminal = filtered["movement.terminal"].value_counts().idxmax()
        top_terminal_count = int(filtered["movement.terminal"].value_counts().max())
        insight_box(
            "Terminal Concentration",
            f"<b>Terminal {top_terminal}</b> handles the largest share of departures in the current filtered view with <b>{top_terminal_count:,}</b> flights."
        )

    peak_day_df = filtered.groupby("day_of_week").size().sort_values(ascending=False)
    if not peak_day_df.empty:
        insight_box(
            "Peak Operating Day",
            f"<b>{peak_day_df.index[0]}</b> has the highest departure activity among all weekdays in the current selection."
        )

    month_df = (
        filtered.groupby(["month", "month_name"])
        .size()
        .reset_index(name="Number of Flights")
        .sort_values("month")
    )
    if not month_df.empty:
        best_month = month_df.sort_values("Number of Flights", ascending=False).iloc[0]
        insight_box(
            "Strongest Month",
            f"<b>{best_month['month_name']}</b> is the strongest month in the selected range with <b>{int(best_month['Number of Flights']):,}</b> flights."
        )

    if "route_type" in filtered.columns:
        route_counts = filtered["route_type"].value_counts()
        if "International" in route_counts.index:
            insight_box(
                "International Route Demand",
                f"The dashboard records <b>{int(route_counts['International']):,}</b> international departures in the filtered view."
            )
        if "Domestic" in route_counts.index:
            insight_box(
                "Domestic Route Demand",
                f"The dashboard records <b>{int(route_counts['Domestic']):,}</b> domestic departures in the filtered view."
            )

    st.markdown('<div class="section-head">Extended Business Insights</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
    The exploratory analysis revealed several important operational patterns:
    <br><br>
    <b>• Stable flight activity:</b><br>
    Daily flight counts fluctuate, but overall they remain within a relatively consistent range, indicating stable airport operations.
    <br><br>
    <b>• Regional connectivity dominance:</b><br>
    Many international routes connect Riyadh with nearby regional hubs such as <b>Dubai, Cairo, and Istanbul</b>, highlighting the strong importance of regional travel demand.
    <br><br>
    <b>• Airline activity concentration:</b><br>
    A small number of airlines account for a large share of departures, which is typical for major hub airports where traffic is concentrated among key carriers.
    <br><br>
    <b>• Terminal utilization imbalance:</b><br>
    The analysis showed that <b>Terminal 5 handled the largest share of flights</b>, mainly because it served a substantial portion of domestic routes. This created higher operational pressure compared with other terminals.
    <br><br>
    <b>• Major operational adjustment:</b><br>
    A notable operational change occurred on <b>February 25, 2026</b>, when domestic flights were redistributed to <b>Terminals 3 and 4</b> at Riyadh Airport. This real-world adjustment illustrates how airport data analysis can support <b>data-driven operational decisions</b> that improve passenger flow and reduce congestion.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Download Filtered Data</div>', unsafe_allow_html=True)
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Dataset as CSV",
        data=csv_data,
        file_name="riyadh_airport_filtered_data.csv",
        mime="text/csv"
    )

    st.markdown('<div class="section-head">Manager Takeaway</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
    This dashboard converts Riyadh Airport departure data into an interactive decision-support tool.
    It supports understanding demand patterns, identifying major routes and airlines, monitoring terminal usage,
    and reviewing short-term flight forecasting in a professional and portfolio-ready format.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 6 DATA OVERVIEW
# =========================================================
with tab6:
    st.markdown('<div class="section-head">Processed Dataset Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
    This tab presents the dataset after preprocessing.
    The preprocessing workflow included:
    <br><br>
    • converting local and UTC time columns into datetime format  
    • removing highly incomplete or less useful columns  
    • filling selected missing categorical values  
    • dropping rows with missing terminal or destination names  
    • creating time-based features such as year, month, weekday, and hour  
    • deriving a route type label (Domestic / International) based on destination airport name  
    • removing the <b>isCargo</b> column because it was not required for the final dashboard analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Dataset Summary</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        kpi_card("Rows", fmt_num(df.shape[0]), "After preprocessing")
    with d2:
        kpi_card("Columns", fmt_num(df.shape[1]), "Available fields")
    with d3:
        kpi_card("Missing Values", fmt_num(df.isna().sum().sum()), "Across processed dataset")

    st.markdown('<div class="section-head">Processed Data Table</div>', unsafe_allow_html=True)
    rows_to_show = st.slider(
        "Select number of rows to display",
        min_value=50,
        max_value=min(5000, len(df)),
        value=min(200, len(df)),
        step=50
    )
    st.dataframe(df.head(rows_to_show), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-head">Column Dictionary</div>', unsafe_allow_html=True)
    st.dataframe(column_dictionary_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-head">Data Types and Missing Values</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": [df[col].nunique(dropna=True) for col in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


st.markdown("---")
st.caption("Built with Streamlit • Professional Riyadh Airport Analytics Dashboard.")