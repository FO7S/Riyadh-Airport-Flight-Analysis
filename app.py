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
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


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

    .subtle-note {
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.18);
        border-radius: 14px;
        padding: 0.75rem 0.95rem;
        color: #dbeafe;
        margin-top: 0.6rem;
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
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


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


def safe_pct(part, total):
    if total == 0:
        return 0
    return (part / total) * 100


def add_headroom(max_value, ratio=0.12):
    if pd.isna(max_value) or max_value == 0:
        return 1
    return max_value * (1 + ratio)


def build_destination_map_data(filtered_df):
    airport_coords = {
        "Riyadh": (24.7136, 46.6753),
        "Jeddah": (21.5433, 39.1728),
        "Dammam": (26.4207, 50.0888),
        "Ad Dammam": (26.4207, 50.0888),
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
# PREPROCESSING
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

    df["destination_airport_name_clean"] = df["destination_airport_name"].astype(str).str.strip()

    df["route_type"] = df["destination_airport_name_clean"].apply(
        lambda x: "Domestic" if x in saudi_cities else "International"
    )

    df = df.sort_values("movement.scheduledTime.local").reset_index(drop=True)
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
        rows.append([col, str(df[col].dtype), descriptions.get(col, "No description added yet.")])

    return pd.DataFrame(rows, columns=["Column Name", "Data Type", "Description"])


@st.cache_data(show_spinner=False)
def get_quality_report(raw_df, processed_df):
    missing_percentage = (raw_df.isnull().sum() / len(raw_df) * 100).sort_values(ascending=False).reset_index()
    missing_percentage.columns = ["Column", "Missing %"]

    missing_ratio = raw_df.isnull().mean()
    cols_over_50 = missing_ratio[missing_ratio > 0.5].index.tolist()

    text_columns = raw_df.select_dtypes(include=["object"]).columns.tolist()
    datetime_like_columns = [col for col in raw_df.columns if "time" in col.lower() or "date" in col.lower()]

    summary = {
        "raw_rows": raw_df.shape[0],
        "raw_cols": raw_df.shape[1],
        "processed_rows": processed_df.shape[0],
        "processed_cols": processed_df.shape[1],
        "columns_over_50_missing": cols_over_50,
        "text_columns": text_columns,
        "datetime_like_columns": datetime_like_columns
    }

    return missing_percentage, summary


# =========================================================
# FORECASTING
# =========================================================
@st.cache_data(show_spinner=False)
def run_forecasting(daily_df):
    result = {"ready": False}

    if len(daily_df) < 30:
        return result

    ts = daily_df.copy().sort_values("date").reset_index(drop=True)
    ts = ts.rename(columns={"Number of Flights": "flights"})

    if len(ts) > 2 and ts.loc[len(ts) - 1, "flights"] < ts["flights"].median() * 0.65:
        ts = ts.iloc[:-1].copy().reset_index(drop=True)

    split_idx = int(len(ts) * 0.8)
    train = ts.iloc[:split_idx].copy()
    test = ts.iloc[split_idx:].copy()

    # Linear Regression
    train_lr = train.copy()
    test_lr = test.copy()

    train_lr["time_index"] = range(len(train_lr))
    test_lr["time_index"] = range(len(train_lr), len(train_lr) + len(test_lr))

    lr_model = LinearRegression()
    lr_model.fit(train_lr[["time_index"]], train_lr["flights"])
    test["Linear Regression"] = lr_model.predict(test_lr[["time_index"]])

    # ARIMA
    if ARIMA_AVAILABLE:
        try:
            arima_model = ARIMA(train["flights"], order=(5, 1, 0))
            arima_fitted = arima_model.fit()
            test["ARIMA"] = arima_fitted.forecast(steps=len(test)).values
        except Exception:
            test["ARIMA"] = np.nan
    else:
        test["ARIMA"] = np.nan

    # Prophet
    if PROPHET_AVAILABLE:
        try:
            prophet_train = train[["date", "flights"]].rename(columns={"date": "ds", "flights": "y"}).copy()
            prophet_train["ds"] = pd.to_datetime(prophet_train["ds"]).dt.tz_localize(None)

            prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            prophet_model.fit(prophet_train)

            future_prophet_test = pd.DataFrame({"ds": pd.to_datetime(test["date"]).dt.tz_localize(None)})
            prophet_pred_test = prophet_model.predict(future_prophet_test)
            test["Prophet"] = prophet_pred_test["yhat"].values
        except Exception:
            test["Prophet"] = np.nan
    else:
        test["Prophet"] = np.nan

    metrics_rows = []
    for model_name in ["Linear Regression", "ARIMA", "Prophet"]:
        if model_name in test.columns and test[model_name].notna().sum() > 0:
            mae = mean_absolute_error(test["flights"], test[model_name])
            rmse = np.sqrt(mean_squared_error(test["flights"], test[model_name]))
            metrics_rows.append([model_name, mae, rmse])

    results_df = pd.DataFrame(metrics_rows, columns=["Model", "MAE", "RMSE"]).sort_values("RMSE")

    # Future 14-day forecast
    future_days = 14
    last_date = ts["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq="D")

    # LR future
    future_lr = pd.DataFrame({"time_index": range(len(ts), len(ts) + future_days)})
    lr_future_pred = lr_model.predict(future_lr[["time_index"]])

    # ARIMA future
    if ARIMA_AVAILABLE and "ARIMA" in test.columns and test["ARIMA"].notna().sum() > 0:
        try:
            arima_model_full = ARIMA(ts["flights"], order=(5, 1, 0))
            arima_full_fitted = arima_model_full.fit()
            arima_future_pred = arima_full_fitted.forecast(steps=future_days).values
        except Exception:
            arima_future_pred = [np.nan] * future_days
    else:
        arima_future_pred = [np.nan] * future_days

    # Prophet future
    if PROPHET_AVAILABLE and "Prophet" in test.columns and test["Prophet"].notna().sum() > 0:
        try:
            prophet_full = ts[["date", "flights"]].rename(columns={"date": "ds", "flights": "y"}).copy()
            prophet_full["ds"] = pd.to_datetime(prophet_full["ds"]).dt.tz_localize(None)

            prophet_model_full = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            prophet_model_full.fit(prophet_full)

            future_full = prophet_model_full.make_future_dataframe(periods=future_days)
            future_pred = prophet_model_full.predict(future_full)
            prophet_future_pred = future_pred.tail(future_days)["yhat"].values
        except Exception:
            prophet_future_pred = [np.nan] * future_days
    else:
        prophet_future_pred = [np.nan] * future_days

    future_df = pd.DataFrame({
        "Forecast Date": future_dates,
        "Linear Regression": lr_future_pred,
        "ARIMA": arima_future_pred,
        "Prophet": prophet_future_pred
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
    missing_percentage_df, quality_summary = get_quality_report(raw_df, df)
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
    - preprocessing and data overview
    - airline and destination analysis
    - domestic vs international route analysis
    - time-based traffic patterns
    - terminal utilization
    - short-term forecasting
    - business insights
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

    route_options = sorted(df["route_type"].dropna().unique())
    route_selected = st.multiselect(
        "Select Route Type",
        route_options,
        default=route_options
    )

    search_text = st.text_input("Search by Destination or Airline", "")

    st.markdown("---")
    st.markdown("## Model Availability")
    st.markdown(f"- ARIMA: {'✅ Available' if ARIMA_AVAILABLE else '❌ Not installed'}")
    st.markdown(f"- Prophet: {'✅ Available' if PROPHET_AVAILABLE else '❌ Not installed'}")


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

if "route_type" in df.columns and route_selected:
    mask &= df["route_type"].isin(route_selected)

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
        destination demand, terminal utilization, route balance, and short-term forecasting.
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
    .reindex(range(24), fill_value=0)
    .reset_index()
)
hour_df.columns = ["hour", "Number of Flights"]

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Overview",
        "Traffic Patterns",
        "Destinations & Airlines",
        "Terminal & Routes",
        "Forecasting",
        "Insights",
        "Data Overview"
    ]
)


# =========================================================
# TAB 1 OVERVIEW
# =========================================================
with tab1:
    left, right = st.columns([1.2, 0.8])

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
            height=430,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Date",
            yaxis_title="Number of Flights",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-head">Route Balance</div>', unsafe_allow_html=True)

        route_counts = filtered["route_type"].value_counts()
        if not route_counts.empty:
            fig_route = go.Figure(data=[go.Pie(
                labels=route_counts.index,
                values=route_counts.values,
                hole=0.55,
                pull=[0.04] * len(route_counts),
                textinfo="label+percent"
            )])

            fig_route.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=25, b=10),
                annotations=[dict(
                    text=f"Flights<br><b>{route_counts.sum():,}</b>",
                    x=0.5, y=0.5, font_size=18, showarrow=False
                )],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_route, use_container_width=True)

    st.markdown('<div class="section-head">Executive Snapshot</div>', unsafe_allow_html=True)

    avg_daily = daily_df["Number of Flights"].mean()
    median_daily = daily_df["Number of Flights"].median()

    domestic_count = int((filtered["route_type"] == "Domestic").sum()) if "route_type" in filtered.columns else 0
    international_count = int((filtered["route_type"] == "International").sum()) if "route_type" in filtered.columns else 0

    st.markdown(f"""
    <div class="glass">
        <div class="small-muted">
        The selected data shows an average of <b>{avg_daily:,.0f}</b> flights per day and a median of
        <b>{median_daily:,.0f}</b>. The route mix remains led by <b>{dominant_route}</b> traffic, with
        <b>{domestic_count:,}</b> domestic departures and <b>{international_count:,}</b> international departures.
        This helps position Riyadh Airport both as a strong domestic connector and a regional international hub.
        </div>
    </div>
    """, unsafe_allow_html=True)


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
        fig_hour.update_xaxes(dtick=1)
        st.plotly_chart(fig_hour, use_container_width=True)

    with c2:
        st.markdown('<div class="section-head">Flights by Day of Week</div>', unsafe_allow_html=True)

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df = filtered.groupby("day_of_week").size().reindex(day_order, fill_value=0).reset_index()
        dow_df.columns = ["day_of_week", "Number of Flights"]

        fig_dow = px.bar(
            dow_df,
            x="day_of_week",
            y="Number of Flights",
            text="Number of Flights"
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

    b1, b2 = st.columns([1.1, 0.9])

    with b1:
        st.markdown('<div class="section-head">Day of Week vs Hour Heatmap</div>', unsafe_allow_html=True)

        heatmap_data = (
            filtered.groupby(["day_of_week", "hour"])
            .size()
            .reset_index(name="flights")
            .pivot(index="day_of_week", columns="hour", values="flights")
        )
        heatmap_data = heatmap_data.reindex(index=day_order, columns=range(24), fill_value=0)

        fig_heat = px.imshow(
            heatmap_data,
            aspect="auto",
            labels=dict(x="Hour of Day", y="Day of Week", color="Flights"),
            text_auto=False
        )
        fig_heat.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with b2:
        st.markdown('<div class="section-head">Flights by Month</div>', unsafe_allow_html=True)

        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_df = filtered.groupby("month_name").size().reindex(month_order).dropna().reset_index()
        month_df.columns = ["month_name", "Number of Flights"]

        fig_month_pie = go.Figure(data=[go.Pie(
            labels=month_df["month_name"],
            values=month_df["Number of Flights"],
            textinfo="label+percent"
        )])
        fig_month_pie.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_month_pie, use_container_width=True)

    st.markdown('<div class="section-head">Monthly Trend Line</div>', unsafe_allow_html=True)

    month_num_df = (
        filtered.groupby(["month", "month_name"])
        .size()
        .reset_index(name="Number of Flights")
        .sort_values("month")
    )

    fig_month_line = px.line(
        month_num_df,
        x="month_name",
        y="Number of Flights",
        markers=True
    )
    fig_month_line.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Month",
        yaxis_title="Number of Flights",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_month_line, use_container_width=True)


# =========================================================
# TAB 3 DESTINATIONS & AIRLINES
# =========================================================
with tab3:
    d1, d2 = st.columns(2)

    top_international = (
        filtered[filtered["route_type"] == "International"]["destination_airport_name"]
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_international.columns = ["Destination Airport", "Number of Flights"]

    top_domestic = (
        filtered[filtered["route_type"] == "Domestic"]["destination_airport_name"]
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_domestic.columns = ["Destination Airport", "Number of Flights"]

    with d1:
        st.markdown('<div class="section-head">Top 10 International Destinations from Riyadh</div>', unsafe_allow_html=True)
        if not top_international.empty:
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
            st.info("No international destination data is available for the selected filters.")

    with d2:
        st.markdown('<div class="section-head">Top 10 Domestic Destinations from Riyadh</div>', unsafe_allow_html=True)
        if not top_domestic.empty:
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
            st.info("No domestic destination data is available for the selected filters.")

    st.markdown('<div class="section-head">Top Destination Airports Overall</div>', unsafe_allow_html=True)

    top_dest = (
        filtered["destination_airport_name"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_dest.columns = ["Destination Airport", "Number of Flights"]

    fig_dest = px.bar(
        top_dest,
        x="Destination Airport",
        y="Number of Flights",
        text="Number of Flights"
    )
    fig_dest.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Destination Airport",
        yaxis_title="Number of Flights",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig_dest.update_traces(textposition="outside")
    st.plotly_chart(fig_dest, use_container_width=True)

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

        fig_map.update_traces(marker=dict(opacity=0.82))
        fig_map.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Location Type"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<div class="section-head">Pareto Analysis of Airlines</div>', unsafe_allow_html=True)

    airline_df = (
        filtered["airline.name"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    airline_df.columns = ["Airline Name", "Number of Flights"]
    airline_df["Cumulative %"] = airline_df["Number of Flights"].cumsum() / airline_df["Number of Flights"].sum() * 100

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=airline_df["Airline Name"],
        y=airline_df["Number of Flights"],
        name="Number of Flights"
    ))
    fig_pareto.add_trace(go.Scatter(
        x=airline_df["Airline Name"],
        y=airline_df["Cumulative %"],
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in airline_df["Cumulative %"]],
        textposition="top center",
        name="Cumulative %",
        yaxis="y2"
    ))

    fig_pareto.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Airline",
        yaxis=dict(title="Number of Flights"),
        yaxis2=dict(
            title="Cumulative Percentage (%)",
            overlaying="y",
            side="right",
            range=[0, 105]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_pareto, use_container_width=True)


# =========================================================
# TAB 4 TERMINAL & ROUTES
# =========================================================
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-head">Flight Distribution by Terminal</div>', unsafe_allow_html=True)

        terminal_df = (
            filtered.groupby("movement.terminal")
            .size()
            .reset_index(name="Number of Flights")
            .sort_values("movement.terminal")
        )
        terminal_df["Terminal"] = terminal_df["movement.terminal"].apply(lambda x: f"Terminal {int(float(x))}")

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

    with c2:
        st.markdown('<div class="section-head">Domestic vs International by Terminal</div>', unsafe_allow_html=True)

        term_route_df = (
            filtered.groupby(["movement.terminal", "route_type"])
            .size()
            .reset_index(name="Number of Flights")
            .sort_values("movement.terminal")
        )
        term_route_df["Terminal"] = term_route_df["movement.terminal"].apply(lambda x: f"Terminal {int(float(x))}")

        fig_term_route = px.bar(
            term_route_df,
            x="Terminal",
            y="Number of Flights",
            color="route_type",
            barmode="stack"
        )
        fig_term_route.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Terminal",
            yaxis_title="Number of Flights",
            legend_title="Route Type",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_term_route, use_container_width=True)

    st.markdown('<div class="section-head">Route Share Summary</div>', unsafe_allow_html=True)

    route_counts = filtered["route_type"].value_counts().reset_index()
    route_counts.columns = ["Route Type", "Number of Flights"]
    route_counts["Share %"] = route_counts["Number of Flights"] / route_counts["Number of Flights"].sum() * 100

    st.dataframe(route_counts, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="glass">
    This section highlights how traffic is distributed across airport terminals and how each terminal contributes
    to domestic and international operations. It supports operational decisions related to gate allocation,
    terminal balancing, and congestion reduction.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 5 FORECASTING
# =========================================================
with tab5:
    st.markdown('<div class="section-head">Short-Term Forecasting</div>', unsafe_allow_html=True)

    fc = run_forecasting(daily_df)

    if not fc["ready"]:
        st.info("There are not enough daily observations to build the forecasting models.")
    else:
        f1, f2 = st.columns([1.15, 0.85])

        with f1:
            st.markdown("#### Train/Test Split with Predictions")

            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(
                x=fc["train"]["date"],
                y=fc["train"]["flights"],
                mode="lines",
                name="Training Data"
            ))
            fig_split.add_trace(go.Scatter(
                x=fc["test"]["date"],
                y=fc["test"]["flights"],
                mode="lines",
                name="Actual Test Data",
                line=dict(width=3)
            ))

            for model_name in ["Linear Regression", "ARIMA", "Prophet"]:
                if model_name in fc["test"].columns and fc["test"][model_name].notna().sum() > 0:
                    fig_split.add_trace(go.Scatter(
                        x=fc["test"]["date"],
                        y=fc["test"][model_name],
                        mode="lines",
                        name=model_name,
                        line=dict(dash="dash")
                    ))

            fig_split.update_layout(
                height=430,
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
                    height=340,
                    margin=dict(l=10, r=10, t=15, b=10),
                    xaxis_title="Forecasting Model",
                    yaxis_title="Error Value",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_err, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### RMSE Comparison")
            fig_rmse = px.bar(
                fc["results"],
                x="Model",
                y="RMSE",
                text="RMSE"
            )
            fig_rmse.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_rmse.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        with c4:
            st.markdown("#### MAE Comparison")
            fig_mae = px.bar(
                fc["results"],
                x="Model",
                y="MAE",
                text="MAE"
            )
            fig_mae.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_mae.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        st.markdown("#### Next 14-Day Forecast")

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=fc["ts"]["date"],
            y=fc["ts"]["flights"],
            mode="lines",
            name="Historical Flights",
            line=dict(width=3)
        ))

        for model_name in ["Linear Regression", "ARIMA", "Prophet"]:
            if model_name in fc["future"].columns and fc["future"][model_name].notna().sum() > 0:
                fig_future.add_trace(go.Scatter(
                    x=fc["future"]["Forecast Date"],
                    y=fc["future"][model_name],
                    mode="lines+markers",
                    name=f"{model_name} Forecast",
                    line=dict(dash="dash")
                ))

        fig_future.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Forecast Date",
            yaxis_title="Predicted Number of Flights",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_future, use_container_width=True)

        st.markdown("#### Actual vs Predicted Scatter Plots")

        model_cols = [m for m in ["Linear Regression", "ARIMA", "Prophet"] if m in fc["test"].columns and fc["test"][m].notna().sum() > 0]
        if model_cols:
            scatter_cols = st.columns(len(model_cols))
            for idx, model_name in enumerate(model_cols):
                with scatter_cols[idx]:
                    scatter_df = pd.DataFrame({
                        "Actual Flights": fc["test"]["flights"],
                        "Predicted Flights": fc["test"][model_name]
                    })

                    fig_scatter = px.scatter(
                        scatter_df,
                        x="Actual Flights",
                        y="Predicted Flights",
                        trendline=None
                    )

                    min_val = min(scatter_df["Actual Flights"].min(), scatter_df["Predicted Flights"].min())
                    max_val = max(scatter_df["Actual Flights"].max(), scatter_df["Predicted Flights"].max())

                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Perfect Prediction"
                    ))

                    fig_scatter.update_layout(
                        title=model_name,
                        height=340,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================
# TAB 6 INSIGHTS
# =========================================================
with tab6:
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

    peak_hour_value = hour_df.loc[hour_df["Number of Flights"].idxmax(), "hour"]
    peak_hour_count = hour_df["Number of Flights"].max()
    insight_box(
        "Peak Hour Pattern",
        f"The busiest operational hour is <b>{int(peak_hour_value)}:00</b>, with <b>{int(peak_hour_count):,}</b> departures in the current filtered dataset."
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
    <b>• Route balance:</b><br>
    Riyadh Airport supports both domestic and international traffic, showing its dual role as a national connector and a regional gateway.
    <br><br>
    <b>• Domestic dominance on key routes:</b><br>
    The Riyadh–Jeddah route remains one of the strongest domestic corridors, reflecting sustained demand between two major Saudi cities.
    <br><br>
    <b>• Regional connectivity:</b><br>
    International traffic is strongly concentrated in nearby regional hubs such as <b>Dubai, Cairo, and Istanbul</b>, highlighting Riyadh’s role in regional mobility.
    <br><br>
    <b>• Airline concentration:</b><br>
    A relatively small number of airlines account for a large portion of departures, indicating an operational structure dominated by major carriers.
    <br><br>
    <b>• Terminal utilization imbalance:</b><br>
    The analysis showed that <b>Terminal 5 handled the largest share of flights</b>, mainly because it served a substantial portion of domestic routes.
    <br><br>
    <b>• Peak operations by time:</b><br>
    Flight demand intensifies during certain hours of the day, especially evening activity, which suggests that staffing and gate planning should align with peak windows.
    <br><br>
    <b>• Operational adjustment:</b><br>
    A notable operational change occurred on <b>February 25, 2026</b>, when domestic flights were redistributed to <b>Terminals 3 and 4</b>. This reflects how airport analytics can support better terminal balancing and congestion reduction.
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
    This dashboard transforms Riyadh Airport departure data into a professional decision-support tool.
    It helps decision-makers understand route demand, airline concentration, terminal usage,
    operating peaks, and short-term traffic expectations in one interactive environment.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 7 DATA OVERVIEW
# =========================================================
with tab7:
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

    st.markdown('<div class="section-head">Initial Data Quality Report</div>', unsafe_allow_html=True)
    q1, q2 = st.columns([0.9, 1.1])

    with q1:
        quality_table = pd.DataFrame({
            "Metric": [
                "Raw Rows",
                "Raw Columns",
                "Processed Rows",
                "Processed Columns",
                "Columns > 50% Missing"
            ],
            "Value": [
                quality_summary["raw_rows"],
                quality_summary["raw_cols"],
                quality_summary["processed_rows"],
                quality_summary["processed_cols"],
                len(quality_summary["columns_over_50_missing"])
            ]
        })
        st.dataframe(quality_table, use_container_width=True, hide_index=True)

    with q2:
        st.markdown("##### Columns with More Than 50% Missing")
        if quality_summary["columns_over_50_missing"]:
            st.write(", ".join(quality_summary["columns_over_50_missing"]))
        else:
            st.write("No columns exceed 50% missingness.")

        st.markdown("##### Datetime-like Columns")
        st.write(", ".join(quality_summary["datetime_like_columns"]) if quality_summary["datetime_like_columns"] else "None")

    st.markdown('<div class="section-head">Missing Value Percentage by Column</div>', unsafe_allow_html=True)
    miss_top = missing_percentage_df.head(15)

    fig_missing = px.bar(
        miss_top,
        x="Column",
        y="Missing %",
        text="Missing %"
    )
    fig_missing.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_missing.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Column",
        yaxis_title="Missing Percentage",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_missing, use_container_width=True)

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