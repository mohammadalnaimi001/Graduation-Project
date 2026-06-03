import io
import json
import os
from typing import Any
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ─── HARDCODED API KEY ───────────────────────────────────────────────────────
GEMINI_API_KEY = "AQ.Ab8RN6KAVRoZH5vAN7W500F5OLBghMOn29RYdpDhXaagMWW2gg"
GEMINI_MODEL   = "gemini-2.0-flash"
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Intelligence Dataset",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 1.4rem; }
        .kpi-card {
            background: linear-gradient(135deg, rgba(38,39,57,0.95), rgba(24,25,37,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px; padding: 1rem; box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }
        .kpi-label { color: #9aa4bf; font-size: 0.85rem; margin-bottom: 0.25rem; }
        .kpi-value { color: #f2f5ff; font-size: 1.65rem; font-weight: 700; line-height: 1.1; }
        .kpi-help { color: #93a0bd; font-size: 0.76rem; margin-top: 0.3rem; }
        .section-title { font-size: 1.04rem; font-weight: 650; color: #e8ecf7; }

        /* ── Power BI Template Styles ── */
        .pbi-header {
            background: linear-gradient(90deg, #F2C811 0%, #E8A000 100%);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
        }
        .pbi-header h2 {
            color: #1a1a2e !important;
            margin: 0;
            font-size: 1.4rem;
            font-weight: 800;
        }
        .pbi-header p {
            color: #3a3a3a;
            margin: 0.2rem 0 0 0;
            font-size: 0.85rem;
        }
        .pbi-kpi-card {
            background: linear-gradient(135deg, #1e1e2e, #16213e);
            border: 1px solid #F2C811;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(242,200,17,0.15);
        }
        .pbi-kpi-label {
            color: #F2C811;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }
        .pbi-kpi-value {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1;
        }
        .pbi-kpi-sub {
            color: #9aa4bf;
            font-size: 0.72rem;
            margin-top: 0.3rem;
        }
        .pbi-section-title {
            color: #F2C811;
            font-size: 0.95rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border-bottom: 2px solid #F2C811;
            padding-bottom: 0.3rem;
            margin-bottom: 0.8rem;
        }
        .pbi-filter-box {
            background: #1a1a2e;
            border: 1px solid rgba(242,200,17,0.3);
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
        }
        .pbi-insight-box {
            background: linear-gradient(135deg, #1e1e2e, #0f3460);
            border-left: 4px solid #F2C811;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
        }
        .pbi-insight-text {
            color: #e2e8f0;
            font-size: 0.88rem;
            margin: 0;
        }
        .pbi-footer {
            background: #1a1a2e;
            border-top: 2px solid #F2C811;
            border-radius: 0 0 12px 12px;
            padding: 0.6rem 1rem;
            text-align: center;
            color: #9aa4bf;
            font-size: 0.75rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    name = file_name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buffer)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(buffer)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def infer_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue
        if series.dtype == "object":
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.6:
                date_cols.append(col)
    return date_cols


def detect_business_columns(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)

    def pick(candidates: list[str], numeric: bool | None = None) -> str | None:
        for keyword in candidates:
            for col in columns:
                if keyword in col.lower():
                    if numeric is True and not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    return col
        return None

    date_col = pick(["date", "day", "booking", "checkin", "timestamp", "created"])
    if date_col is None:
        guessed_dates = infer_date_columns(df)
        date_col = guessed_dates[0] if guessed_dates else None

    return {
        "price_col": pick(["price", "rate", "cost", "amount"], numeric=True),
        "rating_col": pick(["rating", "score", "review"], numeric=True),
        "city_col": pick(["city", "location", "district", "area", "country", "region"]),
        "hotel_col": pick(["hotel", "property", "name"]),
        "date_col": date_col,
    }


def get_stats(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def auto_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = get_stats(df)
    cleaned = df.copy().drop_duplicates()
    num_cols = infer_numeric_columns(cleaned)
    cat_cols = infer_categorical_columns(cleaned)

    for col in num_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in cat_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            cleaned[col] = cleaned[col].fillna("Unknown" if mode.empty else mode.iloc[0])

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    after = get_stats(cleaned)
    report = {
        "rows_before": before["rows"],
        "rows_after": after["rows"],
        "missing_before": before["missing"],
        "missing_after": after["missing"],
        "duplicates_before": before["duplicates"],
        "duplicates_after": after["duplicates"],
    }
    return cleaned, report


def manual_fill_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                if strategy == "mean":
                    out[col] = out[col].fillna(out[col].mean())
                elif strategy == "median":
                    out[col] = out[col].fillna(out[col].median())
                else:
                    mode = out[col].mode(dropna=True)
                    out[col] = out[col].fillna(out[col].median() if mode.empty else mode.iloc[0])
            else:
                mode = out[col].mode(dropna=True)
                out[col] = out[col].fillna("Unknown" if mode.empty else mode.iloc[0])
    return out


def remove_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()


def make_kpi_card(title: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_pbi_kpi_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="pbi-kpi-card">
            <div class="pbi-kpi-label">{label}</div>
            <div class="pbi-kpi-value">{value}</div>
            <div class="pbi-kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_dataset_kpis(df: pd.DataFrame, detected: dict[str, str | None]) -> dict[str, Any]:
    price = detected["price_col"]
    rating = detected["rating_col"]
    hotel = detected["hotel_col"]
    return {
        "total_hotels": int(df[hotel].nunique()) if hotel and hotel in df.columns else int(df.shape[0]),
        "avg_price": float(df[price].mean()) if price and price in df.columns and pd.api.types.is_numeric_dtype(df[price]) else np.nan,
        "avg_rating": float(df[rating].mean()) if rating and rating in df.columns and pd.api.types.is_numeric_dtype(df[rating]) else np.nan,
    }


def power_bi_suggestions(df: pd.DataFrame, detected: dict[str, str | None]) -> dict[str, Any]:
    numeric = infer_numeric_columns(df)
    categorical = infer_categorical_columns(df)
    dates = infer_date_columns(df)
    has_geo = detected["city_col"] is not None or any(
        k in " ".join(df.columns).lower() for k in ["city", "country", "region", "location", "lat", "lon"]
    )

    dashboard_type = "Performance Dashboard"
    if dates and numeric:
        dashboard_type = "Trend & Performance Dashboard"
    if has_geo and numeric:
        dashboard_type = "Geographic Performance Dashboard"
    if len(categorical) >= 2 and numeric:
        dashboard_type = "Operational Comparison Dashboard"

    kpi_suggestions = []
    for col in numeric[:5]:
        kpi_suggestions.append(f"Average {col}")
        kpi_suggestions.append(f"Total {col}")
    if detected["hotel_col"]:
        kpi_suggestions.append(f"Distinct {detected['hotel_col']}")
    if detected["rating_col"]:
        kpi_suggestions.append(f"Average {detected['rating_col']}")

    dax = []
    for col in numeric[:3]:
        dax.append(f"Total {col} = SUM('Table'[{col}])")
        dax.append(f"Avg {col} = AVERAGE('Table'[{col}])")
    if dates and numeric:
        dax.append(
            f"MoM Change = DIVIDE([Total {numeric[0]}] - CALCULATE([Total {numeric[0]}], "
            f"DATEADD('Date'[Date], -1, MONTH)), CALCULATE([Total {numeric[0]}], DATEADD('Date'[Date], -1, MONTH)))"
        )

    visuals = ["Bar chart", "Line chart", "KPI cards"]
    if has_geo:
        visuals.append("Map")
    if len(numeric) >= 2:
        visuals.append("Scatter plot")

    return {
        "dashboard_type": dashboard_type,
        "kpis": sorted(set(kpi_suggestions))[:10],
        "dax": dax[:8],
        "visuals": visuals,
        "notes": [
            f"Detected numeric columns: {len(numeric)}",
            f"Detected categorical columns: {len(categorical)}",
            f"Detected date columns: {len(dates)}",
        ],
    }


def apply_filters(df: pd.DataFrame, detected: dict[str, str | None]) -> pd.DataFrame:
    out = df.copy()
    city_col = detected["city_col"]
    price_col = detected["price_col"]
    rating_col = detected["rating_col"]

    c1, c2, c3 = st.columns(3)
    with c1:
        if city_col and city_col in out.columns:
            cities = sorted(out[city_col].dropna().astype(str).unique().tolist())
            selected = st.multiselect(
                "City / Location",
                options=cities,
                default=cities[: min(8, len(cities))],
                key=f"city_filter_{city_col}",
            )
            if selected:
                out = out[out[city_col].astype(str).isin(selected)]
        else:
            st.caption("No city/location column detected.")
    with c2:
        if price_col and price_col in out.columns and pd.api.types.is_numeric_dtype(out[price_col]):
            p_min, p_max = float(out[price_col].min()), float(out[price_col].max())
            if p_min < p_max:
                rng = st.slider("Price range", p_min, p_max, (p_min, p_max), key=f"price_filter_{price_col}")
                out = out[(out[price_col] >= rng[0]) & (out[price_col] <= rng[1])]
            else:
                st.caption("Price has one unique value.")
        else:
            st.caption("No numeric price column detected.")
    with c3:
        if rating_col and rating_col in out.columns and pd.api.types.is_numeric_dtype(out[rating_col]):
            r_min, r_max = float(out[rating_col].min()), float(out[rating_col].max())
            if r_min < r_max:
                rr = st.slider("Rating range", r_min, r_max, (r_min, r_max), key=f"rating_filter_{rating_col}")
                out = out[(out[rating_col] >= rr[0]) & (out[rating_col] <= rr[1])]
            else:
                st.caption("Rating has one unique value.")
        else:
            st.caption("No numeric rating column detected.")
    return out


def init_state() -> None:
    defaults = {
        "datasets": {},
        "selected_dataset": None,
        "cleaning_reports": {},
        "chat_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

# ─── Configure Gemini once at startup ────────────────────────────────────────
_gemini_model = None
if genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as _cfg_err:
        st.error(f"Gemini config error: {_cfg_err}")
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 AI Intelligence Dataset")
st.caption("Upload, clean, compare, visualize, and generate AI insights from multiple datasets.")

with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader(
        "Upload one or more datasets",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            try:
                loaded = load_data(file.name, file.getvalue())
                cleaned, report = auto_clean_data(loaded)
                st.session_state.datasets[file.name] = {
                    "raw": loaded,
                    "cleaned": cleaned,
                    "detected": detect_business_columns(cleaned),
                }
                st.session_state.cleaning_reports[file.name] = report
            except Exception as exc:
                st.error(f"{file.name}: {exc}")

    names = list(st.session_state.datasets.keys())
    if names:
        default_idx = (
            names.index(st.session_state.selected_dataset)
            if st.session_state.selected_dataset in names
            else 0
        )
        st.session_state.selected_dataset = st.selectbox("Active dataset", options=names, index=default_idx)
        selected_for_compare = st.multiselect(
            "Datasets for comparison", options=names, default=names[: min(2, len(names))]
        )
    else:
        st.session_state.selected_dataset = None
        selected_for_compare = []

if not st.session_state.datasets:
    st.info("Upload one or more CSV/Excel files from the sidebar to begin.")
    st.stop()

active_name = st.session_state.selected_dataset or list(st.session_state.datasets.keys())[0]
active_entry = st.session_state.datasets[active_name]
active_df = active_entry["cleaned"]
detected = active_entry["detected"]
numeric_cols = infer_numeric_columns(active_df)

tabs = st.tabs(["Overview", "Data Cleaning", "Comparison", "AI Insights", "Power BI Suggestions", "Power BI Dashboard"])

# ══════════════════════════════ TAB 0 – OVERVIEW ══════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">📊 Overview Dashboard</div>', unsafe_allow_html=True)
    st.write(f"Active dataset: `{active_name}`")
    filtered_df = apply_filters(active_df, detected)

    kpis = get_dataset_kpis(filtered_df, detected)
    k1, k2, k3 = st.columns(3)
    with k1:
        make_kpi_card("🏨 Total Hotels / Records", f"{kpis['total_hotels']:,}", "Unique hotels if detected, otherwise row count.")
    with k2:
        make_kpi_card("💰 Avg Price", "N/A" if np.isnan(kpis["avg_price"]) else f"{kpis['avg_price']:,.2f}", "Current filtered selection.")
    with k3:
        make_kpi_card("⭐ Avg Rating", "N/A" if np.isnan(kpis["avg_rating"]) else f"{kpis['avg_rating']:.2f}", "Current filtered selection.")

    st.markdown('<div class="section-title">🗂️ Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(40), use_container_width=True)

    st.markdown('<div class="section-title">📈 Interactive Visuals</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    price_col  = detected["price_col"]
    rating_col = detected["rating_col"]
    city_col   = detected["city_col"]
    date_col   = detected["date_col"]

    with c1:
        if price_col and price_col in filtered_df.columns:
            fig_hist = px.histogram(filtered_df, x=price_col, nbins=35, title=f"Histogram: {price_col}", template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
        if city_col and price_col and city_col in filtered_df.columns and price_col in filtered_df.columns:
            grp = (
                filtered_df.groupby(city_col)[price_col]
                .mean()
                .reset_index()
                .sort_values(price_col, ascending=False)
                .head(20)
            )
            fig_bar = px.bar(grp, x=city_col, y=price_col, color=price_col, template="plotly_dark", title=f"Avg {price_col} by {city_col}")
            st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        if price_col and rating_col and price_col in filtered_df.columns and rating_col in filtered_df.columns:
            fig_sc = px.scatter(
                filtered_df, x=price_col, y=rating_col,
                color=city_col if city_col and city_col in filtered_df.columns else None,
                template="plotly_dark", title=f"{price_col} vs {rating_col}",
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        if city_col and rating_col and city_col in filtered_df.columns and rating_col in filtered_df.columns:
            fig_box = px.box(filtered_df, x=city_col, y=rating_col, template="plotly_dark",
                             title=f"Box: {rating_col} by {city_col}", points="outliers")
            st.plotly_chart(fig_box, use_container_width=True)

    if len(numeric_cols) >= 2:
        corr = filtered_df[numeric_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap", template="plotly_dark", color_continuous_scale="RdBu")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Linear Regression ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 Linear Regression</div>', unsafe_allow_html=True)
    if len(numeric_cols) >= 2:
        lr_c1, lr_c2 = st.columns(2)
        with lr_c1:
            lr_x = st.selectbox("X — Independent Variable", options=numeric_cols, key="lr_x")
        with lr_c2:
            lr_y_opts = [c for c in numeric_cols if c != lr_x]
            lr_y = st.selectbox("Y — Dependent Variable", options=lr_y_opts, key="lr_y")

        lr_data = filtered_df[[lr_x, lr_y]].dropna()
        if len(lr_data) >= 2:
            x_vals = lr_data[lr_x].values.astype(float)
            y_vals = lr_data[lr_y].values.astype(float)

            coeffs = np.polyfit(x_vals, y_vals, 1)
            slope, intercept = coeffs
            y_pred = np.polyval(coeffs, x_vals)
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

            rm1, rm2, rm3 = st.columns(3)
            rm1.metric("R² Score", f"{r_squared:.4f}")
            rm2.metric("Slope", f"{slope:.4f}")
            rm3.metric("Intercept", f"{intercept:.4f}")
            st.caption(f"Equation: **{lr_y} = {slope:.4f} × {lr_x} + ({intercept:.4f})**")

            x_line = np.linspace(x_vals.min(), x_vals.max(), 300)
            y_line = slope * x_line + intercept
            fig_lr = px.scatter(lr_data, x=lr_x, y=lr_y, template="plotly_dark",
                                title=f"Linear Regression: {lr_x} → {lr_y}", opacity=0.55)
            fig_lr.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                name=f"Fit (R²={r_squared:.3f})",
                line=dict(color="#F2C811", width=2.5),
            ))
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.warning("Not enough data points for regression after removing nulls.")
    else:
        st.info("Need at least 2 numeric columns for linear regression.")

    if date_col and date_col in filtered_df.columns and price_col and price_col in filtered_df.columns:
        ts_df = filtered_df.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna(subset=[date_col]).sort_values(date_col)
        if not ts_df.empty:
            ts = ts_df.groupby(date_col)[price_col].mean().reset_index()
            fig_ts = px.line(ts, x=date_col, y=price_col, markers=True, template="plotly_dark", title=f"Time Series: {price_col}")
            st.plotly_chart(fig_ts, use_container_width=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    chart_type = chart_col1.selectbox("Chart Type", ["Histogram", "Scatter", "Box", "Bar"])
    x_axis = chart_col2.selectbox("X-axis", options=filtered_df.columns.tolist())
    y_axis = chart_col3.selectbox("Y-axis", options=["(None)"] + filtered_df.columns.tolist())
    color_axis = st.selectbox("Color (optional)", options=["(None)"] + filtered_df.columns.tolist())
    color_arg = None if color_axis == "(None)" else color_axis
    y_arg = None if y_axis == "(None)" else y_axis

    try:
        if chart_type == "Histogram":
            fig_dynamic = px.histogram(filtered_df, x=x_axis, color=color_arg, template="plotly_dark")
        elif chart_type == "Scatter" and y_arg:
            fig_dynamic = px.scatter(filtered_df, x=x_axis, y=y_arg, color=color_arg, template="plotly_dark")
        elif chart_type == "Box" and y_arg:
            fig_dynamic = px.box(filtered_df, x=x_axis, y=y_arg, color=color_arg, template="plotly_dark")
        elif chart_type == "Bar" and y_arg:
            fig_dynamic = px.bar(filtered_df, x=x_axis, y=y_arg, color=color_arg, template="plotly_dark")
        else:
            fig_dynamic = None
            st.warning(f"{chart_type} requires a Y-axis.")
        if fig_dynamic is not None:
            st.plotly_chart(fig_dynamic, use_container_width=True)
    except Exception as err:
        st.error(f"Could not render chart: {err}")


# ══════════════════════════════ TAB 1 – DATA CLEANING ═════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">🧹 Data Cleaning</div>', unsafe_allow_html=True)
    st.write(f"Cleaning dataset: `{active_name}`")

    before = get_stats(active_entry["raw"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", before["rows"])
    c2.metric("Columns", before["cols"])
    c3.metric("Missing Values", before["missing"])
    c4.metric("Duplicates", before["duplicates"])

    btn1, btn2, btn3 = st.columns(3)
    if btn1.button("Clean Data", use_container_width=True):
        cleaned, report = auto_clean_data(active_entry["raw"])
        st.session_state.datasets[active_name]["cleaned"] = cleaned
        st.session_state.datasets[active_name]["detected"] = detect_business_columns(cleaned)
        st.session_state.cleaning_reports[active_name] = report
        st.success("Automatic cleaning applied.")
        st.rerun()

    if btn2.button("Remove Missing Values", use_container_width=True):
        cleaned = remove_missing_rows(active_entry["raw"])
        st.session_state.datasets[active_name]["cleaned"] = cleaned
        st.session_state.datasets[active_name]["detected"] = detect_business_columns(cleaned)
        st.success("Rows with missing values removed.")
        st.rerun()

    fill_strategy = btn3.selectbox("Fill missing values", options=["mean", "median", "mode"], key=f"fill_strategy_{active_name}")
    if st.button("Apply Fill Strategy", use_container_width=True):
        cleaned = manual_fill_missing(active_entry["cleaned"], fill_strategy)
        st.session_state.datasets[active_name]["cleaned"] = cleaned
        st.session_state.datasets[active_name]["detected"] = detect_business_columns(cleaned)
        st.success(f"Missing values filled using `{fill_strategy}` strategy.")
        st.rerun()

    after_df = st.session_state.datasets[active_name]["cleaned"]
    after = get_stats(after_df)

    st.markdown("#### After Cleaning")
    rows_removed = before["rows"] - after["rows"]
    missing_removed = before["missing"] - after["missing"]
    dups_removed = before["duplicates"] - after["duplicates"]

    if rows_removed > 0:
        st.success(f"✅ **{after['rows']:,} rows remaining** — {rows_removed:,} rows removed during cleaning.")
    elif rows_removed == 0:
        st.info(f"ℹ️ **{after['rows']:,} rows remaining** — no rows were removed (missing values were filled).")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Rows After", f"{after['rows']:,}", delta=f"-{rows_removed:,} rows" if rows_removed > 0 else "No change", delta_color="off")
    a2.metric("Missing After", after["missing"], delta=-missing_removed, delta_color="inverse")
    a3.metric("Duplicates After", after["duplicates"], delta=-dups_removed, delta_color="inverse")
    a4.metric("Columns After", after["cols"])

    st.download_button(
        "⬇️ Export Cleaned Dataset (CSV)",
        data=after_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cleaned_{active_name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
    )
    st.caption("Exported file is ready for Power BI ingestion.")

# ══════════════════════════════ TAB 2 – COMPARISON ════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">⚖️ Dataset Comparison Dashboard</div>', unsafe_allow_html=True)
    if len(selected_for_compare) < 2:
        st.info("Select at least 2 datasets from sidebar to compare.")
    else:
        chosen = selected_for_compare[:2]
        dfa = st.session_state.datasets[chosen[0]]["cleaned"]
        dfb = st.session_state.datasets[chosen[1]]["cleaned"]
        det_a = st.session_state.datasets[chosen[0]]["detected"]
        det_b = st.session_state.datasets[chosen[1]]["detected"]

        ka = get_dataset_kpis(dfa, det_a)
        kb = get_dataset_kpis(dfb, det_b)
        comp_kpi = pd.DataFrame(
            [
                {"Dataset": chosen[0], "Total Hotels/Rows": ka["total_hotels"], "Avg Price": ka["avg_price"], "Avg Rating": ka["avg_rating"]},
                {"Dataset": chosen[1], "Total Hotels/Rows": kb["total_hotels"], "Avg Price": kb["avg_price"], "Avg Rating": kb["avg_rating"]},
            ]
        )
        st.dataframe(comp_kpi, use_container_width=True)

        common_numeric = sorted(set(infer_numeric_columns(dfa)).intersection(set(infer_numeric_columns(dfb))))
        if common_numeric:
            metric = st.selectbox("Metric for comparison", options=common_numeric)
            summary = pd.DataFrame(
                [
                    {"dataset": chosen[0], "mean": dfa[metric].mean(), "sum": dfa[metric].sum(), "count": dfa[metric].count()},
                    {"dataset": chosen[1], "mean": dfb[metric].mean(), "sum": dfb[metric].sum(), "count": dfb[metric].count()},
                ]
            )
            st.markdown("#### Mean / Sum / Count Differences")
            st.dataframe(summary, use_container_width=True)
            diff = summary.iloc[0][["mean", "sum", "count"]] - summary.iloc[1][["mean", "sum", "count"]]
            st.write({"difference_first_minus_second": diff.to_dict()})

            comb = pd.concat(
                [dfa[[metric]].assign(dataset=chosen[0]), dfb[[metric]].assign(dataset=chosen[1])],
                ignore_index=True,
            )
            fig_dist = px.histogram(comb, x=metric, color="dataset", barmode="overlay", opacity=0.65,
                                    template="plotly_dark", title=f"Distribution Comparison: {metric}")
            st.plotly_chart(fig_dist, use_container_width=True)

            fig_mean = px.bar(summary, x="dataset", y="mean", color="dataset", template="plotly_dark",
                              title=f"Average {metric} by Dataset")
            st.plotly_chart(fig_mean, use_container_width=True)
        else:
            st.warning("No common numeric columns found between selected datasets.")

# ══════════════════════════════ TAB 3 – AI INSIGHTS ══════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🤖 AI Insights (Gemini)</div>', unsafe_allow_html=True)
    st.write(f"**Analyze dataset:** `{active_name}`")

    if genai is None:
        st.error("❌ google-generativeai not installed. Run: pip install google-generativeai")
        st.stop()

    if _gemini_model is None:
        st.error("❌ Gemini model could not be initialised. Check your API key.")
        st.stop()

    sample_data = active_df.head(15).to_string()

    auto_prompt = f"""You are a professional business analyst.

Analyze this dataset and provide:
- Key insights
- Trends
- Problems
- Recommendations

Data:
{sample_data}
"""

    auto_insights_key = f"auto_insights_{active_name}"

    if auto_insights_key not in st.session_state:
        with st.spinner("Generating AI insights…"):
            try:
                auto_response = _gemini_model.generate_content(auto_prompt)
                st.session_state[auto_insights_key] = auto_response.text
            except Exception as e:
                st.session_state[auto_insights_key] = f"__ERROR__: {e}"

    st.subheader("📊 Automatic Analysis")
    cached = st.session_state[auto_insights_key]
    if cached.startswith("__ERROR__:"):
        st.error(cached.replace("__ERROR__: ", ""))
    else:
        st.success(cached)

    if st.button("🔄 Refresh Analysis"):
        if auto_insights_key in st.session_state:
            del st.session_state[auto_insights_key]
        st.rerun()

    st.subheader("💬 Chat with Data")

    user_question = st.text_input("Ask anything about your data:", key="chat_input")

    if st.button("Send", key="chat_send") and user_question.strip():
        st.session_state.chat_history.append(("You", user_question.strip()))

        chat_prompt = f"""Dataset:
{sample_data}

Question:
{user_question}

Answer clearly with insights.
"""
        with st.spinner("Thinking…"):
            try:
                response = _gemini_model.generate_content(chat_prompt)
                st.session_state.chat_history.append(("AI", response.text))
            except Exception as e:
                st.session_state.chat_history.append(("AI", f"Error: {e}"))

    for role, msg in reversed(st.session_state.chat_history):
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")

    st.subheader("📄 Generate Report")

    if st.button("Generate Full Report PDF"):
        insights_text = st.session_state.get(auto_insights_key, "No insights generated yet.")
        if insights_text.startswith("__ERROR__:"):
            st.error("Cannot generate PDF — AI insights not available.")
        else:
            try:
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer)
                styles = getSampleStyleSheet()
                elements = []

                elements.append(Paragraph("AI Business Report", styles["Title"]))
                elements.append(Spacer(1, 20))
                elements.append(Paragraph("Dataset Summary:", styles["Heading2"]))

                summary_text = active_df.describe().to_string().replace("<", "&lt;").replace(">", "&gt;")
                elements.append(Paragraph(f"<pre>{summary_text}</pre>", styles["Code"]))
                elements.append(Spacer(1, 20))

                elements.append(Paragraph("AI Insights:", styles["Heading2"]))
                safe_insights = insights_text.replace("<", "&lt;").replace(">", "&gt;")
                elements.append(Paragraph(safe_insights, styles["Normal"]))

                doc.build(elements)

                st.download_button(
                    label="📥 Download Report",
                    data=pdf_buffer.getvalue(),
                    file_name="AI_Report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")

# ══════════════════════════════ TAB 4 – POWER BI SUGGESTIONS ═════════════════
with tabs[4]:
    st.markdown('<div class="section-title">📊 Suggested Power BI Dashboard</div>', unsafe_allow_html=True)
    suggestion = power_bi_suggestions(active_df, detected)
    st.write(f"Recommended dashboard type: **{suggestion['dashboard_type']}**")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Suggested KPIs")
        for item in suggestion["kpis"]:
            st.markdown(f"- {item}")
        st.markdown("#### Suggested Visuals")
        for item in suggestion["visuals"]:
            st.markdown(f"- {item}")
    with p2:
        st.markdown("#### Suggested DAX Measures")
        for item in suggestion["dax"]:
            st.code(item, language="DAX")
        st.markdown("#### Dataset Notes")
        for note in suggestion["notes"]:
            st.markdown(f"- {note}")

    # ── Auto Generated Dashboard (kept only in Tab 4) ─────────────────────
    st.markdown("---")
    st.subheader("📊 Auto Generated Dashboard")

    price_col  = detected["price_col"]
    rating_col = detected["rating_col"]
    city_col   = detected["city_col"]
    date_col   = detected["date_col"]

    col1, col2, col3 = st.columns(3)
    with col1:
        if price_col:
            st.metric("Avg Price", f"{active_df[price_col].mean():.2f}")
    with col2:
        if rating_col:
            st.metric("Avg Rating", f"{active_df[rating_col].mean():.2f}")
    with col3:
        st.metric("Total Records", len(active_df))

    c1, c2 = st.columns(2)
    with c1:
        if city_col and price_col:
            fig_bar = px.bar(
                active_df.groupby(city_col)[price_col].mean().reset_index(),
                x=city_col, y=price_col,
                title="Average Price by City",
                template="plotly_dark",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        if date_col and price_col:
            df_temp = active_df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
            df_temp = df_temp.dropna(subset=[date_col])
            if not df_temp.empty:
                fig_line = px.line(
                    df_temp.groupby(date_col)[price_col].mean().reset_index(),
                    x=date_col, y=price_col,
                    title="Trend Over Time",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_line, use_container_width=True)

    if price_col and rating_col:
        fig_scatter = px.scatter(
            active_df, x=price_col, y=rating_col,
            title="Price vs Rating",
            template="plotly_dark",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    if city_col:
        fig_pie = px.pie(active_df, names=city_col, title="Distribution by City")
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════ TAB 5 – POWER BI DASHBOARD TEMPLATE ══════════
with tabs[5]:

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="pbi-header">
            <h2>⚡ Power BI Style Dashboard</h2>
            <p>Dataset: {active_name} &nbsp;|&nbsp; Microsoft-Inspired Template &nbsp;|&nbsp; Live Data</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    price_col  = detected["price_col"]
    rating_col = detected["rating_col"]
    city_col   = detected["city_col"]
    hotel_col  = detected["hotel_col"]
    date_col   = detected["date_col"]
    num_cols   = infer_numeric_columns(active_df)
    cat_cols   = infer_categorical_columns(active_df)

    PBI_TEMPLATE = "plotly_dark"
    PBI_COLOR    = "#F2C811"
    PBI_PALETTE  = ["#F2C811", "#00B0F0", "#00CC6A", "#FF4444", "#A259FF", "#FF8C00"]

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    st.markdown('<div class="pbi-section-title">🔍 Filters</div>', unsafe_allow_html=True)
    pbi_df = active_df.copy()

    f1, f2, f3 = st.columns(3)
    with f1:
        if city_col and city_col in pbi_df.columns:
            all_cities = sorted(pbi_df[city_col].dropna().astype(str).unique().tolist())
            sel_cities = st.multiselect("📍 City / Region", options=all_cities,
                                        default=all_cities[:min(6, len(all_cities))],
                                        key="pbi_city_filter")
            if sel_cities:
                pbi_df = pbi_df[pbi_df[city_col].astype(str).isin(sel_cities)]
        else:
            st.caption("No city column.")

    with f2:
        if price_col and price_col in pbi_df.columns and pd.api.types.is_numeric_dtype(pbi_df[price_col]):
            p_min, p_max = float(active_df[price_col].min()), float(active_df[price_col].max())
            if p_min < p_max:
                p_rng = st.slider("💰 Price Range", p_min, p_max, (p_min, p_max), key="pbi_price_filter")
                pbi_df = pbi_df[(pbi_df[price_col] >= p_rng[0]) & (pbi_df[price_col] <= p_rng[1])]
        else:
            st.caption("No price column.")

    with f3:
        if rating_col and rating_col in pbi_df.columns and pd.api.types.is_numeric_dtype(pbi_df[rating_col]):
            r_min, r_max = float(active_df[rating_col].min()), float(active_df[rating_col].max())
            if r_min < r_max:
                r_rng = st.slider("⭐ Rating Range", r_min, r_max, (r_min, r_max), key="pbi_rating_filter")
                pbi_df = pbi_df[(pbi_df[rating_col] >= r_rng[0]) & (pbi_df[rating_col] <= r_rng[1])]
        else:
            st.caption("No rating column.")

    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="pbi-section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

    kpi_cols = st.columns(5)
    kpi_data: list[tuple[str, str, str]] = []

    kpi_data.append(("📦 Total Records", f"{len(pbi_df):,}", f"of {len(active_df):,} total"))

    if price_col and price_col in pbi_df.columns and pd.api.types.is_numeric_dtype(pbi_df[price_col]):
        avg_p = pbi_df[price_col].mean()
        total_p = pbi_df[price_col].sum()
        kpi_data.append(("💰 Avg Price", f"{avg_p:,.2f}", f"Total: {total_p:,.0f}"))

    if rating_col and rating_col in pbi_df.columns and pd.api.types.is_numeric_dtype(pbi_df[rating_col]):
        avg_r = pbi_df[rating_col].mean()
        kpi_data.append(("⭐ Avg Rating", f"{avg_r:.2f}", f"Max: {pbi_df[rating_col].max():.2f}"))

    if hotel_col and hotel_col in pbi_df.columns:
        unique_h = pbi_df[hotel_col].nunique()
        kpi_data.append(("🏨 Unique Hotels", f"{unique_h:,}", "distinct entries"))

    if city_col and city_col in pbi_df.columns:
        unique_c = pbi_df[city_col].nunique()
        kpi_data.append(("📍 Cities / Regions", f"{unique_c:,}", "in current filter"))

    # fill up to 5 KPIs with extra numeric stats
    for col in num_cols:
        if col in [price_col, rating_col]:
            continue
        if len(kpi_data) >= 5:
            break
        kpi_data.append((f"📊 Avg {col[:14]}", f"{pbi_df[col].mean():,.2f}", f"Σ {pbi_df[col].sum():,.0f}"))

    for i, (label, value, sub) in enumerate(kpi_data[:5]):
        with kpi_cols[i]:
            make_pbi_kpi_card(label, value, sub)

    st.markdown("---")

    # ── Row 1: Bar + Line ─────────────────────────────────────────────────────
    st.markdown('<div class="pbi-section-title">📊 Performance by Category & Time Trend</div>', unsafe_allow_html=True)
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        if city_col and price_col and city_col in pbi_df.columns and price_col in pbi_df.columns:
            grp_city = (
                pbi_df.groupby(city_col)[price_col]
                .mean().reset_index()
                .sort_values(price_col, ascending=False)
                .head(15)
            )
            fig_pbi_bar = px.bar(
                grp_city, x=city_col, y=price_col,
                color=price_col,
                color_continuous_scale=["#1a1a2e", PBI_COLOR],
                title=f"Avg {price_col} by {city_col}",
                template=PBI_TEMPLATE,
            )
            fig_pbi_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color=PBI_COLOR,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_pbi_bar, use_container_width=True)
        elif num_cols and cat_cols:
            fallback_grp = pbi_df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index().head(15)
            fig_fb = px.bar(fallback_grp, x=cat_cols[0], y=num_cols[0],
                            template=PBI_TEMPLATE, title=f"Avg {num_cols[0]} by {cat_cols[0]}")
            fig_fb.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e2e8f0", title_font_color=PBI_COLOR)
            st.plotly_chart(fig_fb, use_container_width=True)
        else:
            st.info("Not enough columns for bar chart.")

    with row1_c2:
        if date_col and price_col and date_col in pbi_df.columns and price_col in pbi_df.columns:
            ts_pbi = pbi_df.copy()
            ts_pbi[date_col] = pd.to_datetime(ts_pbi[date_col], errors="coerce")
            ts_pbi = ts_pbi.dropna(subset=[date_col]).sort_values(date_col)
            if not ts_pbi.empty:
                ts_line = ts_pbi.groupby(date_col)[price_col].mean().reset_index()
                fig_pbi_line = px.line(
                    ts_line, x=date_col, y=price_col,
                    markers=True,
                    title=f"{price_col} Trend Over Time",
                    template=PBI_TEMPLATE,
                    color_discrete_sequence=[PBI_COLOR],
                )
                fig_pbi_line.update_traces(line_width=2.5)
                fig_pbi_line.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    title_font_color=PBI_COLOR,
                )
                st.plotly_chart(fig_pbi_line, use_container_width=True)
            else:
                st.info("No valid date data available.")
        else:
            if len(num_cols) >= 2:
                fig_area = px.area(
                    pbi_df.reset_index(), x=pbi_df.reset_index().index,
                    y=num_cols[0],
                    title=f"{num_cols[0]} Distribution",
                    template=PBI_TEMPLATE,
                    color_discrete_sequence=[PBI_COLOR],
                )
                fig_area.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                        font_color="#e2e8f0", title_font_color=PBI_COLOR)
                st.plotly_chart(fig_area, use_container_width=True)
            else:
                st.info("No date column detected for trend chart.")

    st.markdown('<div class="pbi-section-title">🔗 Correlation & Distribution</div>', unsafe_allow_html=True)
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        if price_col and rating_col and price_col in pbi_df.columns and rating_col in pbi_df.columns:
            fig_pbi_sc = px.scatter(
                pbi_df, x=price_col, y=rating_col,
                color=city_col if city_col and city_col in pbi_df.columns else None,
                color_discrete_sequence=PBI_PALETTE,
                title=f"{price_col} vs {rating_col}",
                template=PBI_TEMPLATE,
                opacity=0.75,
            )
            fig_pbi_sc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color=PBI_COLOR,
            )
            st.plotly_chart(fig_pbi_sc, use_container_width=True)
        elif len(num_cols) >= 2:
            fig_sc2 = px.scatter(pbi_df, x=num_cols[0], y=num_cols[1],
                                  template=PBI_TEMPLATE, title=f"{num_cols[0]} vs {num_cols[1]}",
                                  color_discrete_sequence=[PBI_COLOR])
            fig_sc2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   font_color="#e2e8f0", title_font_color=PBI_COLOR)
            st.plotly_chart(fig_sc2, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for scatter.")

    with row2_c2:
        all_donut_options = cat_cols + [c for c in num_cols if c not in cat_cols]
        default_donut = city_col if city_col and city_col in pbi_df.columns else (cat_cols[0] if cat_cols else None)
        default_idx = all_donut_options.index(default_donut) if default_donut and default_donut in all_donut_options else 0
        if all_donut_options:
            donut_col = st.selectbox(
                "Distribution by",
                options=all_donut_options,
                index=default_idx,
                key="pbi_donut_col",
            )
            value_counts = pbi_df[donut_col].value_counts().head(10).reset_index()
            value_counts.columns = [donut_col, "count"]
            fig_pbi_donut = px.pie(
                value_counts, names=donut_col, values="count",
                hole=0.5,
                color_discrete_sequence=PBI_PALETTE,
                title=f"Distribution by {donut_col}",
                template=PBI_TEMPLATE,
            )
            fig_pbi_donut.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color=PBI_COLOR,
                legend=dict(font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(fig_pbi_donut, use_container_width=True)
        else:
            st.info("No categorical column for donut chart.")

    st.markdown('<div class="pbi-section-title">📦 Statistical Distribution & Correlations</div>', unsafe_allow_html=True)
    row3_c1, row3_c2 = st.columns(2)

    with row3_c1:
        box_x = city_col if city_col and city_col in pbi_df.columns else (cat_cols[0] if cat_cols else None)
        box_y = rating_col if rating_col and rating_col in pbi_df.columns else (num_cols[0] if num_cols else None)
        if box_x and box_y:
            fig_pbi_box = px.box(
                pbi_df, x=box_x, y=box_y,
                color=box_x,
                color_discrete_sequence=PBI_PALETTE,
                title=f"{box_y} Distribution by {box_x}",
                template=PBI_TEMPLATE,
                points="outliers",
            )
            fig_pbi_box.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color=PBI_COLOR,
                showlegend=False,
            )
            st.plotly_chart(fig_pbi_box, use_container_width=True)
        else:
            st.info("Need categorical + numeric column for box plot.")

    with row3_c2:
        if len(num_cols) >= 2:
            corr_matrix = pbi_df[num_cols].corr(numeric_only=True)
            fig_pbi_heat = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale=["#1a1a2e", "#F2C811"],
                title="Correlation Matrix",
                template=PBI_TEMPLATE,
            )
            fig_pbi_heat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color=PBI_COLOR,
            )
            st.plotly_chart(fig_pbi_heat, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation heatmap.")

    st.markdown('<div class="pbi-section-title">📈 Price Distribution & Top Performers</div>', unsafe_allow_html=True)
    row4_c1, row4_c2 = st.columns([1.2, 0.8])

    with row4_c1:
        hist_col = price_col if price_col and price_col in pbi_df.columns else (num_cols[0] if num_cols else None)
        if hist_col:
            fig_pbi_hist = px.histogram(
                pbi_df, x=hist_col, nbins=40,
                color_discrete_sequence=[PBI_COLOR],
                title=f"Distribution: {hist_col}",
                template=PBI_TEMPLATE,
                marginal="box",
            )
            fig_pbi_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color=PBI_COLOR,
            )
            st.plotly_chart(fig_pbi_hist, use_container_width=True)
        else:
            st.info("No numeric column for histogram.")

    with row4_c2:
        top_col = city_col if city_col and city_col in pbi_df.columns else (cat_cols[0] if cat_cols else None)
        sort_col = price_col if price_col and price_col in pbi_df.columns else (num_cols[0] if num_cols else None)
        if top_col and sort_col:
            top10 = (
                pbi_df.groupby(top_col)[sort_col]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": f"Avg {sort_col}", "count": "Records"})
                .sort_values(f"Avg {sort_col}", ascending=False)
                .head(10)
            )
            top10[f"Avg {sort_col}"] = top10[f"Avg {sort_col}"].round(2)
            st.markdown(f"**🏆 Top 10 by Avg {sort_col}**")
            st.dataframe(top10, use_container_width=True, hide_index=True)
        else:
            st.info("Need categorical + numeric column for top 10 table.")

   
    st.markdown("---")
    st.markdown('<div class="pbi-section-title">🤖 AI-Generated Insights</div>', unsafe_allow_html=True)

    auto_insights_key = f"auto_insights_{active_name}"
    pbi_insights_key  = f"pbi_insights_{active_name}"

    if pbi_insights_key not in st.session_state:
        if genai is not None and _gemini_model is not None:
            with st.spinner("Generating Power BI insights…"):
                try:
                    pbi_prompt = f"""You are a Power BI expert analyst. Based on this dataset sample, provide exactly 4 concise business insights (one sentence each) suitable for a Power BI executive dashboard. Format as a numbered list 1-4.

Dataset:
{active_df.head(10).to_string()}
"""
                    pbi_resp = _gemini_model.generate_content(pbi_prompt)
                    st.session_state[pbi_insights_key] = pbi_resp.text
                except Exception as e:
                    st.session_state[pbi_insights_key] = f"__ERROR__: {e}"
        else:
            st.session_state[pbi_insights_key] = "__UNAVAILABLE__"

    pbi_cached = st.session_state.get(pbi_insights_key, "__UNAVAILABLE__")

    if pbi_cached.startswith("__ERROR__:") or pbi_cached == "__UNAVAILABLE__":
        insight_lines = []
        if price_col and price_col in active_df.columns:
            insight_lines.append(f"💡 The average {price_col} across all records is **{active_df[price_col].mean():,.2f}**, with a range from {active_df[price_col].min():,.2f} to {active_df[price_col].max():,.2f}.")
        if rating_col and rating_col in active_df.columns:
            insight_lines.append(f"💡 Overall average {rating_col} stands at **{active_df[rating_col].mean():.2f}**, indicating the general quality level of the dataset.")
        if city_col and city_col in active_df.columns:
            top_city = active_df[city_col].value_counts().idxmax()
            insight_lines.append(f"💡 **{top_city}** is the most represented location in the dataset with {active_df[city_col].value_counts().max():,} records.")
        insight_lines.append(f"💡 Dataset contains **{len(active_df):,} records** across **{active_df.shape[1]} columns** after cleaning.")

        ins_cols = st.columns(min(4, len(insight_lines)))
        for i, ins in enumerate(insight_lines[:4]):
            with ins_cols[i]:
                st.markdown(
                    f'<div class="pbi-insight-box"><p class="pbi-insight-text">{ins}</p></div>',
                    unsafe_allow_html=True,
                )
    else:
        lines = [l.strip() for l in pbi_cached.split("\n") if l.strip() and l.strip()[0].isdigit()][:4]
        if not lines:
            lines = [l.strip() for l in pbi_cached.split("\n") if l.strip()][:4]

        ins_cols = st.columns(min(4, max(1, len(lines))))
        for i, ins in enumerate(lines[:4]):
            with ins_cols[i % len(ins_cols)]:
                clean_ins = ins.lstrip("1234. ").strip()
                st.markdown(
                    f'<div class="pbi-insight-box"><p class="pbi-insight-text">💡 {clean_ins}</p></div>',
                    unsafe_allow_html=True,
                )

    if st.button("🔄 Refresh PBI Insights", key="refresh_pbi_insights"):
        if pbi_insights_key in st.session_state:
            del st.session_state[pbi_insights_key]
        st.rerun()

    st.markdown(
        f"""
        <div class="pbi-footer">
            📊 AI Intelligence Dataset &nbsp;|&nbsp; Power BI Template &nbsp;|&nbsp;
            Dataset: <strong>{active_name}</strong> &nbsp;|&nbsp;
            {len(pbi_df):,} records displayed &nbsp;|&nbsp;
            Built with Streamlit + Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )
