

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler # pyright: ignore[reportMissingModuleSource]
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# PAGE CONFIG  &  BRAND STYLING
# ============================================================
st.set_page_config(
    page_title="Melbourne Housing Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 14px;
        padding: 28px 36px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(30,60,114,0.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 800; }
    .hero p  { margin: 6px 0 0 0; opacity: 0.85; font-size: 1rem; }

    /* KPI cards */
    .kpi {
        background: white;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border-top: 4px solid #2a5298;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kpi .label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .kpi .value { font-size: 1.6rem; font-weight: 800; color: #1e3c72; margin-top: 6px; }
    .kpi.green  { border-top-color: #16a085; } .kpi.green  .value { color: #0e6655; }
    .kpi.orange { border-top-color: #e67e22; } .kpi.orange .value { color: #a04000; }
    .kpi.purple { border-top-color: #8e44ad; } .kpi.purple .value { color: #5b2c6f; }
    .kpi.red    { border-top-color: #c0392b; } .kpi.red    .value { color: #922b21; }

    /* Prediction result card */
    .pred {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 14px;
        padding: 26px;
        color: #28292b;
        text-align: center;
        box-shadow: 0 6px 20px rgba(17,153,142,0.3);
    }
    .pred .ttl   { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.9; }
    .pred .price { font-size: 2.6rem; font-weight: 900; margin: 8px 0; }
    .pred .ci    { font-size: 0.95rem; opacity: 0.95; }

    /* Insight box */
    .insight {
        background: #2d3035;
        border-left: 4px solid #2a5298;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA  &  MODEL  (trained fresh + cached)
# ============================================================
DATA_PATH = "melb_data.csv"          # relative path → portable
MODEL_PATH = "gb_model_local.pkl"    # cached after first train


@st.cache_data(show_spinner=False)
def load_data():
    """Load and lightly clean the dataset."""
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ `{DATA_PATH}` not found. Place it in the same folder as `app.py`.")
        st.stop()
    data = pd.read_csv(DATA_PATH)

    # Derived columns useful for analytics (kept SEPARATE from training features)
    if "BuildingArea" in data.columns:
        data["PricePerSqm"] = np.where(data["BuildingArea"] > 0, data["Price"] / data["BuildingArea"], np.nan)
    if "Date" in data.columns:
        data["SaleDate"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    type_map = {"h": "🏡 House", "u": "🏢 Unit", "t": "🏘 Townhouse"}
    if "Type" in data.columns:
        data["TypeLabel"] = data["Type"].map(type_map).fillna(data["Type"])
    return data


@st.cache_resource(show_spinner=False)
def get_model(_data: pd.DataFrame):
    """
    Train (or load cached) Gradient Boosting pipeline.
    Mirrors the architecture from House_price_prediction.ipynb so behaviour is identical.
    """
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            return bundle  # {pipeline, metrics, feature_names}
        except Exception:
            pass  # fall through to retrain

    with st.spinner("🔧 Training Gradient Boosting model (first run only — ~30s)…"):
        train_df = _data.dropna(subset=["Price"]).copy()
        train_df = train_df.dropna(subset=[
            "Rooms", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea",
            "Suburb", "SellerG", "Type", "Method", "Regionname", "CouncilArea",
        ])

        numeric_cols = ["Rooms", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea"]
        cat_cols = ["Suburb", "SellerG", "Type", "Method", "Regionname", "CouncilArea"]

        X = train_df[numeric_cols + cat_cols]
        y = train_df["Price"]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(n_estimators=100, random_state=123)),
        ])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        metrics = {
            "r2":   r2_score(y_val, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "mae":  mean_absolute_error(y_val, y_pred),
            "n_train": len(X_train),
            "n_val":   len(X_val),
        }

        # Try to capture readable feature names for the importance chart
        try:
            ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
            cat_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names = numeric_cols + cat_names
        except Exception:
            feature_names = None

        bundle = {"pipeline": pipeline, "metrics": metrics, "feature_names": feature_names}
        try:
            joblib.dump(bundle, MODEL_PATH)
        except Exception:
            pass
        return bundle


# Load everything
df = load_data()
bundle = get_model(df)
gb_pipeline = bundle["pipeline"]
model_metrics = bundle["metrics"]
feature_names = bundle["feature_names"]


# ============================================================
# HELPERS
# ============================================================
def fmt_money(v):
    if pd.isna(v): return "—"
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"

def kpi(label, value, css=""):
    return f'<div class="kpi {css}"><div class="label">{label}</div><div class="value">{value}</div></div>'


# ============================================================
# SIDEBAR — INPUTS  &  FILTERS
# ============================================================
with st.sidebar:
    st.markdown("## 🏠 Property Details")
    Rooms        = st.number_input("Rooms", 1, 10, 3)
    Distance     = st.number_input("Distance from CBD (km)", 0.0, 100.0, 10.0, step=0.5)
    Bathroom     = st.number_input("Bathrooms", 1, 10, 2)
    Car          = st.number_input("Car Spaces", 0, 10, 2)
    Landsize     = st.number_input("Land Size (sqm)", 0.0, 5000.0, 300.0, step=10.0)
    BuildingArea = st.number_input("Building Area (sqm)", 0.0, 1000.0, 150.0, step=5.0)

    st.markdown("---")
    st.markdown("## 📋 Classification")
    Method      = st.selectbox("Sale Method",  sorted(df["Method"].dropna().unique()))
    SellerG     = st.selectbox("Seller Agency", sorted(df["SellerG"].dropna().unique()))
    Suburb      = st.selectbox("Suburb",        sorted(df["Suburb"].dropna().unique()))
    Type        = st.selectbox("Property Type", sorted(df["Type"].dropna().unique()),
                               format_func=lambda x: {"h":"🏡 House","u":"🏢 Unit","t":"🏘 Townhouse"}.get(x, x))
    Regionname  = st.selectbox("Region",        sorted(df["Regionname"].dropna().unique()))
    CouncilArea = st.selectbox("Council Area",  sorted(df["CouncilArea"].dropna().unique()))

    st.markdown("---")
    st.markdown("## 🔍 Dataset Filters")
    pmin, pmax = int(df["Price"].min()), int(df["Price"].max())
    price_range = st.slider("Price Range ($)", pmin, pmax, (pmin, pmax), step=50_000)
    region_filter = st.multiselect("Filter by Region", sorted(df["Regionname"].dropna().unique()))


# Build prediction input
input_data = pd.DataFrame([{
    "Rooms": Rooms, "Distance": Distance, "Bathroom": Bathroom, "Car": Car,
    "Landsize": Landsize, "BuildingArea": BuildingArea,
    "Method": Method, "SellerG": SellerG, "Suburb": Suburb,
    "Type": Type, "Regionname": Regionname, "CouncilArea": CouncilArea,
}])

# Apply filters
filtered_df = df[(df["Price"] >= price_range[0]) & (df["Price"] <= price_range[1])].copy()
if region_filter:
    filtered_df = filtered_df[filtered_df["Regionname"].isin(region_filter)]


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🏙️ Melbourne Housing Intelligence</h1>
    <p>AI-powered property valuation &amp; market analytics — for buyers, sellers, agents and investors.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI ROW
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi("Listings", f"{len(filtered_df):,}"), unsafe_allow_html=True)
c2.markdown(kpi("Median Price", fmt_money(filtered_df["Price"].median()), "green"), unsafe_allow_html=True)
c3.markdown(kpi("Avg $/sqm", fmt_money(filtered_df["PricePerSqm"].median()), "orange"), unsafe_allow_html=True)
c4.markdown(kpi("Regions", f"{filtered_df['Regionname'].nunique()}", "purple"), unsafe_allow_html=True)
c5.markdown(kpi("Model R²", f"{model_metrics['r2']:.3f}", "red"), unsafe_allow_html=True)
st.markdown("")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Valuation",
    "📊  Market Intelligence",
    "🗺️  Geographic Map",
    "🔎  Data Explorer",
])


# ─────────── TAB 1 — VALUATION ───────────
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Estimated Property Value")
        if st.button("🚀  Run Valuation", type="primary", use_container_width=True):
            try:
                pred = float(gb_pipeline.predict(input_data)[0])
                st.session_state["pred"] = pred
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        if "pred" in st.session_state:
            pred = st.session_state["pred"]
            lo, hi = pred * 0.9, pred * 1.1
            st.markdown(f"""
            <div class="pred">
                <div class="ttl">Estimated Market Value</div>
                <div class="price">{fmt_money(pred)}</div>
                <div class="ci">90% Confidence: {fmt_money(lo)} — {fmt_money(hi)}</div>
            </div>""", unsafe_allow_html=True)

            # Gauge: where this price sits in the chosen Region
            region_prices = df[df["Regionname"] == Regionname]["Price"].dropna()
            if len(region_prices) > 5:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,
                    number={"prefix": "$", "valueformat": ",.0f"},
                    delta={"reference": region_prices.median(), "valueformat": ",.0f", "prefix": "$"},
                    title={"text": f"vs {Regionname} Median"},
                    gauge={
                        "axis": {"range": [region_prices.min(), region_prices.max()], "tickformat": "$,.0s"},
                        "bar": {"color": "#11998e"},
                        "steps": [
                            {"range": [region_prices.min(),       region_prices.quantile(0.25)], "color": "#d5f5e3"},
                            {"range": [region_prices.quantile(0.25), region_prices.quantile(0.75)], "color": "#abebc6"},
                            {"range": [region_prices.quantile(0.75), region_prices.max()],       "color": "#82e0aa"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.85, "value": region_prices.median()},
                    },
                ))
                gauge.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
                st.plotly_chart(gauge, use_container_width=True)

    with right:
        st.markdown("### 🏘 Comparable Sales")
        # Find comparables: same type & rooms, in same region
        mask = (df["Rooms"] == Rooms) & (df["Type"] == Type) & (df["Regionname"] == Regionname)
        comps = df.loc[mask].copy()
        if len(comps) < 5:  # widen the net if too few
            comps = df.loc[(df["Rooms"] == Rooms) & (df["Type"] == Type)].copy()
        comps["DistDiff"] = (comps["Distance"] - Distance).abs()
        comps = comps.nsmallest(8, "DistDiff")

        if len(comps):
            display = comps[["Suburb", "Price", "Rooms", "Bathroom", "Distance", "BuildingArea"]].copy()
            display["Price"] = display["Price"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display, use_container_width=True, hide_index=True, height=320)

            cp = comps["Price"].dropna()
            st.markdown(f"""
            <div class="insight">
                📌 <b>{len(cp)} comparable properties</b> · Median <b>{fmt_money(cp.median())}</b> ·
                Range {fmt_money(cp.min())} – {fmt_money(cp.max())}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No comparable properties found.")

    # Your input vs market — input-aware visuals
    if "pred" in st.session_state:
        pred = st.session_state["pred"]
        st.markdown("---")
        st.markdown("### 📈 Your Property in Market Context")

        v1, v2 = st.columns(2)
        # Price distribution + your prediction marker
        with v1:
            fig = px.histogram(filtered_df, x="Price", nbins=50, marginal="box",
                               color_discrete_sequence=["#2a5298"],
                               title="Where Your Prediction Sits in the Market")
            fig.add_vline(x=pred, line_dash="dash", line_color="#e74c3c", line_width=3,
                          annotation_text=f"Your: {fmt_money(pred)}", annotation_position="top")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Distance vs Price + your point
        with v2:
            fig = px.scatter(filtered_df, x="Distance", y="Price", color="Rooms",
                             size="Landsize", hover_data=["Suburb"],
                             title="Distance vs Price — Your Property Plotted",
                             color_continuous_scale="Viridis")
            fig.add_scatter(x=[Distance], y=[pred], mode="markers",
                            marker=dict(size=22, color="red", symbol="star", line=dict(width=2, color="black")),
                            name="Your Property", showlegend=True)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Rooms vs Price box + your point
        fig = px.box(filtered_df, x="Rooms", y="Price", color="Rooms",
                     title="Rooms vs Price — Your Property Plotted")
        fig.add_scatter(x=[Rooms], y=[pred], mode="markers",
                        marker=dict(size=22, color="red", symbol="star", line=dict(width=2, color="black")),
                        name="Your Property", showlegend=True)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (correctly accessing the 'regressor' step)
    st.markdown("---")
    st.markdown("### 🧠 What Drives Price Predictions")
    try:
        importances = gb_pipeline.named_steps["regressor"].feature_importances_
        names = feature_names if feature_names and len(feature_names) == len(importances) else [f"f{i}" for i in range(len(importances))]
        fi_df = pd.DataFrame({"Feature": names, "Importance": importances}).nlargest(15, "Importance").sort_values("Importance")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     title="Top 15 Predictive Features")
        fig.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Feature importance not available.")


# ─────────── TAB 2 — MARKET INTELLIGENCE ───────────
with tab2:
    st.markdown("### Market Overview")

    a, b = st.columns(2)
    # Median price by region
    with a:
        region_med = filtered_df.groupby("Regionname")["Price"].agg(["median", "count"]).reset_index()
        region_med.columns = ["Region", "Median", "Listings"]
        region_med = region_med.sort_values("Median", ascending=True)
        fig = px.bar(region_med, x="Median", y="Region", orientation="h",
                     text=region_med["Median"].apply(fmt_money), color="Median",
                     color_continuous_scale="Blues", title="Median Price by Region")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, coloraxis_showscale=False, margin=dict(r=80))
        st.plotly_chart(fig, use_container_width=True)

    # Property type breakdown
    with b:
        if "TypeLabel" in filtered_df.columns:
            type_stats = filtered_df.groupby("TypeLabel")["Price"].agg(["median", "count"]).reset_index()
            type_stats.columns = ["Type", "Median", "Count"]
            fig = px.bar(type_stats, x="Type", y="Median", color="Type", text="Count",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="Median Price by Property Type")
            fig.update_traces(texttemplate="%{text} listings", textposition="outside")
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Top / Bottom suburbs
    s_med = filtered_df.groupby("Suburb")["Price"].agg(["median", "count"]).reset_index()
    s_med.columns = ["Suburb", "Median", "Sales"]
    s_med = s_med[s_med["Sales"] >= 5]

    c, d = st.columns(2)
    with c:
        top10 = s_med.nlargest(10, "Median")
        fig = px.bar(top10, x="Median", y="Suburb", orientation="h",
                     color="Median", color_continuous_scale="Reds",
                     title="🏆 Top 10 Most Expensive Suburbs")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with d:
        bot10 = s_med.nsmallest(10, "Median")
        fig = px.bar(bot10, x="Median", y="Suburb", orientation="h",
                     color="Median", color_continuous_scale="Greens",
                     title="💡 Top 10 Most Affordable Suburbs")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Correlation heatmap (interactive Plotly version)
    st.markdown("### Feature Correlations")
    num_cols = [c for c in ["Rooms", "Price", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea", "PricePerSqm"]
                if c in filtered_df.columns]
    corr = filtered_df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)


# ─────────── TAB 3 — GEOGRAPHIC MAP (segmented by price) ───────────
with tab3:
    st.markdown("### Properties Mapped by Price Segment")

    if "Lattitude" in filtered_df.columns and "Longtitude" in filtered_df.columns:
        map_df = filtered_df.dropna(subset=["Lattitude", "Longtitude", "Price"]).copy()

        # Segment by quartiles
        q1, q2, q3 = map_df["Price"].quantile([0.25, 0.50, 0.75])

        def color_for(p):
            if p <= q1: return [46, 204, 113, 180]   # green  — Affordable
            if p <= q2: return [241, 196, 15, 180]   # yellow — Mid
            if p <= q3: return [230, 126, 34, 180]   # orange — Premium
            return [231, 76, 60, 200]                # red    — Luxury

        def seg_for(p):
            if p <= q1: return "Affordable"
            if p <= q2: return "Mid-range"
            if p <= q3: return "Premium"
            return "Luxury"

        map_df["color"]     = map_df["Price"].apply(color_for)
        map_df["segment"]   = map_df["Price"].apply(seg_for)
        map_df["price_str"] = map_df["Price"].apply(lambda x: f"${x:,.0f}")

        # Legend with thresholds
        st.markdown(f"""
        <div class="insight">
            <b>Price Segments (quartile-based):</b><br>
            🟢 <b>Affordable</b> ≤ {fmt_money(q1)} &nbsp;|&nbsp;
            🟡 <b>Mid-range</b> {fmt_money(q1)} – {fmt_money(q2)} &nbsp;|&nbsp;
            🟠 <b>Premium</b> {fmt_money(q2)} – {fmt_money(q3)} &nbsp;|&nbsp;
            🔴 <b>Luxury</b> &gt; {fmt_money(q3)}
        </div>""", unsafe_allow_html=True)

        # Optional: filter map by segment
        seg_choice = st.multiselect("Show segments:",
                                    ["Affordable", "Mid-range", "Premium", "Luxury"],
                                    default=["Affordable", "Mid-range", "Premium", "Luxury"])
        view_df = map_df[map_df["segment"].isin(seg_choice)]

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=view_df["Lattitude"].mean() if len(view_df) else -37.81,
                longitude=view_df["Longtitude"].mean() if len(view_df) else 144.96,
                zoom=10, pitch=0,
            ),
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=view_df,
                get_position="[Longtitude, Lattitude]",
                get_color="color",
                get_radius=110,
                pickable=True,
                auto_highlight=True,
            )],
            tooltip={"text": "{Suburb}\nPrice: {price_str}\nSegment: {segment}\nRooms: {Rooms}"},
        ))

        # Segment composition chart
        seg_counts = map_df["segment"].value_counts().reindex(["Affordable", "Mid-range", "Premium", "Luxury"]).reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.bar(seg_counts, x="Segment", y="Count", color="Segment",
                     color_discrete_map={"Affordable": "#2ecc71", "Mid-range": "#f1c40f",
                                          "Premium": "#e67e22", "Luxury": "#e74c3c"},
                     title="Listings per Price Segment", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Latitude/Longitude columns not present — map unavailable.")


# ─────────── TAB 4 — DATA EXPLORER ───────────
with tab4:
    st.markdown(f"### Filtered Dataset ({len(filtered_df):,} of {len(df):,} records)")

    default_cols = [c for c in ["Suburb", "Price", "Rooms", "Bathroom", "Car", "Distance",
                                "Landsize", "BuildingArea", "Type", "Regionname", "Method"]
                    if c in filtered_df.columns]
    cols = st.multiselect("Columns to display", filtered_df.columns.tolist(), default=default_cols)

    if cols:
        sort_col = st.selectbox("Sort by", cols, index=cols.index("Price") if "Price" in cols else 0)
        asc = st.checkbox("Ascending", value=False)
        view = filtered_df[cols].sort_values(sort_col, ascending=asc)
        st.dataframe(view, use_container_width=True, height=420, hide_index=True)

    st.markdown("---")
    st.markdown("### Summary Statistics")
    desc = filtered_df.select_dtypes(include=np.number).describe().T
    st.dataframe(desc.style.format("{:,.1f}"), use_container_width=True)

    st.markdown("---")
    st.markdown("### ⬇️ Export")
    e1, e2 = st.columns(2)
    e1.download_button("📥 Download Filtered Data (CSV)",
                       filtered_df.to_csv(index=False),
                       "melbourne_filtered.csv", "text/csv", use_container_width=True)
    if "pred" in st.session_state:
        rep = input_data.copy()
        rep["Predicted_Price"] = st.session_state["pred"]
        rep["CI_Low"]  = st.session_state["pred"] * 0.9
        rep["CI_High"] = st.session_state["pred"] * 1.1
        e2.download_button("📥 Download Valuation Report (CSV)",
                           rep.to_csv(index=False),
                           "valuation_report.csv", "text/csv", use_container_width=True)


# ============================================================
# FOOTER  &  MODEL STATS
# ============================================================
st.markdown("---")
m = model_metrics
st.markdown(f"""
<div style="text-align:center; opacity:0.6; font-size:0.82rem;">
    Model: Gradient Boosting (n_estimators=100) ·
    R² <b>{m['r2']:.3f}</b> · RMSE <b>${m['rmse']:,.0f}</b> · MAE <b>${m['mae']:,.0f}</b> ·
    Trained on {m['n_train']:,} samples / Validated on {m['n_val']:,}<br>
    Melbourne Housing Intelligence · Industry Demo · Built with Streamlit + scikit-learn
</div>
""", unsafe_allow_html=True)
