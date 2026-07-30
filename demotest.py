import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from openai import OpenAI

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="EV Insights",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# THEME TOKENS — dark luxury BI: black canvas, neon mint accent
# ------------------------------------------------------------------
INK = "#0B0B0B"
CARD_GLASS = "rgba(255,255,255,0.035)"
CARD_SOLID = "#111417"          # solid near-black used for Plotly backgrounds
BORDER_MINT = "rgba(82,242,198,0.22)"
MINT = "#52F2C6"
MINT_DIM = "#2FBE97"
WHITE = "#FFFFFF"
GRAY_LIGHT = "#B8BCC4"
GRAY_MID = "#6B7280"
GRAY_DIM = "#3A3F47"
ORANGE = "#FF8C42"              # reserved strictly for highlighting / negative signal

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .stApp {{
        background: radial-gradient(circle at top, #121517 0%, {INK} 55%);
    }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1360px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* ================= SIDEBAR — control panel ================= */
    section[data-testid="stSidebar"] {{
        background-color: {INK};
        border-right: 1px solid {BORDER_MINT};
        box-shadow: inset -50px 0 70px -60px rgba(82,242,198,0.15);
    }}
    section[data-testid="stSidebar"] .block-container {{ padding: 1.6rem 1.1rem; }}
    section[data-testid="stSidebar"] h3 {{
        color: {WHITE} !important; font-weight: 800; letter-spacing: .02em;
    }}
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p {{ color: {GRAY_LIGHT} !important; font-weight: 600; }}

    /* ================= dashboard-card / sidebar-card / filter-card ================= */
    .dashboard-card,
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {CARD_GLASS} !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {BORDER_MINT} !important;
        border-radius: 22px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.55) !important;
        padding: 8px !important;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
        overflow: visible !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 10px 40px rgba(82,242,198,0.12), 0 0 0 1px {BORDER_MINT} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.02) !important;
        border-radius: 18px !important;
        margin-bottom: 14px;
        padding: 14px 14px 22px 14px !important;
        overflow: visible !important;
    }}
    [data-testid="stForm"] {{
        background: {CARD_GLASS} !important;
        backdrop-filter: blur(14px);
        border: 1px solid {BORDER_MINT} !important;
        border-radius: 22px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.55) !important;
        padding: 22px !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] label,
    [data-testid="stForm"] label {{ color: {WHITE} !important; }}

    /* ================= form widgets ================= */
    .stSelectbox > div > div, .stMultiSelect > div > div,
    .stNumberInput > div > div, .stTextArea textarea, [data-baseweb="input"] {{
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px solid {BORDER_MINT} !important;
        border-radius: 12px !important;
        color: {WHITE} !important;
    }}
    .stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within,
    .stTextArea textarea:focus {{
        border-color: {MINT} !important; box-shadow: 0 0 0 3px rgba(82,242,198,0.18) !important;
    }}

    /* multiselect pills — scoped to multiselect only, glass mint */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: rgba(82,242,198,0.18) !important;
        border: 1px solid rgba(82,242,198,0.5) !important;
        border-radius: 999px !important;
    }}
    .stMultiSelect [data-baseweb="tag"] span, .stMultiSelect [data-baseweb="tag"] * {{
        color: {MINT} !important; font-weight: 700;
    }}
    .stMultiSelect [data-baseweb="tag"] svg {{ fill: {MINT} !important; }}

    /* ================= sliders =================
       Track fill color is inherited from the Streamlit theme's primaryColor
       (set in .streamlit/config.toml — this is the actual fix for the red
       track, since Streamlit sets that color at the source, not via CSS
       we can reliably beat with guesses). CSS below only adds polish. */
    .stSlider, .stSelectSlider {{ overflow: visible !important; padding-top: 6px; margin-bottom: 6px; }}
    .stSlider [data-baseweb="slider"], .stSelectSlider [data-baseweb="slider"] {{ overflow: visible !important; }}
    .stSlider [role="slider"], .stSelectSlider [role="slider"] {{
        background-color: {MINT} !important; border: 3px solid {INK} !important;
        box-shadow: 0 0 10px rgba(82,242,198,0.7) !important;
    }}
    /* value label above the thumb — Streamlit only shows this on hover by
       default; force it always visible with proper contrast */
    [data-testid="stThumbValue"] {{
        opacity: 1 !important; visibility: visible !important;
        background-color: {MINT} !important; color: {INK} !important;
        font-weight: 800 !important; border-radius: 8px !important;
        padding: 2px 9px !important; white-space: nowrap !important;
    }}
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: {GRAY_LIGHT} !important; }}

    /* ================= buttons — filled mint pill CTA =================
       Text color forced on every nested element too, since Streamlit wraps
       button labels in their own <p>/<div> that otherwise keeps the
       theme's white text color regardless of the button background. */
    .stButton>button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {{
        background-color: {MINT} !important; color: {INK} !important;
        border-radius: 999px !important; font-weight: 800 !important; border: none !important;
        padding: 0.55rem 1.4rem !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }}
    .stButton>button *, [data-testid="stFormSubmitButton"] button *,
    [data-testid="stDownloadButton"] button * {{ color: {INK} !important; }}
    .stButton>button:hover, [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {{
        box-shadow: 0 0 22px rgba(82,242,198,0.55) !important; transform: translateY(-1px);
    }}

    /* ================= nav-pill (tabs) =================
       Also kills BaseWeb's default underline indicator, which was rendering
       as a stray red bar under whichever tab it thought was "selected" —
       our filled mint pill already shows the active state, so it's redundant. */
    .nav-pill, .stTabs [data-baseweb="tab-list"] {{ gap: 10px; border-bottom: none; margin-bottom: 22px; flex-wrap: wrap; }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
    .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
    .stTabs [data-baseweb="tab"] {{
        height: 42px; background-color: rgba(255,255,255,0.03); border-radius: 999px;
        color: {MINT}; font-weight: 700; padding: 0 22px; border: 1px solid {BORDER_MINT};
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ box-shadow: 0 0 14px rgba(82,242,198,0.3); border-color: {MINT}; }}
    .stTabs [aria-selected="true"] {{
        background-color: {MINT} !important; color: {INK} !important; border-color: {MINT} !important;
        box-shadow: 0 0 18px rgba(82,242,198,0.4) !important;
    }}

    /* ================= metric-card (KPI cards) ================= */
    .metric-card {{
        position: relative;
        background: {CARD_GLASS};
        backdrop-filter: blur(14px);
        border: 1px solid {BORDER_MINT};
        border-left: 4px solid {MINT};
        border-radius: 20px;
        padding: 18px 20px;
        min-height: 128px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.5);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 16px 40px rgba(82,242,198,0.18), 0 0 0 1px {BORDER_MINT};
    }}
    .metric-icon {{ font-size: 1.2rem; margin-bottom: 6px; color: {MINT}; font-weight: 900; }}
    .metric-icon.neg {{ color: {ORANGE}; }}
    .metric-label {{
        color: {GRAY_LIGHT}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: .08em; margin-bottom: 6px;
    }}
    .metric-value {{ color: {WHITE}; font-size: 1.9rem; font-weight: 900; line-height: 1.1; }}
    .metric-sub {{ color: {MINT}; font-size: 0.75rem; margin-top: 6px; font-weight: 700; }}
    .metric-sub.neg {{ color: {ORANGE}; }}

    /* ================= chart-card title ================= */
    .chart-card-title {{
        color: {WHITE}; font-size: 0.98rem; font-weight: 700; margin: 8px 0 10px 10px;
        letter-spacing: .01em;
    }}
    .chart-card-title span {{ color: {MINT}; }}

    /* ================= typography ================= */
    .hero-title {{
        font-size: 2.3rem; font-weight: 900; color: {WHITE}; letter-spacing: -0.01em; margin-bottom: 4px;
    }}
    .hero-title span {{ color: {MINT}; }}
    .hero-sub {{ color: {GRAY_MID}; font-size: 0.95rem; margin-bottom: 14px; font-weight: 500; }}

    .navbar {{ display: flex; align-items: center; justify-content: space-between; padding: 6px 4px 20px 4px; margin-bottom: 4px; }}
    .navbar-brand {{ font-size: 1.6rem; font-weight: 900; color: {WHITE}; letter-spacing: -0.01em; }}
    .navbar-brand span {{ color: {MINT}; }}
    .navbar-sub {{ color: {GRAY_MID}; font-size: 0.82rem; font-weight: 600; }}

    .filter-pill {{
        display: inline-block; background: rgba(82,242,198,0.1); color: {MINT};
        border: 1px solid {BORDER_MINT}; border-radius: 999px; padding: 5px 16px;
        font-size: 0.8rem; font-weight: 700; margin-bottom: 18px;
    }}

    /* alerts / dataframe polish */
    .stAlert {{
        background-color: rgba(255,255,255,0.04) !important; border-radius: 16px !important;
        border: 1px solid {BORDER_MINT} !important; color: {WHITE} !important;
    }}
    .stProgress > div > div > div > div {{ background-color: {MINT} !important; }}
    [data-testid="stDataFrame"] {{ border-radius: 16px; overflow: hidden; border: 1px solid {BORDER_MINT}; }}
</style>
""", unsafe_allow_html=True)

ACCENT = MINT
PALETTE_NEUTRAL = [MINT, GRAY_LIGHT, MINT_DIM, GRAY_MID]
PALETTE_BINARY = [ORANGE, MINT]
PALETTE_SEVERITY = [GRAY_MID, MINT_DIM, ORANGE]
PLOT_TEMPLATE = "plotly_dark"
CH = 300

# ------------------------------------------------------------------
# DATA + MODEL LOADING
# ------------------------------------------------------------------
import json
CH = 340
CARD_SOLID = "#111827"
PLOT_TEMPLATE = "plotly_dark"

TARGET = "ev_adoption_likelihood"
TARGET_ORDER = ["Low", "Medium", "High"]


@st.cache_data
def load_data():
    df = pd.read_csv("global_ev_adoption_behavior_2026.csv")
    # -------------------------
    # Fix categorical columns
    # -------------------------
    df["education_level"] = df["education_level"].fillna("Unknown").astype(str)
    df["city_type"] = df["city_type"].fillna("Unknown").astype(str)
    df["current_vehicle_type"] = df["current_vehicle_type"].fillna("Unknown").astype(str)

    # -------------------------
    # Missing Value Imputation
    # -------------------------
    for col in ["charging_station_accessibility", "ev_knowledge_score"]:
        df[col] = df[col].fillna(df[col].median())

    # -------------------------
    # Data Cleaning
    # -------------------------
    df["fuel_expense_per_month"] = df["fuel_expense_per_month"].clip(lower=0)

    # -------------------------
    # Feature Engineering
    # -------------------------
    df["log_annual_income"] = np.log1p(df["annual_income"])

    df["fuel_cost_to_income_ratio"] = (
        df["fuel_expense_per_month"]
        / (df["annual_income"] / 12 + 1)
    )

    df["charging_cost_per_kwh_actual"] = (
        df["monthly_charging_cost"]
        / (df["monthly_energy_consumption_kwh"] + 1)
    )

    df["commute_consistency"] = (
        df["weekly_travel_distance_km"]
        / (df["daily_commute_km"] * 7 + 1)
    )

    df["anxiety_minus_knowledge"] = (
        df["range_anxiety_score"]
        - df["ev_knowledge_score"]
    )

    df["awareness_composite"] = (
        df["environmental_awareness_score"]
        + df["government_incentive_awareness"]
        + df["technology_affinity_score"]
    ) / 3

    return df


@st.cache_resource
def load_model():
    return joblib.load("best_ev_adoption_model.pkl")


@st.cache_resource
def load_feature_columns():
    with open("feature_columns.json", "r") as f:
        return json.load(f)


@st.cache_resource
def load_llm_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )


df = load_data()
model = load_model()
feature_columns = load_feature_columns()
client = load_llm_client()

# ------------------------------------------------------------------
# UI HELPERS
# ------------------------------------------------------------------
def kpi(col, icon, label, value, sub=None, sub_neg=False):
    sub_html = f"<div class='metric-sub{' neg' if sub_neg else ''}'>{sub}</div>" if sub else ""
    icon_cls = "metric-icon neg" if sub_neg else "metric-icon"
    col.markdown(
        f"""<div class="metric-card">
            <div class="{icon_cls}">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>{sub_html}</div>""",
        unsafe_allow_html=True,
    )

def chart_layout(fig, height=CH, **kw):
    fig.update_layout(
        paper_bgcolor=CARD_SOLID, plot_bgcolor=CARD_SOLID,
        font_color=GRAY_LIGHT, height=height,
        margin=dict(t=10, b=10, l=10, r=10), **kw
    )
    return fig

def purchase_rate_by(data, col, title, order=None, color_seq=PALETTE_NEUTRAL):
    with st.container(border=True):
        st.markdown(f'<div class="chart-card-title">{title}</div>', unsafe_allow_html=True)
        if data.empty:
            st.info("No records match the current filters.")
            return
        tmp = data.groupby(col, observed=True)[TARGET].apply(
    lambda s: (s == "Yes").mean() * 100
)
        if order:
            tmp = tmp.reindex(order)
        tmp = tmp.reset_index()
        tmp.columns = [col, "Rate (%)"]
        fig = px.bar(tmp, x=col, y="Rate (%)", text="Rate (%)", color=col,
                     color_discrete_sequence=color_seq, template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = chart_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{col}_{title}")

def page_header(title, subtitle):
    parts = title.split(" ")
    accented = f'{" ".join(parts[:-1])} <span>{parts[-1]}</span>' if len(parts) > 1 else f'<span>{title}</span>'
    st.markdown(f'<div class="hero-title">{accented}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{subtitle}</div>', unsafe_allow_html=True)
    pct = len(fdf) / len(df) * 100 if len(df) else 0
    st.markdown(
        f'<div class="filter-pill">◆ {len(fdf):,} of {len(df):,} buyers in view ({pct:.0f}%) — via sidebar filters</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# SIDEBAR — Control Panel
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙ Control Panel")
    st.caption("Filters apply across every dashboard page.")

    with st.container(border=True):
        city_f = st.multiselect(
            "🏙 City Type",
            sorted(df["city_type"].dropna().astype(str).unique()),
            default=sorted(df["city_type"].unique())
        )

        education_f = st.multiselect(
            "🎓 Education",
            sorted(df["education_level"].dropna().astype(str).unique()),
            default=sorted(df["education_level"].unique())
        )

    with st.container(border=True):

        vehicle_f = st.multiselect(
            "🚗 Vehicle Type",
            sorted(df["current_vehicle_type"].dropna().astype(str).unique()),
            default=sorted(df["current_vehicle_type"].unique())
        )

        home_charge_f = st.select_slider(
            "🔌 Home Charging",
            options=["All", "Yes", "No"],
            value="All"
        )

    with st.container(border=True):

        income_f = st.slider(
            "💰 Annual Income",
            int(df["annual_income"].min()),
            int(df["annual_income"].max()),
            (
                int(df["annual_income"].min()),
                int(df["annual_income"].max())
            ),
            step=1000
        )

        anxiety_f = st.slider(
            "⚠ Range Anxiety Score",
            int(df["range_anxiety_score"].min()),
            int(df["range_anxiety_score"].max()),
            (
                int(df["range_anxiety_score"].min()),
                int(df["range_anxiety_score"].max())
            )
        )

    if st.button("↺ Reset Filters", use_container_width=True):
        st.rerun()

fdf = df[
    df["city_type"].isin(city_f)
    & df["education_level"].isin(education_f)
    & df["current_vehicle_type"].isin(vehicle_f)
    & df["annual_income"].between(*income_f)
    & df["range_anxiety_score"].between(*anxiety_f)
]

if home_charge_f != "All":
    fdf = fdf[
        fdf["home_charging_available"] == home_charge_f
    ]

# ------------------------------------------------------------------
# NAVBAR
# ------------------------------------------------------------------
st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">⚡ EV<span>Insights</span></div>
    <div class="navbar-sub">AI-Powered EV Adoption Analytics Platform</div>
</div>
""", unsafe_allow_html=True)

tab_dashboard, tab_insights, tab_predict, tab_ai = st.tabs([
    "🏠 Dashboard",
    "📈 Consumer Insights",
    "🤖 EV Prediction",
    "💬 AI Assistant"
])

if fdf.empty:
    st.warning("⚠️ No records match your current sidebar filters — widen them to see data on any page.")

# ==================================================================
# PAGE 1: DASHBOARD
# ==================================================================
with tab_dashboard:
    page_header(
        "EV Adoption Dashboard",
        "Executive overview of EV adoption trends and consumer behaviour."
    )

    if not fdf.empty:

        adoption_rate = (fdf[TARGET] == "Yes").mean() * 100
        avg_income = fdf["annual_income"].mean()
        avg_awareness = fdf["awareness_composite"].mean()
        home_charge_pct = (fdf["home_charging_available"] == "Yes").mean() * 100
        high_anxiety_pct = (
            fdf["range_anxiety_score"] >= 7
        ).mean() * 100

        k1, k2, k3, k4, k5 = st.columns(5)

        kpi(k1, "👥", "Respondents", f"{len(fdf):,}")

        kpi(k2, "⚡", "EV Adoption", f"{adoption_rate:.1f}%")

        kpi(k3, "💰", "Avg Annual Income", f"${avg_income:,.0f}")

        kpi(
            k4,
            "⚠️",
            "High Range Anxiety",
            f"{high_anxiety_pct:.1f}%"
        )

        kpi(
            k5,
            "🔌",
            "Home Charging",
            f"{home_charge_pct:.1f}%"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            with st.container(border=True):
                st.markdown(
                    '<div class="chart-card-title">EV Adoption Distribution</div>',
                    unsafe_allow_html=True,
                )

                tmp = (
                    fdf[TARGET]
                    .value_counts()
                    .reset_index()
                )
                tmp.columns = ["Decision", "Count"]

                fig = px.pie(
                    tmp,
                    names="Decision",
                    values="Count",
                    hole=0.62,
                    color="Decision",
                    color_discrete_map={
                        "Yes": MINT,
                        "No": ORANGE,
                    },
                    template=PLOT_TEMPLATE,
                )

                fig.update_traces(textinfo="percent+label")
                fig = chart_layout(fig, showlegend=False)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="dashboard_pie",
                )

        with c2:
            purchase_rate_by(
                fdf,
                "charging_station_accessibility",
                "EV Adoption by Charging Accessibility",
            )

        with c3:
            purchase_rate_by(
                fdf,
                "city_type",
                "EV Adoption by City Type",
            )
# ==================================================================
# PAGE 3: CHARGING INFRASTRUCTURE
# ==================================================================
with tab_insights:
    page_header("Charging Infrastructure", "How access to chargers shapes range anxiety and buying decisions.")

    if not fdf.empty:
        c1, c2, c3 = st.columns(3)

with c1:
    purchase_rate_by(
        fdf,
        "education_level",
        "EV Adoption by Education"
    )

with c2:
    purchase_rate_by(
        fdf,
        "city_type",
        "EV Adoption by City Type"
    )

with c3:
    purchase_rate_by(
        fdf,
        "current_vehicle_type",
        "EV Adoption by Vehicle Type"
    )

c4, c5 = st.columns(2)

with c4:
    purchase_rate_by(
        fdf,
        "previous_ev_experience",
        "Previous EV Experience"
    )

with c5:
    purchase_rate_by(
        fdf,
        "home_charging_available",
        "Home Charging Availability"
    )

c6, c7 = st.columns(2)

with c6:

    with st.container(border=True):

        st.markdown(
            '<div class="chart-card-title">Environmental Awareness Distribution</div>',
            unsafe_allow_html=True
        )

        fig = px.histogram(
            fdf,
            x="environmental_awareness_score",
            nbins=10,
            color_discrete_sequence=[MINT],
            template=PLOT_TEMPLATE
        )

        fig = chart_layout(fig)

        st.plotly_chart(fig, use_container_width=True)

with c7:

    with st.container(border=True):

        st.markdown(
            '<div class="chart-card-title">Technology Affinity Distribution</div>',
            unsafe_allow_html=True
        )

        fig = px.histogram(
            fdf,
            x="technology_affinity_score",
            nbins=10,
            color_discrete_sequence=[MINT],
            template=PLOT_TEMPLATE
        )

        fig = chart_layout(fig)

        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE 4: BUYING BEHAVIOR
# ==================================================================
with tab_insights:
    page_header("Buying Behavior Drivers", "Subsidies, environmental concern, and other purchase triggers.")

    if not fdf.empty:
        c1, c2, c3 = st.columns(3)

with c1:
    purchase_rate_by(
        fdf,
        "government_incentive_awareness",
        "Government Incentive Awareness"
    )

with c2:
    purchase_rate_by(
        fdf,
        "battery_replacement_concern",
        "Battery Replacement Concern"
    )

with c3:
    purchase_rate_by(
        fdf,
        "charging_station_accessibility",
        "Charging Station Accessibility"
    )

c4, c5 = st.columns(2)

with c4:

    with st.container(border=True):

        st.markdown(
            '<div class="chart-card-title">Range Anxiety Score Distribution</div>',
            unsafe_allow_html=True
        )

        fig = px.histogram(
            fdf,
            x="range_anxiety_score",
            nbins=10,
            color_discrete_sequence=[MINT],
            template=PLOT_TEMPLATE
        )

        fig = chart_layout(fig)

        st.plotly_chart(fig, use_container_width=True)

with c5:

    with st.container(border=True):

        st.markdown(
            '<div class="chart-card-title">Awareness Composite Distribution</div>',
            unsafe_allow_html=True
        )

        fig = px.histogram(
            fdf,
            x="awareness_composite",
            nbins=10,
            color_discrete_sequence=[MINT],
            template=PLOT_TEMPLATE
        )

        fig = chart_layout(fig)

        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE 5: PREDICTION
# ==================================================================
# ==================================================================
# PAGE 3: EV PREDICTION
# ==================================================================
with tab_predict:

    page_header(
        "EV Adoption Prediction",
        "Predict a user's EV adoption likelihood using the trained CatBoost model."
    )

    with st.form("prediction_form"):

        st.markdown(
            '<div class="chart-card-title">Enter Customer Information</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        # -------------------------------------------------------
        # Personal Information
        # -------------------------------------------------------
        with c1:

            annual_income = st.number_input(
                "Annual Income",
                min_value=0,
                value=60000,
                step=1000
            )

            education_level = st.selectbox(
                "Education Level",
                sorted(df["education_level"].unique())
            )

            city_type = st.selectbox(
                "City Type",
                sorted(df["city_type"].unique())
            )

            current_vehicle_type = st.selectbox(
                "Current Vehicle Type",
                sorted(df["current_vehicle_type"].unique())
            )

            previous_ev_experience = st.selectbox(
                "Previous EV Experience",
                ["Yes", "No"]
            )

        # -------------------------------------------------------
        # Charging & Travel
        # -------------------------------------------------------
        with c2:

            home_charging_available = st.selectbox(
                "Home Charging Available",
                ["Yes", "No"]
            )

            charging_station_accessibility = st.slider(
                "Charging Station Accessibility",
                1,
                10,
                5
            )

            nearest_charging_station_km = st.number_input(
                "Nearest Charging Station (km)",
                min_value=0.0,
                value=5.0
            )

            daily_commute_km = st.number_input(
                "Daily Commute (km)",
                min_value=0.0,
                value=25.0
            )

            weekly_travel_distance_km = st.number_input(
                "Weekly Travel Distance (km)",
                min_value=0.0,
                value=180.0
            )

        # -------------------------------------------------------
        # EV Behaviour
        # -------------------------------------------------------
        with c3:

            monthly_energy_consumption_kwh = st.number_input(
                "Monthly Energy Consumption (kWh)",
                min_value=0.0,
                value=300.0
            )

            monthly_charging_cost = st.number_input(
                "Monthly Charging Cost",
                min_value=0.0,
                value=70.0
            )

            fuel_expense_per_month = st.number_input(
                "Monthly Fuel Expense",
                min_value=0.0,
                value=150.0
            )

            electricity_cost_per_kwh = st.number_input(
                "Electricity Cost / kWh",
                min_value=0.0,
                value=0.20,
                format="%.2f"
            )

            vehicle_age_years = st.number_input(
                "Vehicle Age (Years)",
                min_value=0,
                value=4
            )

        st.markdown("---")

        st.markdown(
            '<div class="chart-card-title">Behaviour & Awareness Scores</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3 = st.columns(3)

        with s1:

            environmental_awareness_score = st.slider(
                "Environmental Awareness",
                1,
                10,
                7
            )

            government_incentive_awareness = st.slider(
                "Government Incentive Awareness",
                1,
                10,
                6
            )

        with s2:

            technology_affinity_score = st.slider(
                "Technology Affinity",
                1,
                10,
                7
            )

            ev_knowledge_score = st.slider(
                "EV Knowledge",
                1,
                10,
                6
            )

        with s3:

            range_anxiety_score = st.slider(
                "Range Anxiety",
                1,
                10,
                5
            )

            battery_replacement_concern = st.slider(
                "Battery Replacement Concern",
                1,
                10,
                6
            )

        submitted = st.form_submit_button(
            "⚡ Predict EV Adoption",
            use_container_width=True
        )
 

if submitted:

    # ------------------------------------------------------
    # Create Input DataFrame
    # ------------------------------------------------------

    input_df = pd.DataFrame({

        "annual_income": [annual_income],
        "education_level": [education_level],
        "city_type": [city_type],
        "current_vehicle_type": [current_vehicle_type],
        "previous_ev_experience": [previous_ev_experience],
        "home_charging_available": [home_charging_available],
        "charging_station_accessibility": [charging_station_accessibility],
        "nearest_charging_station_km": [nearest_charging_station_km],
        "daily_commute_km": [daily_commute_km],
        "weekly_travel_distance_km": [weekly_travel_distance_km],
        "monthly_energy_consumption_kwh": [monthly_energy_consumption_kwh],
        "monthly_charging_cost": [monthly_charging_cost],
        "fuel_expense_per_month": [fuel_expense_per_month],
        "electricity_cost_per_kwh": [electricity_cost_per_kwh],
        "vehicle_age_years": [vehicle_age_years],
        "environmental_awareness_score": [environmental_awareness_score],
        "government_incentive_awareness": [government_incentive_awareness],
        "technology_affinity_score": [technology_affinity_score],
        "ev_knowledge_score": [ev_knowledge_score],
        "range_anxiety_score": [range_anxiety_score],
        "battery_replacement_concern": [battery_replacement_concern]

    })

    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------

    input_df["fuel_expense_per_month"] = input_df[
        "fuel_expense_per_month"
    ].clip(lower=0)

    input_df["log_annual_income"] = np.log1p(
        input_df["annual_income"]
    )

    input_df["fuel_cost_to_income_ratio"] = (
        input_df["fuel_expense_per_month"]
        /
        (input_df["annual_income"] / 12 + 1)
    )

    input_df["charging_cost_per_kwh_actual"] = (
        input_df["monthly_charging_cost"]
        /
        (input_df["monthly_energy_consumption_kwh"] + 1)
    )

    input_df["commute_consistency"] = (
        input_df["weekly_travel_distance_km"]
        /
        (input_df["daily_commute_km"] * 7 + 1)
    )

    input_df["anxiety_minus_knowledge"] = (
        input_df["range_anxiety_score"]
        -
        input_df["ev_knowledge_score"]
    )

    input_df["awareness_composite"] = (

        input_df["environmental_awareness_score"]

        +

        input_df["government_incentive_awareness"]

        +

        input_df["technology_affinity_score"]

    ) / 3

    # ------------------------------------------------------
    # Drop Weak Features
    # ------------------------------------------------------

    weak_features = [

        "fuel_expense_per_month",
        "monthly_energy_consumption_kwh",
        "monthly_charging_cost",
        "daily_commute_km",
        "weekly_travel_distance_km",
        "vehicle_age_years",
        "electricity_cost_per_kwh"

    ]

    input_df.drop(
        columns=weak_features,
        inplace=True
    )

    # ------------------------------------------------------
    # One Hot Encoding
    # ------------------------------------------------------

    input_df = pd.get_dummies(

        input_df,

        columns=[

            "education_level",

            "city_type",

            "current_vehicle_type"

        ],

        drop_first=True

    )

    # ------------------------------------------------------
    # Match Training Columns
    # ------------------------------------------------------

    input_df = input_df.reindex(

        columns=feature_columns,

        fill_value=0

    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    confidence = probabilities.max() * 100

    label_map = {

        0: "Low",

        1: "Medium",

        2: "High"

    }

    predicted_label = label_map[prediction]

    # ------------------------------------------------------
    # Prediction Card
    # ------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        left, right = st.columns([1, 1])

        with left:

            if predicted_label == "High":

                st.success("🟢 High EV Adoption Likelihood")

            elif predicted_label == "Medium":

                st.warning("🟡 Medium EV Adoption Likelihood")

            else:

                st.error("🔴 Low EV Adoption Likelihood")

            st.progress(int(confidence))

            st.caption(
                f"Prediction Confidence: {confidence:.2f}%"
            )

        with right:

            k1, k2 = st.columns(2)

            kpi(
                k1,
                "⚡",
                "Prediction",
                predicted_label
            )

            kpi(
                k2,
                "🎯",
                "Confidence",
                f"{confidence:.1f}%"
            )

    # ------------------------------------------------------
    # Probability Chart
    # ------------------------------------------------------

    st.markdown("### Class Probabilities")

    prob_df = pd.DataFrame({

        "Likelihood": [

            "Low",

            "Medium",

            "High"

        ],

        "Probability": probabilities * 100

    })

    fig = px.bar(

        prob_df,

        x="Likelihood",

        y="Probability",

        text="Probability",

        color="Likelihood",

        color_discrete_map={

            "Low": "#EF4444",

            "Medium": "#F59E0B",

            "High": MINT

        },

        template=PLOT_TEMPLATE

    )

    fig.update_traces(

        texttemplate="%{text:.1f}%",

        textposition="outside"

    )

    fig = chart_layout(

        fig,

        showlegend=False

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------

    st.markdown("### Recommendation")

    if predicted_label == "High":

        st.success("""

The customer is highly likely to adopt an EV.

Reasons:

• Strong environmental awareness

• Good charging accessibility

• High EV knowledge

• Positive technology affinity

Keep promoting home charging and government incentives.

""")

    elif predicted_label == "Medium":

        st.info("""

The customer has a moderate likelihood of adopting an EV.

Recommendations:

• Improve EV awareness

• Increase charging accessibility

• Highlight long-term fuel savings

• Educate about government incentives

""")

    else:

        st.error("""

The customer currently has a low likelihood of adopting an EV.

Recommendations:

• Reduce range anxiety

• Improve charging infrastructure

• Increase EV awareness

• Educate about battery life and incentives

""")


# ==================================================================
# PAGE 6: AI ASSISTANT
# ==================================================================
with tab_ai:
    page_header("EV AI Assistant", "Ask anything about EV adoption — answers reflect your current sidebar filters.")

@st.cache_data
def build_context(data):

    if data.empty:
        return "No records match the current filters."

    adoption_rate = (data[TARGET] == "High").mean() * 100

    return f"""
Records in current view: {len(data):,}

High EV Adoption Likelihood: {adoption_rate:.1f}%

Average Annual Income: ${data["annual_income"].mean():,.0f}

Average Awareness Composite:
{data["awareness_composite"].mean():.2f}

Average Range Anxiety Score:
{data["range_anxiety_score"].mean():.2f}

Average Charging Accessibility:
{data["charging_station_accessibility"].mean():.2f}

Previous EV Experience:
{data["previous_ev_experience"].value_counts().to_dict()}

Home Charging Availability:
{data["home_charging_available"].value_counts().to_dict()}

Education Distribution:
{data["education_level"].value_counts().to_dict()}

City Distribution:
{data["city_type"].value_counts().to_dict()}

Current Vehicle Types:
{data["current_vehicle_type"].value_counts().to_dict()}

Top CatBoost Features:

1. anxiety_minus_knowledge

2. awareness_composite

3. charging_station_accessibility

4. battery_replacement_concern

5. home_charging_available

6. environmental_awareness_score

7. previous_ev_experience

8. nearest_charging_station_km

9. range_anxiety_score

10. technology_affinity_score
""".strip()

with st.form("ai_form"):
        question = st.text_area("Your question")
        ask = st.form_submit_button("Ask AI")

if ask:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            context = build_context(fdf)
            prompt = f"""
You are an AI EV Adoption Consultant.

Answer ONLY using the dashboard summary below.

If the user asks for insights:

- Explain the dashboard.
- Explain EV adoption behaviour.
- Explain why certain factors influence EV adoption.
- Explain the CatBoost prediction.
- Explain feature importance.
- Explain charging accessibility.
- Explain awareness composite.

If the user asks something unrelated to EV adoption or the dashboard, politely say you can only answer EV-related dashboard questions.

Dashboard Summary

{context}

User Question

{question}
"""



            models_to_try = [
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-chat-v3.1:free",
                "google/gemma-4-31b-it:free",
                "openai/gpt-oss-20b:free",
            ]

            with st.spinner("Thinking..."):
                answer, last_error = None, None
                for m in models_to_try:
                    try:
                        response = client.chat.completions.create(
                            model=m, messages=[{"role": "user", "content": prompt}]
                        )
                        answer = response.choices[0].message.content
                        break
                    except Exception as e:
                        last_error = e
                        continue

                if answer:
                    st.success(answer)
                else:
                    st.error(f"All models are currently rate-limited. Try again shortly. ({last_error})")