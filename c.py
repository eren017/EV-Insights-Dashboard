import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from catboost import Pool
from openai import OpenAI

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="EV Adoption Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# THEME TOKENS — dark luxury BI: black canvas, neon mint accent
# (same system as EV Insights v1 — colors pull from Streamlit's own
# theme in .streamlit/config.toml, this stylesheet layers cards/type on top)
# ------------------------------------------------------------------
INK = "#0B0B0B"
CARD_GLASS = "rgba(255,255,255,0.035)"
CARD_SOLID = "#111417"
BORDER_MINT = "rgba(82,242,198,0.22)"
MINT = "#52F2C6"
MINT_DIM = "#2FBE97"
WHITE = "#FFFFFF"
GRAY_LIGHT = "#B8BCC4"
GRAY_MID = "#6B7280"
ORANGE = "#FF8C42"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: radial-gradient(circle at top, #121517 0%, {INK} 55%); }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1360px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}

    section[data-testid="stSidebar"] {{
        background-color: {INK}; border-right: 1px solid {BORDER_MINT};
        box-shadow: inset -50px 0 70px -60px rgba(82,242,198,0.15);
    }}
    section[data-testid="stSidebar"] .block-container {{ padding: 1.6rem 1.1rem; }}
    section[data-testid="stSidebar"] h3 {{ color: {WHITE} !important; font-weight: 800; }}
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p {{ color: {GRAY_LIGHT} !important; font-weight: 600; }}

    .dashboard-card, [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {CARD_GLASS} !important; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        border: 1px solid {BORDER_MINT} !important; border-radius: 22px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.55) !important; padding: 8px !important;
        transition: box-shadow 0.25s ease; overflow: visible !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 10px 40px rgba(82,242,198,0.12), 0 0 0 1px {BORDER_MINT} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.02) !important; border-radius: 18px !important;
        margin-bottom: 14px; padding: 14px 14px 20px 14px !important;
    }}
    [data-testid="stForm"] {{
        background: {CARD_GLASS} !important; backdrop-filter: blur(14px);
        border: 1px solid {BORDER_MINT} !important; border-radius: 22px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.55) !important; padding: 22px !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] label, [data-testid="stForm"] label {{ color: {WHITE} !important; }}

    .stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div,
    .stTextArea textarea, [data-baseweb="input"] {{
        background-color: rgba(255,255,255,0.05) !important; border: 1px solid {BORDER_MINT} !important;
        border-radius: 12px !important; color: {WHITE} !important;
    }}
    .stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within {{
        border-color: {MINT} !important; box-shadow: 0 0 0 3px rgba(82,242,198,0.18) !important;
    }}

    .stMultiSelect [data-baseweb="tag"] {{
        background-color: rgba(82,242,198,0.18) !important; border: 1px solid rgba(82,242,198,0.5) !important;
        border-radius: 999px !important;
    }}
    .stMultiSelect [data-baseweb="tag"] span, .stMultiSelect [data-baseweb="tag"] * {{ color: {MINT} !important; font-weight: 700; }}
    .stMultiSelect [data-baseweb="tag"] svg {{ fill: {MINT} !important; }}

    .stSlider, .stSelectSlider {{ overflow: visible !important; padding-top: 6px; margin-bottom: 6px; }}
    .stSlider [role="slider"] {{
        background-color: {MINT} !important; border: 3px solid {INK} !important;
        box-shadow: 0 0 10px rgba(82,242,198,0.7) !important;
    }}
    [data-testid="stThumbValue"] {{
        opacity: 1 !important; visibility: visible !important; background-color: {MINT} !important;
        color: {INK} !important; font-weight: 800 !important; border-radius: 8px !important; padding: 2px 9px !important;
    }}
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: {GRAY_LIGHT} !important; }}

    .stButton>button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {{
        background-color: {MINT} !important; color: {INK} !important; border-radius: 999px !important;
        font-weight: 800 !important; border: none !important; padding: 0.55rem 1.4rem !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }}
    .stButton>button *, [data-testid="stFormSubmitButton"] button * {{ color: {INK} !important; }}
    .stButton>button:hover, [data-testid="stFormSubmitButton"] button:hover {{
        box-shadow: 0 0 22px rgba(82,242,198,0.55) !important; transform: translateY(-1px);
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; border-bottom: none; margin-bottom: 22px; flex-wrap: wrap; }}
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

    .metric-card {{
        position: relative; background: {CARD_GLASS}; backdrop-filter: blur(14px);
        border: 1px solid {BORDER_MINT}; border-left: 4px solid {MINT}; border-radius: 20px;
        padding: 18px 20px; min-height: 118px; box-shadow: 0 8px 28px rgba(0,0,0,0.5);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 16px 40px rgba(82,242,198,0.18), 0 0 0 1px {BORDER_MINT}; }}
    .metric-icon {{ font-size: 1.2rem; margin-bottom: 6px; color: {MINT}; font-weight: 900; }}
    .metric-icon.neg {{ color: {ORANGE}; }}
    .metric-label {{ color: {GRAY_LIGHT}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }}
    .metric-value {{ color: {WHITE}; font-size: 1.8rem; font-weight: 900; line-height: 1.1; }}
    .metric-sub {{ color: {MINT}; font-size: 0.74rem; margin-top: 6px; font-weight: 700; }}
    .metric-sub.neg {{ color: {ORANGE}; }}

    .chart-card-title {{ color: {WHITE}; font-size: 0.98rem; font-weight: 700; margin: 8px 0 10px 10px; }}
    .chart-card-title span {{ color: {MINT}; }}

    .hero-title {{ font-size: 2.2rem; font-weight: 900; color: {WHITE}; letter-spacing: -0.01em; margin-bottom: 4px; }}
    .hero-title span {{ color: {MINT}; }}
    .hero-sub {{ color: {GRAY_MID}; font-size: 0.95rem; margin-bottom: 14px; font-weight: 500; }}

    .navbar {{ display: flex; align-items: center; justify-content: space-between; padding: 6px 4px 20px 4px; }}
    .navbar-brand {{ font-size: 1.6rem; font-weight: 900; color: {WHITE}; }}
    .navbar-brand span {{ color: {MINT}; }}
    .navbar-sub {{ color: {GRAY_MID}; font-size: 0.82rem; font-weight: 600; }}

    .filter-pill {{
        display: inline-block; background: rgba(82,242,198,0.1); color: {MINT};
        border: 1px solid {BORDER_MINT}; border-radius: 999px; padding: 5px 16px;
        font-size: 0.8rem; font-weight: 700; margin-bottom: 18px;
    }}
    .model-page-pill {{
        display: inline-block; background: rgba(255,140,66,0.1); color: {ORANGE};
        border: 1px solid rgba(255,140,66,0.35); border-radius: 999px; padding: 5px 16px;
        font-size: 0.8rem; font-weight: 700; margin-bottom: 18px;
    }}

    .stAlert {{ background-color: rgba(255,255,255,0.04) !important; border-radius: 16px !important; border: 1px solid {BORDER_MINT} !important; color: {WHITE} !important; }}
    .stProgress > div > div > div > div {{ background-color: {MINT} !important; }}
    [data-testid="stDataFrame"] {{ border-radius: 16px; overflow: hidden; border: 1px solid {BORDER_MINT}; }}

    .verdict-badge {{
        display: inline-block; padding: 10px 22px; border-radius: 999px; font-weight: 900;
        font-size: 1.1rem; margin-bottom: 10px;
    }}
    .verdict-high {{ background: {MINT}; color: {INK}; box-shadow: 0 0 24px rgba(82,242,198,0.5); }}
    .verdict-medium {{ background: {MINT_DIM}; color: {INK}; box-shadow: 0 0 24px rgba(47,190,151,0.4); }}
    .verdict-low {{ background: {ORANGE}; color: {INK}; box-shadow: 0 0 24px rgba(255,140,66,0.4); }}

    .factor-row {{
        display: flex; justify-content: space-between; padding: 10px 14px; border-radius: 12px;
        background: rgba(255,255,255,0.03); margin-bottom: 8px; border-left: 3px solid {MINT};
    }}
    .factor-row.neg {{ border-left-color: {ORANGE}; }}
    .factor-name {{ color: {WHITE}; font-weight: 600; font-size: 0.88rem; }}
    .factor-impact {{ font-weight: 800; font-size: 0.88rem; }}
    .factor-impact.pos {{ color: {MINT}; }}
    .factor-impact.negimpact {{ color: {ORANGE}; }}
</style>
""", unsafe_allow_html=True)

ACCENT = MINT
PALETTE_NEUTRAL = [MINT, GRAY_LIGHT, MINT_DIM, GRAY_MID]
PALETTE_SEVERITY = [GRAY_MID, MINT_DIM, ORANGE]
PLOT_TEMPLATE = "plotly_dark"
CH = 300
TARGET = "ev_adoption_likelihood"
TARGET_ORDER = ["Low", "Medium", "High"]
TARGET_COLORS = {"Low": GRAY_MID, "Medium": MINT_DIM, "High": MINT}

# ------------------------------------------------------------------
# DATA + MODEL
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("global_ev_adoption_behavior_2026.csv")
    df["education_level"] = df["education_level"].fillna("Unknown")
    df["charging_station_accessibility"] = df["charging_station_accessibility"].fillna(
        df["charging_station_accessibility"].median()
    )
    df["ev_knowledge_score"] = df["ev_knowledge_score"].fillna(df["ev_knowledge_score"].median())

    df["Income_Bracket"] = pd.cut(
        df["annual_income"], bins=[0, 25000, 40000, 60000, 90000, float("inf")],
        labels=["<25k", "25k-40k", "40k-60k", "60k-90k", "90k+"]
    )
    df["Age_Group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    )

    def band(s):
        return pd.cut(s, bins=[0, 3, 7, 10], labels=["Low", "Medium", "High"], include_lowest=True)

    df["Env_Awareness_Band"] = band(df["environmental_awareness_score"])
    df["Tech_Affinity_Band"] = band(df["technology_affinity_score"])
    df["Gov_Incentive_Band"] = band(df["government_incentive_awareness"])
    df["Charging_Access_Band"] = band(df["charging_station_accessibility"])
    df["Range_Anxiety_Band"] = band(df["range_anxiety_score"])
    df["Distance_Band"] = pd.cut(
        df["nearest_charging_station_km"], bins=[-0.1, 2, 5, 10, float("inf")],
        labels=["<2 km", "2-5 km", "5-10 km", "10+ km"]
    )

    # same two derived features used in the Prediction page's model input,
    # added here as real columns so the AI Assistant can summarize them too
    df["awareness_composite"] = df[
        ["environmental_awareness_score", "technology_affinity_score", "government_incentive_awareness"]
    ].mean(axis=1)
    df["anxiety_minus_knowledge"] = df["range_anxiety_score"] - df["ev_knowledge_score"]
    return df

@st.cache_resource
def load_model():
    return joblib.load("best_ev_adoption_model.pkl")

@st.cache_resource
def load_llm_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])

df = load_data()
model = load_model()
client = load_llm_client()
CLASS_MAP = {0: "Low", 1: "Medium", 2: "High"}

def kpi(col, icon, label, value, sub=None, sub_neg=False):
    sub_html = f"<div class='metric-sub{' neg' if sub_neg else ''}'>{sub}</div>" if sub else ""
    icon_cls = "metric-icon neg" if sub_neg else "metric-icon"
    col.markdown(
        f"""<div class="metric-card"><div class="{icon_cls}">{icon}</div>
        <div class="metric-label">{label}</div><div class="metric-value">{value}</div>{sub_html}</div>""",
        unsafe_allow_html=True,
    )

def chart_layout(fig, height=CH, **kw):
    fig.update_layout(paper_bgcolor=CARD_SOLID, plot_bgcolor=CARD_SOLID, font_color=GRAY_LIGHT,
                       height=height, margin=dict(t=10, b=10, l=10, r=10), **kw)
    return fig

def stacked_mix_chart(data, col, title, order=None, key=None):
    with st.container(border=True):
        st.markdown(f'<div class="chart-card-title">{title}</div>', unsafe_allow_html=True)
        if data.empty:
            st.info("No records match the current filters.")
            return
        ct = pd.crosstab(data[col], data["ev_adoption_likelihood"], normalize="index")[TARGET_ORDER] * 100
        if order:
            ct = ct.reindex(order)
        fig = go.Figure()
        for t in TARGET_ORDER:
            fig.add_bar(name=t, x=ct.index.astype(str), y=ct[t], marker_color=TARGET_COLORS[t])
        fig.update_layout(barmode="stack", legend_title="")
        fig = chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True, key=key or f"mix_{col}")

def adoption_rate_by_band(data, col, title, order=("Low", "Medium", "High"), key=None):
    with st.container(border=True):
        st.markdown(f'<div class="chart-card-title">{title}</div>', unsafe_allow_html=True)
        if data.empty:
            st.info("No records match the current filters.")
            return
        tmp = data.groupby(col, observed=True)["ev_adoption_likelihood"].apply(lambda s: (s == "High").mean() * 100)
        tmp = tmp.reindex(order).reset_index()
        tmp.columns = [col, "High Adoption Rate (%)"]
        fig = px.bar(tmp, x=col, y="High Adoption Rate (%)", text="High Adoption Rate (%)",
                     color=col, color_discrete_sequence=PALETTE_SEVERITY, template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = chart_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=key or f"band_{col}")

def page_header(title, subtitle, model_page=False):
    parts = title.split(" ")
    accented = f'{" ".join(parts[:-1])} <span>{parts[-1]}</span>' if len(parts) > 1 else f'<span>{title}</span>'
    st.markdown(f'<div class="hero-title">{accented}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{subtitle}</div>', unsafe_allow_html=True)
    if model_page:
        st.markdown('<div class="model-page-pill">◆ Model-level view — sidebar filters do not apply here</div>', unsafe_allow_html=True)
    else:
        pct = len(fdf) / len(df) * 100 if len(df) else 0
        st.markdown(
            f'<div class="filter-pill">◆ {len(fdf):,} of {len(df):,} respondents in view ({pct:.0f}%) — via sidebar filters</div>',
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------
# SIDEBAR — lean filter set, business pages only
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙ Filters")
    st.caption("Apply to Home, Demographics, Charging & EV Insights only — not to Feature Importance or Prediction.")

    with st.container(border=True):
        city_f = st.multiselect("🏙 City Type", sorted(df["city_type"].unique()), default=sorted(df["city_type"].unique()))
        edu_f = st.multiselect("🎓 Education Level", sorted(df["education_level"].unique()), default=sorted(df["education_level"].unique()))

    with st.container(border=True):
        income_f = st.select_slider(
            "💵 Income Bracket", options=["<25k", "25k-40k", "40k-60k", "60k-90k", "90k+"],
            value=("<25k", "90k+")
        )

    if st.button("↺ Reset All Filters", use_container_width=True):
        st.rerun()

income_order = ["<25k", "25k-40k", "40k-60k", "60k-90k", "90k+"]
lo_idx, hi_idx = income_order.index(income_f[0]), income_order.index(income_f[1])
income_range = income_order[lo_idx:hi_idx + 1]

fdf = df[
    df["city_type"].isin(city_f)
    & df["education_level"].isin(edu_f)
    & df["Income_Bracket"].isin(income_range)
]

# ------------------------------------------------------------------
# NAVBAR
# ------------------------------------------------------------------
st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">⚡ EV Adoption<span> Intelligence</span></div>
    <div class="navbar-sub">Powered by CatBoost · 50,000 respondents</div>
</div>
""", unsafe_allow_html=True)

tab_home, tab_demo, tab_charge, tab_insights, tab_importance, tab_predict, tab_ai = st.tabs([
    "🏠 Home", "👥 Demographics", "🔌 Charging Infrastructure",
    "🚗 EV Insights", "🧠 Feature Importance", "🤖 Prediction", "💬 AI Assistant"
])

# ==================================================================
# PAGE 1: HOME
# ==================================================================
with tab_home:
    page_header("Executive Overview", "The state of EV adoption across your surveyed population, at a glance.")

    if not fdf.empty:
        adoption_rate = (fdf["ev_adoption_likelihood"] == "High").mean() * 100
        high_anxiety_pct = (fdf["range_anxiety_score"] >= 7).mean() * 100
        home_charge_pct = fdf["home_charging_available"].mean() * 100
        avg_income = fdf["annual_income"].mean()
        avg_knowledge = fdf["ev_knowledge_score"].mean()

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi(k1, "◆", "Total Respondents", f"{len(fdf):,}")
        kpi(k2, "▲", "EV Adoption Rate", f"{adoption_rate:.1f}%", "Share rated 'High' likelihood")
        kpi(k3, "!", "High Range Anxiety", f"{high_anxiety_pct:.1f}%", "Score ≥ 7 / 10", sub_neg=True)
        kpi(k4, "●", "Home Charging Access", f"{home_charge_pct:.1f}%")
        kpi(k5, "$", "Avg Annual Income", f"${avg_income:,.0f}")
        kpi(k6, "◈", "Avg EV Knowledge Score", f"{avg_knowledge:.1f} / 10")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Adoption Likelihood Split</div>', unsafe_allow_html=True)
                tmp = fdf["ev_adoption_likelihood"].value_counts().reindex(TARGET_ORDER).reset_index()
                tmp.columns = ["Likelihood", "Count"]
                fig = px.pie(tmp, names="Likelihood", values="Count", hole=0.62,
                             color="Likelihood", color_discrete_map=TARGET_COLORS, template=PLOT_TEMPLATE)
                fig.update_traces(textinfo="percent+label")
                fig = chart_layout(fig, showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True, key="home_pie")
        with c2:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">What Separates Adopters From Skeptics</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                drivers = [
                    ("EV Knowledge Score", fdf[fdf.ev_adoption_likelihood=="High"]["ev_knowledge_score"].mean(),
                     fdf[fdf.ev_adoption_likelihood=="Low"]["ev_knowledge_score"].mean()),
                    ("Environmental Awareness", fdf[fdf.ev_adoption_likelihood=="High"]["environmental_awareness_score"].mean(),
                     fdf[fdf.ev_adoption_likelihood=="Low"]["environmental_awareness_score"].mean()),
                    ("Range Anxiety (lower is better)", fdf[fdf.ev_adoption_likelihood=="High"]["range_anxiety_score"].mean(),
                     fdf[fdf.ev_adoption_likelihood=="Low"]["range_anxiety_score"].mean()),
                ]
                for name, high_val, low_val in drivers:
                    st.markdown(f"**{name}**")
                    b1, b2 = st.columns(2)
                    b1.markdown(f"<span style='color:{MINT};font-weight:800;'>High adopters: {high_val:.1f}</span>", unsafe_allow_html=True)
                    b2.markdown(f"<span style='color:{ORANGE};font-weight:800;'>Low adopters: {low_val:.1f}</span>", unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)

# ==================================================================
# PAGE 2: DEMOGRAPHICS
# ==================================================================
with tab_demo:
    page_header("Buyer Demographics", "Who adopts EVs — broken down by age, income, and education.")

    if not fdf.empty:
        c1, c2 = st.columns(2)
        with c1:
            adoption_rate_by_band(
            fdf.assign(Battery_Concern_Band=pd.cut(
            fdf["battery_replacement_concern"], bins=[0, 3, 7, 10],
            labels=["Low", "Medium", "High"], include_lowest=True
        )),
            "Battery_Concern_Band", "Battery Replacement Concern vs Adoption", key="demo_battery"
    )
        with c2:
            stacked_mix_chart(fdf, "Income_Bracket", "Income Bracket vs Adoption Mix",
                               order=income_order, key="demo_income")

        c3, c4 = st.columns(2)
        with c3:
            stacked_mix_chart(fdf, "education_level", "Education Level vs Adoption Mix",
                               order=["High School", "Bachelor", "Master", "PhD", "Unknown"], key="demo_edu")
        with c4:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">City Type Distribution</div>', unsafe_allow_html=True)
                tmp = fdf["city_type"].value_counts().reset_index()
                tmp.columns = ["City Type", "Count"]
                fig = px.pie(tmp, names="City Type", values="Count", hole=0.55,
                             color_discrete_sequence=PALETTE_NEUTRAL, template=PLOT_TEMPLATE)
                fig.update_traces(textinfo="percent+label")
                fig = chart_layout(fig, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="demo_city_donut")

# ==================================================================
# PAGE 3: CHARGING INFRASTRUCTURE
# ==================================================================
with tab_charge:
    page_header("Charging Infrastructure", "How charging access and cost shape adoption.")

    if not fdf.empty:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Home Charging Available</div>', unsafe_allow_html=True)
                tmp = fdf["home_charging_available"].map({1: "Yes", 0: "No"}).value_counts().reset_index()
                tmp.columns = ["Home Charging", "Count"]
                fig = px.pie(tmp, names="Home Charging", values="Count", hole=0.6,
                             color="Home Charging", color_discrete_map={"Yes": MINT, "No": ORANGE}, template=PLOT_TEMPLATE)
                fig.update_traces(textinfo="percent+label")
                fig = chart_layout(fig, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="charge_home_donut")
        with c2:
            adoption_rate_by_band(fdf, "Charging_Access_Band", "Adoption Rate by Charging Accessibility", key="charge_access_band")

        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Distance to Nearest Charger</div>', unsafe_allow_html=True)
                tmp = fdf["Distance_Band"].value_counts().reindex(["<2 km", "2-5 km", "5-10 km", "10+ km"]).reset_index()
                tmp.columns = ["Distance", "Count"]
                fig = px.bar(tmp, x="Distance", y="Count", text="Count", color_discrete_sequence=[MINT], template=PLOT_TEMPLATE)
                fig.update_traces(textposition="outside")
                fig = chart_layout(fig)
                st.plotly_chart(fig, use_container_width=True, key="charge_distance_band")
        with c4:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Charging Cost vs Energy Consumption</div>', unsafe_allow_html=True)
                sample = fdf.sample(min(3000, len(fdf)), random_state=42)
                fig = px.scatter(sample, x="monthly_energy_consumption_kwh", y="monthly_charging_cost",
                                  color_discrete_sequence=[MINT], opacity=0.5, trendline="ols",
                                  trendline_color_override=ORANGE, template=PLOT_TEMPLATE)
                fig = chart_layout(fig, xaxis_title="Monthly Energy (kWh)", yaxis_title="Monthly Charging Cost ($)")
                st.plotly_chart(fig, use_container_width=True, key="charge_scatter")

# ==================================================================
# PAGE 4: EV INSIGHTS
# ==================================================================
with tab_insights:
    page_header("Adoption Drivers", "The psychological and awareness factors that move the needle most.")

    if not fdf.empty:
        c1, c2 = st.columns(2)
        with c1:
            adoption_rate_by_band(fdf, "Env_Awareness_Band", "Environmental Awareness vs Adoption", key="ins_env")
        with c2:
            adoption_rate_by_band(fdf, "Tech_Affinity_Band", "Technology Affinity vs Adoption", key="ins_tech")

        c3, c4 = st.columns(2)
        with c3:
            adoption_rate_by_band(fdf, "Gov_Incentive_Band", "Govt. Incentive Awareness vs Adoption", key="ins_gov")
        with c4:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Previous EV Experience vs Adoption</div>', unsafe_allow_html=True)
                tmp = fdf.groupby(fdf["previous_ev_experience"].map({1: "Yes", 0: "No"}))["ev_adoption_likelihood"].apply(
                    lambda s: (s == "High").mean() * 100
                ).reset_index()
                tmp.columns = ["Previous EV Experience", "High Adoption Rate (%)"]
                fig = px.bar(tmp, x="Previous EV Experience", y="High Adoption Rate (%)", text="High Adoption Rate (%)",
                             color="Previous EV Experience", color_discrete_map={"Yes": MINT, "No": ORANGE}, template=PLOT_TEMPLATE)
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig = chart_layout(fig, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="ins_prev_exp")

# ==================================================================
# PAGE 5: FEATURE IMPORTANCE
# ==================================================================
with tab_importance:
    page_header("Feature Importance", "What the trained CatBoost model actually relies on to predict adoption.")

    imp = model.get_feature_importance()
    imp_df = pd.DataFrame({"Feature": model.feature_names_, "Importance": imp}).sort_values("Importance", ascending=True).tail(15)

    with st.container(border=True):
        st.markdown('<div class="chart-card-title">Top 15 Features — CatBoost Importance Score</div>', unsafe_allow_html=True)
        colors = [MINT if i >= len(imp_df) - 3 else MINT_DIM if i >= len(imp_df) - 8 else GRAY_MID for i in range(len(imp_df))]
        fig = go.Figure(go.Bar(x=imp_df["Importance"], y=imp_df["Feature"], orientation="h", marker_color=colors))
        fig = chart_layout(fig, height=520, xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True, key="feat_importance")

    st.info("💡 The top 4 features — anxiety-vs-knowledge gap, awareness composite, charging accessibility, "
            "and battery replacement concern — account for roughly two-thirds of the model's total decision weight. "
            "Everything past the top 10 contributes marginally.")

# ==================================================================
# PAGE 6: PREDICTION
# ==================================================================
with tab_predict:
    page_header("EV Adoption Predictor", "Score an individual profile using the live CatBoost model.", model_page=True)

    with st.form("prediction_form"):
        st.markdown('<div class="chart-card-title">Respondent Profile</div>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Financial**")
            income = st.number_input("Annual Income ($)", min_value=5000, max_value=300000, value=45000, step=1000)
            fuel_expense = st.number_input("Fuel Expense per Month ($)", min_value=0, max_value=1000, value=250)
            charging_cost = st.number_input("Monthly Charging Cost ($, if EV)", min_value=0, max_value=500, value=40)
            energy_kwh = st.number_input("Monthly Energy Consumption (kWh, if EV)", min_value=1, max_value=2000, value=300)

        with col_b:
            st.markdown("**Commute & Vehicle**")
            daily_commute = st.number_input("Daily Commute (km)", min_value=0, max_value=200, value=30)
            weekly_travel = st.number_input("Weekly Travel Distance (km)", min_value=1, max_value=1500, value=210,
                                             help="Should be roughly 7x daily commute if travel is consistent day to day.")
            education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Unknown"])
            city = st.selectbox("City Type", ["Rural", "Suburban", "Urban"])
            vehicle = st.selectbox("Current Vehicle Type", ["Hatchback", "Sedan", "SUV", "Truck"])

        with col_c:
            st.markdown("**Awareness & Charging Access**")
            env_awareness = st.slider("Environmental Awareness", 1, 10, 6, help="1 = not aware at all, 10 = extremely aware")
            tech_affinity = st.slider("Technology Affinity", 1, 10, 6, help="1 = avoids new tech, 10 = early adopter")
            gov_awareness = st.slider("Government Incentive Awareness", 1, 10, 6, help="1 = unaware of subsidies/tax breaks, 10 = fully informed")
            range_anxiety = st.slider("Range Anxiety", 1, 10, 5, help="1 = no concern about running out of charge, 10 = very anxious")
            battery_concern = st.slider("Battery Replacement Concern", 1, 10, 5, help="1 = not worried about battery lifespan/cost, 10 = major concern")
            ev_knowledge = st.slider("EV Knowledge Score", 1, 10, 6, help="1 = knows very little about EVs, 10 = highly knowledgeable")
            charging_access = st.slider("Charging Station Accessibility", 1, 10, 5, help="1 = very hard to find chargers nearby, 10 = chargers everywhere")
            nearest_km = st.number_input("Distance to Nearest Charger (km)", min_value=0.0, max_value=100.0, value=6.0)
            home_charging = st.selectbox("Home Charging Available", ["Yes", "No"])
            prev_experience = st.selectbox("Previous EV Experience", ["Yes", "No"])

        submitted = st.form_submit_button("🔮 Predict Adoption Likelihood", use_container_width=True)

    # ---- sanity check on commute consistency, shown regardless of submit ----
    expected_weekly = daily_commute * 7 if 'daily_commute' in dir() else None

    if submitted:
        try:
            # soft data-quality warning — doesn't block prediction, just flags it
            if weekly_travel > 0 and abs(daily_commute * 7 - weekly_travel) / weekly_travel > 0.5:
                st.warning(
                    f"⚠️ Daily commute × 7 ({daily_commute*7:.0f} km) is quite different from weekly travel "
                    f"distance ({weekly_travel} km) — double check these are consistent before trusting the result."
                )

            row = {
                "annual_income": income,
                "charging_station_accessibility": charging_access,
                "nearest_charging_station_km": nearest_km,
                "home_charging_available": 1 if home_charging == "Yes" else 0,
                "environmental_awareness_score": env_awareness,
                "government_incentive_awareness": gov_awareness,
                "technology_affinity_score": tech_affinity,
                "range_anxiety_score": range_anxiety,
                "battery_replacement_concern": battery_concern,
                "ev_knowledge_score": ev_knowledge,
                "previous_ev_experience": 1 if prev_experience == "Yes" else 0,
                "log_annual_income": np.log1p(income),
                "log_monthly_charging_cost": np.log1p(charging_cost),
                "fuel_cost_to_income_ratio": (fuel_expense * 12) / income,
                "charging_cost_per_kwh_actual": charging_cost / energy_kwh,
                "commute_consistency": (daily_commute * 7) / weekly_travel,
                "anxiety_minus_knowledge": range_anxiety - ev_knowledge,
                "awareness_composite": np.mean([env_awareness, tech_affinity, gov_awareness]),
                "education_level_High School": 1 if education == "High School" else 0,
                "education_level_Master": 1 if education == "Master" else 0,
                "education_level_PhD": 1 if education == "PhD" else 0,
                "education_level_Unknown": 1 if education == "Unknown" else 0,
                "city_type_Suburban": 1 if city == "Suburban" else 0,
                "city_type_Urban": 1 if city == "Urban" else 0,
                "current_vehicle_type_SUV": 1 if vehicle == "SUV" else 0,
                "current_vehicle_type_Sedan": 1 if vehicle == "Sedan" else 0,
                "current_vehicle_type_Truck": 1 if vehicle == "Truck" else 0,
            }
            input_df = pd.DataFrame([row])[model.feature_names_]

            pred_idx = int(model.predict(input_df)[0][0])
            pred_label = CLASS_MAP[pred_idx]
            proba = model.predict_proba(input_df)[0]

            pool = Pool(input_df)
            shap_vals = model.get_feature_importance(pool, type="ShapValues")
            contribs = shap_vals[0][pred_idx][:-1]
            top_idx = np.argsort(np.abs(contribs))[::-1][:6]

            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                r1, r2 = st.columns([1, 1.3])

                with r1:
                    badge_cls = {"Low": "verdict-low", "Medium": "verdict-medium", "High": "verdict-high"}[pred_label]
                    st.markdown(f'<div class="verdict-badge {badge_cls}">Adoption Likelihood: {pred_label}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    for cls, p in zip(["Low", "Medium", "High"], proba):
                        st.markdown(f"**{cls}** — {p*100:.1f}%")
                        st.progress(float(p))

                    # ---- export this prediction ----
                    export_row = row.copy()
                    export_row["predicted_likelihood"] = pred_label
                    export_row["prob_low"], export_row["prob_medium"], export_row["prob_high"] = proba
                    export_df = pd.DataFrame([export_row])
                    st.download_button(
                        "⬇ Export This Prediction (CSV)",
                        export_df.to_csv(index=False).encode("utf-8"),
                        "ev_prediction.csv", "text/csv", use_container_width=True
                    )

                with r2:
                    st.markdown("**Top factors driving this prediction**")
                    fig = go.Figure(go.Bar(
                        x=[contribs[i] for i in top_idx][::-1],
                        y=[model.feature_names_[i].replace("_", " ").title() for i in top_idx][::-1],
                        orientation="h",
                        marker_color=[MINT if contribs[i] > 0 else ORANGE for i in top_idx][::-1],
                    ))
                    fig = chart_layout(fig, height=280, xaxis_title="SHAP contribution (→ pushes toward predicted class)")
                    st.plotly_chart(fig, use_container_width=True, key="pred_shap_bar")

                    # ---- percentile context for top factors that map to real columns ----
                    with st.expander("See where these values sit vs. all respondents"):
                        for i in top_idx[:4]:
                            fname = model.feature_names_[i]
                            if fname in row and fname in df.columns:
                                pct = (df[fname] < row[fname]).mean() * 100
                                st.markdown(f"- **{fname.replace('_',' ').title()}** ({row[fname]:.2f}) is higher than **{pct:.0f}%** of respondents")

            # ---- sensitivity: "what would change this prediction" ----
            LEVERS = {
                "ev_knowledge_score": ("ev_knowledge_score", +2, 1, 10),
                "range_anxiety_score": ("range_anxiety_score", -2, 1, 10),
                "environmental_awareness_score": ("environmental_awareness_score", +2, 1, 10),
                "technology_affinity_score": ("technology_affinity_score", +2, 1, 10),
                "government_incentive_awareness": ("government_incentive_awareness", +2, 1, 10),
                "charging_station_accessibility": ("charging_station_accessibility", +2, 1, 10),
                "battery_replacement_concern": ("battery_replacement_concern", -2, 1, 10),
                "anxiety_minus_knowledge": ("ev_knowledge_score", +2, 1, 10),
                "awareness_composite": ("environmental_awareness_score", +2, 1, 10),
            }
            negative_idx = [i for i in top_idx if contribs[i] < 0]
            suggestions = []
            for i in negative_idx[:2]:
                fname = model.feature_names_[i]
                if fname in LEVERS:
                    raw_field, delta, lo, hi = LEVERS[fname]
                    new_val = int(np.clip(row.get(raw_field, ev_knowledge) + delta, lo, hi))
                    sim_inputs = dict(
                        income=income, fuel_expense=fuel_expense, charging_cost=charging_cost, energy_kwh=energy_kwh,
                        daily_commute=daily_commute, weekly_travel=weekly_travel, education=education, city=city,
                        vehicle=vehicle, env_awareness=env_awareness, tech_affinity=tech_affinity,
                        gov_awareness=gov_awareness, range_anxiety=range_anxiety, battery_concern=battery_concern,
                        ev_knowledge=ev_knowledge, charging_access=charging_access, nearest_km=nearest_km,
                        home_charging=home_charging, prev_experience=prev_experience,
                    )
                    field_to_var = {
                        "ev_knowledge_score": "ev_knowledge", "range_anxiety_score": "range_anxiety",
                        "environmental_awareness_score": "env_awareness", "technology_affinity_score": "tech_affinity",
                        "government_incentive_awareness": "gov_awareness",
                        "charging_station_accessibility": "charging_access",
                        "battery_replacement_concern": "battery_concern",
                    }
                    sim_inputs[field_to_var[raw_field]] = new_val
                    sim_row = row.copy()
                    sim_row[raw_field] = new_val
                    sim_row["anxiety_minus_knowledge"] = sim_inputs["range_anxiety"] - sim_inputs["ev_knowledge"]
                    sim_row["awareness_composite"] = np.mean(
                        [sim_inputs["env_awareness"], sim_inputs["tech_affinity"], sim_inputs["gov_awareness"]]
                    )
                    sim_df = pd.DataFrame([sim_row])[model.feature_names_]
                    sim_pred = CLASS_MAP[int(model.predict(sim_df)[0][0])]
                    if sim_pred != pred_label:
                        suggestions.append(
                            f"If **{raw_field.replace('_',' ')}** moved from {row.get(raw_field)} to **{new_val}**, "
                            f"the prediction would likely shift from **{pred_label} → {sim_pred}**."
                        )
                    else:
                        suggestions.append(
                            f"Even raising **{raw_field.replace('_',' ')}** to {new_val} keeps the prediction at **{pred_label}** — not the deciding factor alone."
                        )

            if suggestions:
                st.markdown("**What would change this prediction**")
                for s in suggestions:
                    st.markdown(f"- {s}")

            # ---- prediction history ----
            if "pred_history" not in st.session_state:
                st.session_state.pred_history = []
            st.session_state.pred_history.append({
                "Income": income, "Knowledge": ev_knowledge, "Anxiety": range_anxiety,
                "Awareness (env)": env_awareness, "Charging Access": charging_access,
                "Prediction": pred_label, "P(High)": f"{proba[2]*100:.1f}%"
            })
            st.session_state.pred_history = st.session_state.pred_history[-5:]

        except Exception as e:
            st.error(f"Something went wrong computing this prediction: {e}")

    if st.session_state.get("pred_history"):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="chart-card-title">Recent Predictions (this session)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state.pred_history), use_container_width=True, hide_index=True)
            if st.button("🗑 Clear History"):
                st.session_state.pred_history = []
                st.rerun()

# ==================================================================
# PAGE 7: AI ASSISTANT
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

Average Annual Income: ${data['annual_income'].mean():,.0f}

Average Awareness Composite: {data['awareness_composite'].mean():.2f}

Average Range Anxiety Score: {data['range_anxiety_score'].mean():.2f}

Average Charging Accessibility: {data['charging_station_accessibility'].mean():.2f}

Previous EV Experience (1=Yes, 0=No): {data['previous_ev_experience'].value_counts().to_dict()}

Home Charging Availability (1=Yes, 0=No): {data['home_charging_available'].value_counts().to_dict()}

Education Distribution: {data['education_level'].value_counts().to_dict()}

City Distribution: {data['city_type'].value_counts().to_dict()}

Current Vehicle Types: {data['current_vehicle_type'].value_counts().to_dict()}

Top CatBoost Features (in order of importance):
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
            prompt = f"""You are an AI EV Adoption Consultant.

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
{question}"""

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