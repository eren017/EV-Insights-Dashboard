import streamlit as st
import pandas as pd
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
# GLOBAL STYLE
# ------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1300px; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar = filter panel */
    section[data-testid="stSidebar"] {
        background-color: #12151d; border-right: 1px solid #262d3d;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #f5f7fa; }

    /* Navbar */
    .navbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 4px 18px 4px; border-bottom: 1px solid #262d3d; margin-bottom: 6px;
    }
    .navbar-brand { font-size: 1.35rem; font-weight: 800; color: #f5f7fa; }
    .navbar-brand span { color: #5cd6a5; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; border-bottom: 1px solid #262d3d; margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px; background-color: transparent; border-radius: 8px 8px 0 0;
        color: #8b93a7; font-weight: 600; padding: 0 18px;
    }
    .stTabs [aria-selected="true"] { color: #5cd6a5 !important; border-bottom: 2px solid #5cd6a5; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2b 0%, #151a24 100%);
        border: 1px solid #262d3d; border-radius: 14px; padding: 16px 18px;
    }
    .kpi-label { color: #8b93a7; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
    .kpi-value { color: #f5f7fa; font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
    .kpi-sub { color: #5cd6a5; font-size: 0.75rem; margin-top: 4px; font-weight: 500; }
    .kpi-sub.neg { color: #ff6b6b; }

    /* Hero */
    .hero-title { font-size: 2.1rem; font-weight: 800; color: #f5f7fa; margin-bottom: 2px; }
    .hero-sub { color: #8b93a7; font-size: 0.95rem; margin-bottom: 10px; }
    .section-title { color: #f5f7fa; font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }

    /* Filter status pill shown at top of every page */
    .filter-pill {
        display: inline-block; background: #1a2b24; color: #5cd6a5;
        border: 1px solid #2a4a3d; border-radius: 999px; padding: 4px 14px;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 16px;
    }

    h1, h2, h3 { color: #f5f7fa !important; }
</style>
""", unsafe_allow_html=True)

ACCENT = "#5cd6a5"
PALETTE = ["#5cd6a5", "#4f8cff", "#ffb454", "#ff6b6b", "#a78bfa", "#38bdf8"]
PLOT_TEMPLATE = "plotly_dark"
CH = 300  # standard chart height so pages fit without excess scrolling

# ------------------------------------------------------------------
# DATA + MODEL LOADING
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("EV_Adoption_and_Range_Anxiety_Dataset.csv")
    df["Annual_Income_USD"] = df["Annual_Income_USD"].fillna(df["Annual_Income_USD"].median())
    df["Daily_Commute_km"] = df["Daily_Commute_km"].fillna(df["Daily_Commute_km"].median())
    df["Environmental_Concern_Level"] = df["Environmental_Concern_Level"].fillna(
        df["Environmental_Concern_Level"].median()
    )
    df["Range_Anxiety_Rank"] = df["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    df["Income_Bracket"] = pd.cut(
        df["Annual_Income_USD"], bins=[0, 40000, 70000, 100000, 150000, float("inf")],
        labels=["<40k", "40k-70k", "70k-100k", "100k-150k", "150k+"]
    )
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    )
    return df

@st.cache_resource
def load_model():
    return joblib.load("ev_xgboost.pkl")

@st.cache_resource
def load_llm_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])

df = load_data()
model = load_model()
client = load_llm_client()

# ------------------------------------------------------------------
# SMALL REUSABLE UI HELPERS (keeps chart code short + consistent)
# ------------------------------------------------------------------
def kpi(col, label, value, sub=None, sub_neg=False):
    sub_html = f"<div class='kpi-sub{' neg' if sub_neg else ''}'>{sub}</div>" if sub else ""
    col.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>{sub_html}</div>""",
        unsafe_allow_html=True,
    )

def purchase_rate_by(data, col, title, order=None, color_seq=PALETTE):
    """Bar chart: % who will buy an EV, grouped by any categorical column."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if data.empty:
        st.info("No records match the current filters.")
        return
    tmp = data.groupby(col, observed=True)["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100)
    if order:
        tmp = tmp.reindex(order)
    tmp = tmp.reset_index()
    tmp.columns = [col, "Rate (%)"]
    fig = px.bar(tmp, x=col, y="Rate (%)", text="Rate (%)", color=col,
                 color_discrete_sequence=color_seq, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH)
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{col}_{title}")

def page_header(title, subtitle):
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{subtitle}</div>', unsafe_allow_html=True)
    pct = len(fdf) / len(df) * 100 if len(df) else 0
    st.markdown(
        f'<div class="filter-pill">🔎 Showing {len(fdf):,} of {len(df):,} buyers ({pct:.0f}%) — based on sidebar filters</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# SIDEBAR — GLOBAL FILTERS (apply to every page)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Global Filters")
    st.caption("These apply across every page, including charts and predictions context.")

    city_f = st.multiselect("City Type", sorted(df["City_Type"].unique()), default=sorted(df["City_Type"].unique()))
    anxiety_f = st.multiselect("Range Anxiety", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    subsidy_f = st.select_slider("Subsidy Available", options=["All", "Yes", "No"], value="All")
    home_charge_f = st.select_slider("Home Charging Possible", options=["All", "Yes", "No"], value="All")
    age_f = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()),
                       (int(df["Age"].min()), int(df["Age"].max())))
    income_f = st.slider("Annual Income (USD)", int(df["Annual_Income_USD"].min()), int(df["Annual_Income_USD"].max()),
                          (int(df["Annual_Income_USD"].min()), int(df["Annual_Income_USD"].max())), step=1000)

    st.markdown("---")
    if st.button("↺ Reset All Filters", use_container_width=True):
        st.rerun()

# ---- Apply filters once, reuse everywhere as `fdf` ----
fdf = df[
    df["City_Type"].isin(city_f)
    & df["Range_Anxiety_Level"].isin(anxiety_f)
    & df["Age"].between(*age_f)
    & df["Annual_Income_USD"].between(*income_f)
]
if subsidy_f != "All":
    fdf = fdf[fdf["Subsidy_Available"] == subsidy_f]
if home_charge_f != "All":
    fdf = fdf[fdf["Home_Charging_Possible"] == home_charge_f]

# ------------------------------------------------------------------
# NAVBAR
# ------------------------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">⚡ EV<span>Insights</span></div>
    <div style="color:#8b93a7; font-size:0.85rem;">EV Adoption & Range Anxiety Study</div>
</div>
""", unsafe_allow_html=True)

tab_home, tab_demo, tab_charge, tab_buy, tab_predict, tab_ai, tab_data = st.tabs([
    "🏠 Home", "👥 Demographics", "🔌 Charging", "🛒 Buying",
    "🤖 Prediction", "💬 AI Assistant", "📄 Explore Data"
])

if fdf.empty:
    st.warning("⚠️ No records match your current sidebar filters — widen them to see data on any page.")

# ==================================================================
# PAGE 1: HOME
# ==================================================================
with tab_home:
    page_header("EV Adoption at a Glance", "A snapshot of who buys electric vehicles — and what holds the rest back.")

    if not fdf.empty:
        buy_rate = (fdf["Will_Buy_EV"] == "Yes").mean() * 100
        avg_income = fdf["Annual_Income_USD"].mean()
        high_anxiety_pct = (fdf["Range_Anxiety_Level"] == "High").mean() * 100
        home_charge_pct = (fdf["Home_Charging_Possible"] == "Yes").mean() * 100

        k1, k2, k3, k4, k5 = st.columns(5)
        kpi(k1, "Buyers in View", f"{len(fdf):,}")
        kpi(k2, "Will Buy an EV", f"{buy_rate:.1f}%")
        kpi(k3, "Avg Annual Income", f"${avg_income:,.0f}")
        kpi(k4, "High Range Anxiety", f"{high_anxiety_pct:.1f}%",
            "Biggest adoption blocker" if high_anxiety_pct > 20 else None, sub_neg=True)
        kpi(k5, "Home Charging Access", f"{home_charge_pct:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-title">Purchase Decision Split</div>', unsafe_allow_html=True)
            tmp = fdf["Will_Buy_EV"].value_counts().reset_index()
            tmp.columns = ["Decision", "Count"]
            fig = px.pie(tmp, names="Decision", values="Count", hole=0.6,
                         color="Decision", color_discrete_map={"Yes": ACCENT, "No": "#ff6b6b"},
                         template=PLOT_TEMPLATE)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="home_pie")
        with c2:
            purchase_rate_by(fdf, "Range_Anxiety_Level", "Purchase Rate by Range Anxiety", order=["Low", "Medium", "High"])
        with c3:
            purchase_rate_by(fdf, "City_Type", "Purchase Rate by City Type")

# ==================================================================
# PAGE 2: DEMOGRAPHICS
# ==================================================================
with tab_demo:
    page_header("Who Are the Buyers?", "Age, income, and household profile of survey respondents.")

    if not fdf.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-title">Purchase Rate by Age Group</div>', unsafe_allow_html=True)
            tmp = fdf.groupby("Age_Group", observed=True)["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            tmp.columns = ["Age Group", "Rate (%)"]
            fig = px.area(tmp, x="Age Group", y="Rate (%)", template=PLOT_TEMPLATE, color_discrete_sequence=[PALETTE[1]])
            fig.update_traces(line=dict(width=3))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
            st.plotly_chart(fig, use_container_width=True, key="demo_age_area")
        with c2:
            st.markdown('<div class="section-title">Purchase Rate by Income Bracket</div>', unsafe_allow_html=True)
            tmp = fdf.groupby("Income_Bracket", observed=True)["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            tmp.columns = ["Income Bracket", "Rate (%)"]
            fig = px.line(tmp, x="Income Bracket", y="Rate (%)", markers=True,
                          template=PLOT_TEMPLATE, color_discrete_sequence=[ACCENT])
            fig.update_traces(line=dict(width=3), marker=dict(size=9))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
            st.plotly_chart(fig, use_container_width=True, key="demo_income_line")
        with c3:
            purchase_rate_by(fdf, "Gender", "Purchase Rate by Gender")

        c4, c5 = st.columns(2)
        with c4:
            st.markdown('<div class="section-title">Current Car Type Distribution</div>', unsafe_allow_html=True)
            tmp = fdf["Current_Car_Type"].value_counts().reset_index()
            tmp.columns = ["Car Type", "Count"]
            fig = px.pie(tmp, names="Car Type", values="Count", hole=0.55,
                         color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="demo_car_pie")
        with c5:
            st.markdown('<div class="section-title">Number of Cars Owned vs Purchase</div>', unsafe_allow_html=True)
            tmp = fdf.groupby("Number_of_Cars_Owned")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            tmp.columns = ["Cars Owned", "Rate (%)"]
            fig = px.bar(tmp, x="Cars Owned", y="Rate (%)", text="Rate (%)",
                         color_discrete_sequence=[PALETTE[4]], template=PLOT_TEMPLATE)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
            st.plotly_chart(fig, use_container_width=True, key="demo_cars_owned")

# ==================================================================
# PAGE 3: CHARGING INFRASTRUCTURE
# ==================================================================
with tab_charge:
    page_header("Charging Infrastructure", "How access to chargers shapes range anxiety and buying decisions.")

    if not fdf.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-title">Stations Near Home vs Anxiety</div>', unsafe_allow_html=True)
            fig = px.box(fdf, x="Range_Anxiety_Level", y="Charging_Stations_Near_Home",
                         category_orders={"Range_Anxiety_Level": ["Low", "Medium", "High"]},
                         color="Range_Anxiety_Level", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH,
                               xaxis_title="Range Anxiety", yaxis_title="Stations Near Home")
            st.plotly_chart(fig, use_container_width=True, key="charge_box_home")
        with c2:
            st.markdown('<div class="section-title">Home Charging Impact on Purchase</div>', unsafe_allow_html=True)
            tmp = fdf.groupby(["Home_Charging_Possible", "Will_Buy_EV"]).size().reset_index(name="Count")
            fig = px.bar(tmp, x="Home_Charging_Possible", y="Count", color="Will_Buy_EV", barmode="group",
                         color_discrete_sequence=[PALETTE[3], PALETTE[0]], template=PLOT_TEMPLATE)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH,
                               xaxis_title="Home Charging Possible", legend_title="")
            st.plotly_chart(fig, use_container_width=True, key="charge_home_impact")
        with c3:
            st.markdown('<div class="section-title">Stations Near Work vs Anxiety</div>', unsafe_allow_html=True)
            fig = px.box(fdf, x="Range_Anxiety_Level", y="Charging_Stations_Near_Work",
                         category_orders={"Range_Anxiety_Level": ["Low", "Medium", "High"]},
                         color="Range_Anxiety_Level", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH,
                               xaxis_title="Range Anxiety", yaxis_title="Stations Near Work")
            st.plotly_chart(fig, use_container_width=True, key="charge_box_work")

        c4, c5 = st.columns(2)
        with c4:
            st.markdown('<div class="section-title">Daily Commute vs Range Anxiety</div>', unsafe_allow_html=True)
            fig = px.violin(fdf, x="Range_Anxiety_Level", y="Daily_Commute_km", box=True,
                             category_orders={"Range_Anxiety_Level": ["Low", "Medium", "High"]},
                             color="Range_Anxiety_Level", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH,
                               xaxis_title="Range Anxiety", yaxis_title="Daily Commute (km)")
            st.plotly_chart(fig, use_container_width=True, key="charge_violin_commute")
        with c5:
            st.markdown('<div class="section-title">Correlation Between Numeric Factors</div>', unsafe_allow_html=True)
            num_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Charging_Stations_Near_Home",
                        "Charging_Stations_Near_Work", "Environmental_Concern_Level", "Range_Anxiety_Rank"]
            corr = fdf[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", template=PLOT_TEMPLATE,
                             aspect="auto", zmin=-1, zmax=1)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
            st.plotly_chart(fig, use_container_width=True, key="charge_heatmap")

# ==================================================================
# PAGE 4: BUYING BEHAVIOR
# ==================================================================
with tab_buy:
    page_header("What Drives the Buy Decision?", "Subsidies, environmental concern, and other purchase triggers.")

    if not fdf.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-title">Subsidy Availability Impact</div>', unsafe_allow_html=True)
            tmp = fdf.groupby(["Subsidy_Available", "Will_Buy_EV"]).size().reset_index(name="Count")
            fig = px.bar(tmp, x="Subsidy_Available", y="Count", color="Will_Buy_EV", barmode="group",
                         color_discrete_sequence=[PALETTE[3], PALETTE[0]], template=PLOT_TEMPLATE)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH,
                               xaxis_title="Subsidy Available", legend_title="")
            st.plotly_chart(fig, use_container_width=True, key="buy_subsidy_impact")
        with c2:
            purchase_rate_by(fdf, "Environmental_Concern_Level", "Environmental Concern vs Purchase", color_seq=[PALETTE[4]] * 10)
        with c3:
            purchase_rate_by(fdf, "Current_Car_Type", "Purchase Rate by Car Type Owned")

        c4, c5 = st.columns(2)
        with c4:
            st.markdown('<div class="section-title">Subsidy + Home Charging Combined Effect</div>', unsafe_allow_html=True)
            tmp = fdf.groupby(["Subsidy_Available", "Home_Charging_Possible"])["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            tmp.columns = ["Subsidy", "Home Charging", "Rate (%)"]
            fig = px.bar(tmp, x="Subsidy", y="Rate (%)", color="Home Charging", barmode="group", text="Rate (%)",
                         color_discrete_sequence=[PALETTE[3], PALETTE[0]], template=PLOT_TEMPLATE)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
            st.plotly_chart(fig, use_container_width=True, key="buy_combined_effect")
        with c5:
            st.markdown('<div class="section-title">Range Anxiety Distribution</div>', unsafe_allow_html=True)
            tmp = fdf["Range_Anxiety_Level"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
            tmp.columns = ["Range Anxiety", "Count"]
            fig = px.pie(tmp, names="Range Anxiety", values="Count", hole=0.55,
                         color="Range Anxiety", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="buy_anxiety_pie")

# ==================================================================
# PAGE 5: PREDICTION
# ==================================================================
with tab_predict:
    page_header("EV Purchase Prediction", "Enter a customer's profile and get a live XGBoost prediction.")

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">Customer Details</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            gender_in = st.selectbox("Gender", ["Male", "Female"])
            income = st.number_input("Annual Income (USD)", min_value=0, value=50000, step=1000)
            cars = st.number_input("Number of Cars Owned", min_value=0, max_value=10, value=1)
        with col_b:
            commute = st.number_input("Daily Commute (km)", min_value=0, value=20)
            home_station = st.number_input("Charging Stations Near Home", min_value=0, value=2)
            work_station = st.number_input("Charging Stations Near Work", min_value=0, value=1)
            home_charge_in = st.selectbox("Home Charging Possible", ["Yes", "No"])
        with col_c:
            concern = st.slider("Environmental Concern", 1, 10, 5)
            subsidy_in = st.selectbox("Subsidy Available", ["Yes", "No"])
            range_anxiety_in = st.selectbox("Range Anxiety", ["Low", "Medium", "High"])
            city_in = st.selectbox("City Type", ["Urban", "Suburban", "Rural"])

        # Current car only makes sense if the customer owns at least one car.
        if cars == 0:
            st.selectbox("Current Car", ["None — owns 0 cars"], index=0, disabled=True,
                         help="Disabled because Number of Cars Owned is 0.")
            car_in = "None"
        else:
            car_in = st.selectbox("Current Car", ["SUV", "Sedan", "Truck", "Hatchback"])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        gender_n = 1 if gender_in == "Male" else 0
        home_charge_n = 1 if home_charge_in == "Yes" else 0
        subsidy_n = 1 if subsidy_in == "Yes" else 0
        range_dict = {"Low": 0, "Medium": 1, "High": 2}
        range_n = range_dict[range_anxiety_in]
        city_suburban = 1 if city_in == "Suburban" else 0
        city_urban = 1 if city_in == "Urban" else 0
        # "None" and "Hatchback" both correctly encode as all-zero (baseline category) —
        # this matches how the model was originally trained, so no schema change needed.
        car_suv = 1 if car_in == "SUV" else 0
        car_sedan = 1 if car_in == "Sedan" else 0
        car_truck = 1 if car_in == "Truck" else 0

        input_df = pd.DataFrame({
            "Age": [age], "Gender": [gender_n], "Annual_Income_USD": [income],
            "Daily_Commute_km": [commute], "Number_of_Cars_Owned": [cars],
            "Charging_Stations_Near_Home": [home_station], "Charging_Stations_Near_Work": [work_station],
            "Home_Charging_Possible": [home_charge_n], "Environmental_Concern_Level": [concern],
            "Subsidy_Available": [subsidy_n], "Range_Anxiety_Level": [range_n],
            "City_Type_Suburban": [city_suburban], "City_Type_Urban": [city_urban],
            "Current_Car_Type_SUV": [car_suv], "Current_Car_Type_Sedan": [car_sedan],
            "Current_Car_Type_Truck": [car_truck],
        })

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        buy_prob = probability[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1])
        with r1:
            if prediction[0] == 1:
                st.success("✅ Customer is likely to BUY an EV")
            else:
                st.error("❌ Customer is NOT likely to buy an EV")
            st.progress(int(buy_prob * 100))
            st.caption(f"Probability of buying: {buy_prob*100:.2f}% · Not buying: {(1-buy_prob)*100:.2f}%")
        with r2:
            kc1, kc2 = st.columns(2)
            kpi(kc1, "Buy Probability", f"{buy_prob*100:.1f}%")
            kpi(kc2, "Confidence", "High" if abs(buy_prob - 0.5) > 0.3 else "Moderate")

# ==================================================================
# PAGE 6: AI ASSISTANT
# ==================================================================
with tab_ai:
    page_header("💬 EV AI Assistant", "Ask anything about EV adoption — answers reflect your current sidebar filters.")

    @st.cache_data
    def build_context(data):
        if data.empty:
            return "No records match the current filters."
        buy_rate = (data["Will_Buy_EV"] == "Yes").mean() * 100
        return f"""
Records in current view: {len(data):,}
Overall EV purchase rate in this view: {buy_rate:.1f}%
Purchase rate by range anxiety: {data.groupby('Range_Anxiety_Level')['Will_Buy_EV'].apply(lambda s: (s=='Yes').mean()*100).round(1).to_dict()}
Purchase rate by city type: {data.groupby('City_Type')['Will_Buy_EV'].apply(lambda s: (s=='Yes').mean()*100).round(1).to_dict()}
Purchase rate by subsidy availability: {data.groupby('Subsidy_Available')['Will_Buy_EV'].apply(lambda s: (s=='Yes').mean()*100).round(1).to_dict()}
Avg annual income: ${data['Annual_Income_USD'].mean():,.0f}
Avg daily commute: {data['Daily_Commute_km'].mean():.1f} km
Top predictive features from the trained XGBoost model: Subsidy_Available, Environmental_Concern_Level, Range_Anxiety_Level, Annual_Income_USD, City_Type_Urban.
""".strip()

    with st.form("ai_form"):
        question = st.text_area("Your question")
        ask = st.form_submit_button("Ask AI")

    if ask:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            context = build_context(fdf)
            prompt = f"""You are an EV data analyst. Answer using only this data summary. Be concise and specific.

DATA SUMMARY:
{context}

USER QUESTION:
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

# ==================================================================
# PAGE 7: EXPLORE DATA
# ==================================================================
with tab_data:
    page_header("Explore the Raw Data", "Drill into individual records within your current sidebar filters.")

    buy_sel = st.selectbox("Further narrow by: Will Buy EV", ["All", "Yes", "No"])
    view = fdf if buy_sel == "All" else fdf[fdf["Will_Buy_EV"] == buy_sel]

    st.caption(f"Showing {len(view):,} of {len(df):,} total records")
    st.dataframe(view, use_container_width=True, height=420)
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download This View as CSV", csv, "filtered_ev_data.csv", "text/csv")