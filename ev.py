import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)

model = joblib.load("ev_xgboost.pkl")
# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="EV Insights",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# GLOBAL STYLE — website-like navbar, hero, cards, tighter spacing
# ------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1300px; }

    /* Hide default sidebar toggle look, keep it minimal */
    header[data-testid="stHeader"] { background: transparent; }

    /* Navbar */
    .navbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 4px 18px 4px; border-bottom: 1px solid #262d3d; margin-bottom: 6px;
    }
    .navbar-brand { font-size: 1.35rem; font-weight: 800; color: #f5f7fa; }
    .navbar-brand span { color: #5cd6a5; }

    /* Tabs styled like a navbar menu */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; border-bottom: 1px solid #262d3d; margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px; background-color: transparent; border-radius: 8px 8px 0 0;
        color: #8b93a7; font-weight: 600; padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        color: #5cd6a5 !important; border-bottom: 2px solid #5cd6a5;
    }

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
    .hero-sub { color: #8b93a7; font-size: 0.95rem; margin-bottom: 18px; }

    .section-title { color: #f5f7fa; font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }

    h1, h2, h3 { color: #f5f7fa !important; }
</style>
""", unsafe_allow_html=True)

ACCENT = "#5cd6a5"
PALETTE = ["#5cd6a5", "#4f8cff", "#ffb454", "#ff6b6b", "#a78bfa", "#38bdf8"]
PLOT_TEMPLATE = "plotly_dark"
CH = 300  # standard chart height so pages fit without scrolling

# ------------------------------------------------------------------
# DATA
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

df = load_data()

def kpi(col, label, value, sub=None, sub_neg=False):
    sub_html = f"<div class='kpi-sub{' neg' if sub_neg else ''}'>{sub}</div>" if sub else ""
    col.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>{sub_html}</div>""",
        unsafe_allow_html=True,
    )

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
    "🏠 Home",
    "👥 Demographics",
    "🔌 Charging",
    "🛒 Buying",
    "🤖 Prediction",
    "💬 AI Assistant",
    "📄 Explore Data"
])

# ==================================================================
# PAGE 1: HOME
# ==================================================================
with tab_home:
    st.markdown('<div class="hero-title">EV Adoption at a Glance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">A snapshot of who buys electric vehicles — and what holds the rest back.</div>', unsafe_allow_html=True)

    buy_rate = (df["Will_Buy_EV"] == "Yes").mean() * 100
    avg_income = df["Annual_Income_USD"].mean()
    high_anxiety_pct = (df["Range_Anxiety_Level"] == "High").mean() * 100
    home_charge_pct = (df["Home_Charging_Possible"] == "Yes").mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Total Buyers Surveyed", f"{len(df):,}")
    kpi(k2, "Will Buy an EV", f"{buy_rate:.1f}%")
    kpi(k3, "Avg Annual Income", f"${avg_income:,.0f}")
    kpi(k4, "High Range Anxiety", f"{high_anxiety_pct:.1f}%", "Biggest adoption blocker" if high_anxiety_pct > 20 else None, sub_neg=True)
    kpi(k5, "Home Charging Access", f"{home_charge_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.markdown('<div class="section-title">Purchase Decision Split</div>', unsafe_allow_html=True)
        tmp = df["Will_Buy_EV"].value_counts().reset_index()
        tmp.columns = ["Decision", "Count"]
        fig = px.pie(tmp, names="Decision", values="Count", hole=0.6,
                     color="Decision", color_discrete_map={"Yes": ACCENT, "No": "#ff6b6b"},
                     template=PLOT_TEMPLATE)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Purchase Rate by Range Anxiety</div>', unsafe_allow_html=True)
        tmp = df.groupby("Range_Anxiety_Level")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reindex(["Low", "Medium", "High"]).reset_index()
        tmp.columns = ["Range Anxiety", "Rate (%)"]
        fig = px.bar(tmp, x="Range Anxiety", y="Rate (%)", text="Rate (%)",
                     color="Range Anxiety", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">Purchase Rate by City Type</div>', unsafe_allow_html=True)
        tmp = df.groupby("City_Type")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["City Type", "Rate (%)"]
        fig = px.bar(tmp, x="City Type", y="Rate (%)", text="Rate (%)",
                     color="City Type", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE 2: DEMOGRAPHICS
# ==================================================================
with tab_demo:
    st.markdown('<div class="hero-title">Who Are the Buyers?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Age, income, and household profile of survey respondents.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">Purchase Rate by Age Group</div>', unsafe_allow_html=True)
        tmp = df.groupby("Age_Group", observed=True)["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["Age Group", "Rate (%)"]
        fig = px.area(tmp, x="Age Group", y="Rate (%)", template=PLOT_TEMPLATE, color_discrete_sequence=[PALETTE[1]])
        fig.update_traces(line=dict(width=3))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Purchase Rate by Income Bracket</div>', unsafe_allow_html=True)
        tmp = df.groupby("Income_Bracket", observed=True)["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["Income Bracket", "Rate (%)"]
        fig = px.line(tmp, x="Income Bracket", y="Rate (%)", markers=True,
                      template=PLOT_TEMPLATE, color_discrete_sequence=[ACCENT])
        fig.update_traces(line=dict(width=3), marker=dict(size=9))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">Purchase Rate by Gender</div>', unsafe_allow_html=True)
        tmp = df.groupby("Gender")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["Gender", "Rate (%)"]
        fig = px.bar(tmp, x="Gender", y="Rate (%)", text="Rate (%)",
                     color="Gender", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown('<div class="section-title">Current Car Type Distribution</div>', unsafe_allow_html=True)
        tmp = df["Current_Car_Type"].value_counts().reset_index()
        tmp.columns = ["Car Type", "Count"]
        fig = px.pie(tmp, names="Car Type", values="Count", hole=0.55,
                     color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        st.markdown('<div class="section-title">Number of Cars Owned vs Purchase</div>', unsafe_allow_html=True)
        tmp = df.groupby("Number_of_Cars_Owned")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["Cars Owned", "Rate (%)"]
        fig = px.bar(tmp, x="Cars Owned", y="Rate (%)", text="Rate (%)",
                     color_discrete_sequence=[PALETTE[4]], template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE 3: CHARGING INFRASTRUCTURE
# ==================================================================
with tab_charge:
    st.markdown('<div class="hero-title">Charging Infrastructure</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">How access to chargers shapes range anxiety and buying decisions.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">Charging Stations Near Home vs Anxiety</div>', unsafe_allow_html=True)
        fig = px.box(df, x="Range_Anxiety_Level", y="Charging_Stations_Near_Home",
                     category_orders={"Range_Anxiety_Level": ["Low", "Medium", "High"]},
                     color="Range_Anxiety_Level", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH,
                           xaxis_title="Range Anxiety", yaxis_title="Stations Near Home")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Home Charging Impact on Purchase</div>', unsafe_allow_html=True)
        tmp = df.groupby(["Home_Charging_Possible", "Will_Buy_EV"]).size().reset_index(name="Count")
        fig = px.bar(tmp, x="Home_Charging_Possible", y="Count", color="Will_Buy_EV", barmode="group",
                     color_discrete_sequence=[PALETTE[3], PALETTE[0]], template=PLOT_TEMPLATE)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH,
                           xaxis_title="Home Charging Possible", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">Stations Near Work vs Anxiety</div>', unsafe_allow_html=True)
        fig = px.box(df, x="Range_Anxiety_Level", y="Charging_Stations_Near_Work",
                     category_orders={"Range_Anxiety_Level": ["Low", "Medium", "High"]},
                     color="Range_Anxiety_Level", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH,
                           xaxis_title="Range Anxiety", yaxis_title="Stations Near Work")
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown('<div class="section-title">Daily Commute vs Range Anxiety</div>', unsafe_allow_html=True)
        fig = px.violin(df, x="Range_Anxiety_Level", y="Daily_Commute_km", box=True,
                         category_orders={"Range_Anxiety_Level": ["Low", "Medium", "High"]},
                         color="Range_Anxiety_Level", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH,
                           xaxis_title="Range Anxiety", yaxis_title="Daily Commute (km)")
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        st.markdown('<div class="section-title">Correlation Between Numeric Factors</div>', unsafe_allow_html=True)
        num_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Charging_Stations_Near_Home",
                    "Charging_Stations_Near_Work", "Environmental_Concern_Level", "Range_Anxiety_Rank"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", template=PLOT_TEMPLATE,
                         aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE 4: BUYING BEHAVIOR
# ==================================================================
with tab_buy:
    st.markdown('<div class="hero-title">What Drives the Buy Decision?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Subsidies, environmental concern, and other purchase triggers.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">Subsidy Availability Impact</div>', unsafe_allow_html=True)
        tmp = df.groupby(["Subsidy_Available", "Will_Buy_EV"]).size().reset_index(name="Count")
        fig = px.bar(tmp, x="Subsidy_Available", y="Count", color="Will_Buy_EV", barmode="group",
                     color_discrete_sequence=[PALETTE[3], PALETTE[0]], template=PLOT_TEMPLATE)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH,
                           xaxis_title="Subsidy Available", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Environmental Concern vs Purchase</div>', unsafe_allow_html=True)
        tmp = df.groupby("Environmental_Concern_Level")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["Concern Level", "Rate (%)"]
        fig = px.bar(tmp, x="Concern Level", y="Rate (%)", text="Rate (%)",
                     color_discrete_sequence=[PALETTE[4]], template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-title">Purchase Rate by Car Type Owned</div>', unsafe_allow_html=True)
        tmp = df.groupby("Current_Car_Type")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["Car Type", "Rate (%)"]
        fig = px.bar(tmp, x="Car Type", y="Rate (%)", text="Rate (%)",
                     color="Car Type", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown('<div class="section-title">Subsidy + Home Charging Combined Effect</div>', unsafe_allow_html=True)
        tmp = df.groupby(["Subsidy_Available", "Home_Charging_Possible"])["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tmp.columns = ["Subsidy", "Home Charging", "Rate (%)"]
        fig = px.bar(tmp, x="Subsidy", y="Rate (%)", color="Home Charging", barmode="group", text="Rate (%)",
                     color_discrete_sequence=[PALETTE[3], PALETTE[0]], template=PLOT_TEMPLATE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        st.markdown('<div class="section-title">Range Anxiety Distribution</div>', unsafe_allow_html=True)
        tmp = df["Range_Anxiety_Level"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
        tmp.columns = ["Range Anxiety", "Count"]
        fig = px.pie(tmp, names="Range Anxiety", values="Count", hole=0.55,
                     color="Range Anxiety", color_discrete_sequence=PALETTE, template=PLOT_TEMPLATE)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=CH, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE 5: EXPLORE DATA
# ================================================================== 
with tab_data:
    st.markdown('<div class="hero-title">Explore the Raw Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Filter, inspect, and export the underlying dataset.</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        city_sel = st.multiselect("City Type", sorted(df["City_Type"].unique()), default=sorted(df["City_Type"].unique()))
    with f2:
        anxiety_sel = st.multiselect("Range Anxiety", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    with f3:
        subsidy_sel = st.selectbox("Subsidy Available", ["All", "Yes", "No"])
    with f4:
        buy_sel = st.selectbox("Will Buy EV", ["All", "Yes", "No"])

    filtered = df[df["City_Type"].isin(city_sel) & df["Range_Anxiety_Level"].isin(anxiety_sel)]
    if subsidy_sel != "All":
        filtered = filtered[filtered["Subsidy_Available"] == subsidy_sel]
    if buy_sel != "All":
        filtered = filtered[filtered["Will_Buy_EV"] == buy_sel]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, use_container_width=True, height=420)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Filtered Data as CSV", csv, "filtered_ev_data.csv", "text/csv")


with tab_predict:

    st.title("EV Purchase Prediction")

    st.subheader("Enter Customer Details")

    age = st.number_input(
    "Age",
    min_value=18,
    max_value=80,
    value=30
) 
    gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
) 
    income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)
    commute = st.number_input(
    "Daily Commute (km)",
    min_value=0,
    value=20
)
    cars = st.number_input(
    "Number of Cars Owned",
    min_value=0,
    value=1
)
    home_station = st.number_input(
    "Charging Stations Near Home",
    min_value=0,
    value=2
)

    work_station = st.number_input(
    "Charging Stations Near Work",
    min_value=0,
    value=1
)
    home_charge = st.selectbox(
    "Home Charging Possible",
    ["Yes","No"]
)
    concern = st.slider(
    "Environmental Concern",
    1,
    10,
    5
)
    subsidy = st.selectbox(
    "Subsidy Available",
    ["Yes","No"]
)
    range_anxiety = st.selectbox(
    "Range Anxiety",
    ["Low","Medium","High"]
)
    city = st.selectbox(
    "City Type",
    ["Urban","Suburban","Rural"]
)
    car = st.selectbox(
    "Current Car",
    ["SUV","Sedan","Truck","Hatchback"]
)
    
    # -----------------------------
    # Convert text to numbers
    # -----------------------------

    gender = 1 if gender=="Male" else 0

    home_charge = 1 if home_charge=="Yes" else 0

    subsidy = 1 if subsidy=="Yes" else 0

    range_dict={
        "Low":0,
        "Medium":1,
        "High":2
    }

    range_anxiety=range_dict[range_anxiety]

    city_suburban=1 if city=="Suburban" else 0
    city_urban=1 if city=="Urban" else 0

    car_suv=1 if car=="SUV" else 0
    car_sedan=1 if car=="Sedan" else 0
    car_truck=1 if car=="Truck" else 0


    input_df=pd.DataFrame({

        "Age":[age],

        "Gender":[gender],

        "Annual_Income_USD":[income],

        "Daily_Commute_km":[commute],

        "Number_of_Cars_Owned":[cars],

        "Charging_Stations_Near_Home":[home_station],

        "Charging_Stations_Near_Work":[work_station],

        "Home_Charging_Possible":[home_charge],

        "Environmental_Concern_Level":[concern],

        "Subsidy_Available":[subsidy],

        "Range_Anxiety_Level":[range_anxiety],

        "City_Type_Suburban":[city_suburban],

        "City_Type_Urban":[city_urban],

        "Current_Car_Type_SUV":[car_suv],

        "Current_Car_Type_Sedan":[car_sedan],

        "Current_Car_Type_Truck":[car_truck]

    })

    if st.button("Predict"):

        prediction=model.predict(input_df)

        probability=model.predict_proba(input_df)

        st.subheader("Prediction")

        if prediction[0]==1:

            st.success("Customer is likely to BUY an EV")

        else:

            st.error("Customer is NOT likely to buy an EV")
            st.subheader("Prediction Probability")

        st.write(f"Probability of Buying EV : {probability[0][1]*100:.2f}%")

        st.write(f"Probability of NOT Buying EV : {probability[0][0]*100:.2f}%")

        st.subheader("Confidence")

        buy_prob = probability[0][1]

        st.progress(int(buy_prob * 100))

        st.metric(
        "Likelihood of Buying EV",
        f"{buy_prob*100:.2f}%"
                            )


with tab_ai:

    st.title("💬 EV AI Assistant")

    question = st.text_area("Your Question")

    if st.button("Ask AI"):

        prompt = f"""
        You are an EV data analyst.

        Dataset Summary:
        {df.describe(include='all').to_string()}

        User Question:
        {question}
        """

        with st.spinner("Thinking..."):

            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response.choices[0].message.content
                st.success(answer)

            except Exception as e:
                st.error(str(e))