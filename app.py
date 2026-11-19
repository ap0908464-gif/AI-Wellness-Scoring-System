import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 1. Page Configuration (Mobile + Laptop Responsive)
st.set_page_config(
    page_title="AI Wellness Scoring System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Mobile-First Clean Mint Theme & High-Contrast Inputs
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F4F7F6 !important;
        color: #1B4332 !important;
    }
    label, p, h1, h2, h3, h4, span {
        color: #1B4332 !important;
    }
    .stTextInput input, .stNumberInput input, [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1B4332 !important;
        border: 1px solid #B7D4C5 !important;
        border-radius: 10px !important;
    }
    .health-card {
        background-color: #FFFFFF;
        padding: 16px 18px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.06);
        border: 1px solid #DCE8E2;
        margin-bottom: 12px;
    }
    .card-title { font-size: 13px; font-weight: 700; color: #2D6A4F !important; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 6px; }
    .big-num { font-size: 30px; font-weight: 800; color: #1B4332 !important; line-height: 1.1; }
    .unit-text { font-size: 15px; font-weight: 600; color: #40916C !important; }
    .sub-text { font-size: 13px; color: #4A6358 !important; margin-top: 8px; }
    .progress-bg { background-color: #E0ECE6; border-radius: 10px; height: 8px; width: 100%; margin-top: 10px; }
    .progress-fill { background: linear-gradient(90deg, #52B788, #1B4332); height: 8px; border-radius: 10px; }
    .tip-box {
        background-color: #FFFFFF;
        border-left: 4px solid #2D6A4F;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #1B4332 !important;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricValue"] { color: #1B4332 !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #2D6A4F !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        background-color: #E3EDE8;
        border-radius: 10px;
        padding: 8px 14px;
        font-weight: 600;
        color: #1B4332 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B4332 !important;
        color: #FFFFFF !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. ML Model Training & Evaluation
@st.cache_resource
def train_wellness_model():
    np.random.seed(42)
    n = 1500
    age = np.random.randint(18, 65, n)
    bmi = np.round(np.random.uniform(17.0, 35.0, n), 1)
    sleep = np.round(np.random.uniform(4.0, 9.5, n), 1)
    glasses = np.random.randint(1, 13, n)
    steps = np.random.randint(1000, 16000, n)
    hr = np.random.randint(55, 108, n)
    calories = np.random.randint(400, 2400, n)
    stress = np.random.randint(1, 11, n)

    score = (
        np.where((sleep >= 7) & (sleep <= 9), 20, np.where(sleep >= 5.5, 12, 5)) +
        np.where(glasses >= 8, 15, glasses * 1.6) +
        np.where(steps >= 8000, 20, (steps / 8000) * 18) +
        np.where((hr >= 60) & (hr <= 80), 15, 7) +
        np.where((bmi >= 18.5) & (bmi <= 24.9), 15, 8) +
        np.where(calories >= 1100, 8, 4) +
        (11 - stress) * 0.7 +
        np.random.normal(0, 1.2, n)
    )
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        'Age': age, 'BMI': bmi, 'Sleep_Hrs': sleep, 'Water_Glasses': glasses,
        'Daily_Steps': steps, 'Heart_Rate': hr, 'Calories': calories, 'Stress_Level': stress
    })

    X_train, X_test, y_train, y_test = train_test_split(df, score, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    accuracy = round(r2_score(y_test, model.predict(X_test)) * 100, 2)

    importances = pd.DataFrame({
        'Parameter': df.columns,
        'Importance (%)': np.round(model.feature_importances_ * 100, 1)
    }).sort_values(by='Importance (%)', ascending=True)

    return model, accuracy, importances, df.head(10)

model, model_accuracy, feature_imp_df, sample_dataset = train_wellness_model()

# 4. Header
st.markdown("<h2 style='color:#1B4332; margin-bottom:2px;'>🩺 AI-Based Wellness Scoring System</h2>", unsafe_allow_html=True)
st.caption("Interactive Health Analytics & Predictive Wellness Engine")

# 5. Main-Screen Input Card (Directly visible on Phone & Laptop!)
with st.expander("✏️ Enter / Edit Your Health Parameters (Tap Here)", expanded=True):
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([2, 1, 1, 1])
    with r1_c1:
        user_name = st.text_input("Your Name", value="", placeholder="Enter your name (e.g., Aarav)", key="u_name")
    with r1_c2:
        age = st.number_input("Age (Years)", 12, 100, 24, key="u_age")
    with r1_c3:
        height_cm = st.number_input("Height (cm)", 120, 220, 170, key="u_ht")
    with r1_c4:
        weight_kg = st.number_input("Weight (kg)", 30, 180, 65, key="u_wt")

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        steps = st.slider("Daily Steps", 500, 16000, 7800, 100, key="u_steps")
        sleep_hrs = st.slider("Sleep Duration (hrs)", 3.0, 11.0, 7.5, 0.5, key="u_sleep")
    with r2_c2:
        calories = st.slider("Calories Burned (kcal)", 300, 2500, 1250, 50, key="u_cal")
        water_glasses = st.slider("Water Intake (Glasses)", 0, 14, 6, key="u_water")
    with r2_c3:
        heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 130, 72, key="u_hr")
        stress_level = st.slider("Stress Level (1=Low, 10=High)", 1, 10, 4, key="u_stress")

display_name = user_name.strip() if user_name.strip() else "Guest User"
bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
bmi_status = "Healthy" if 18.5 <= bmi <= 24.9 else ("Underweight" if bmi < 18.5 else "Overweight")

# 6. Live ML Prediction
input_df = pd.DataFrame([{
    'Age': age, 'BMI': bmi, 'Sleep_Hrs': sleep_hrs, 'Water_Glasses': water_glasses,
    'Daily_Steps': steps, 'Heart_Rate': heart_rate, 'Calories': calories, 'Stress_Level': stress_level
}])
ai_score = round(float(model.predict(input_df)[0]), 1)

if ai_score >= 80:
    status_label = "Optimal Wellness"
elif ai_score >= 60:
    status_label = "Moderate Wellness"
else:
    status_label = "Needs Attention"

# Score Banner
st.markdown(f"""
<div class="health-card" style
