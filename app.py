import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# 1. Page Config (Mobile + Laptop Responsive)
st.set_page_config(page_title="My Health Dashboard", page_icon="🌿", layout="wide")

# 2. Clean Mint-Green Theme & High-Contrast Text
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F4F7F6 !important;
        color: #1B4332 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #EBF2EE !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
        color: #1B4332 !important;
    }
    .health-card {
        background-color: #FFFFFF;
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #DCE8E2;
        margin-bottom: 10px;
    }
    .card-title { font-size: 15px; font-weight: 700; color: #1B4332; margin-bottom: 6px; }
    .big-num { font-size: 32px; font-weight: 800; color: #1B4332; line-height: 1.1; }
    .unit-text { font-size: 16px; font-weight: 600; color: #40916C; }
    .sub-text { font-size: 13px; color: #4A6358; margin-top: 8px; }
    .progress-bg { background-color: #E0ECE6; border-radius: 10px; height: 8px; width: 100%; margin-top: 10px; }
    .progress-fill { background: linear-gradient(90deg, #52B788, #1B4332); height: 8px; border-radius: 10px; }
    [data-testid="stMetricValue"] { color: #1B4332 !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #2D6A4F !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# 3. ML Model Training (Cached)
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    sleep = np.round(np.random.uniform(4.0, 9.5, n), 1)
    glasses = np.random.randint(1, 12, n)
    steps = np.random.randint(1000, 15000, n)
    hr = np.random.randint(55, 105, n)
    calories = np.random.randint(400, 2200, n)
    
    score = (
        np.where((sleep >= 7) & (sleep <= 9), 25, 12) +
        np.where(glasses >= 8, 20, glasses * 2) +
        np.where(steps >= 8000, 25, (steps / 8000) * 20) +
        np.where((hr >= 60) & (hr <= 80), 15, 8) +
        np.where(calories >= 1000, 15, 8)
    )
    score = np.clip(score, 0, 100)
    df = pd.DataFrame({'Sleep': sleep, 'Glasses': glasses, 'Steps': steps, 'HR': hr, 'Calories': calories})
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df, score)
    return model

model = train_model()

# 4. Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Update Health Data")
    user_name = st.text_input("Name", value="Nikku", key="name_input")
    steps = st.slider("Daily Steps", 500, 15000, 7540, 100, key="steps_slider")
    calories = st.slider("Calories Burned (kcal)", 300, 2500, 1200, 50, key="cal_slider")
    heart_rate = st.slider("Heart Rate (bpm)", 50, 130, 72, key="hr_slider")
    sleep_hrs = st.slider("Last Night Sleep (hrs)", 3.0, 11.0, 7.5, 0.5, key="sleep_slider")
    water_glasses = st.slider("Water Intake (Glasses)", 0, 12, 5, key="water_slider")

# Predict AI Wellness Score
input_df = pd.DataFrame([{'Sleep': sleep_hrs, 'Glasses': water_glasses, 'Steps': steps, 'HR': heart_rate, 'Calories': calories}])
ai_score = round(float(model.predict(input_df)[0]), 1)

# 5. Header Row
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.markdown("<h1 style='color:#1B4332; margin:0; padding:0;'>My Health Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#4A6358; margin-top:4px;'>Project No. 37 • Welcome back, <b>{user_name}</b></p>", unsafe_allow_html=True)
with h_col2:
    st.metric(label="🧠 AI Wellness Score", value=f"{ai_score} / 100", delta="Optimal" if ai_score >= 75 else "Needs Boost")

# 6. ROW 1: Top 3 Metric Cards
c1, c2, c3 = st.columns(3)

with c1:
    step_pct = min(int((steps / 10000) * 100), 100)
    st.markdown(f"""
    <div class="health-card">
        <div class="card-title">🏃 Daily Steps</div>
        <div class="big-num">{steps:,}</div>
        <div class="progress-bg"><div class="progress-fill" style="width:{step_pct}%;"></div></div>
        <div class="sub-text">Goal: 10,000 • <span style="color:#2D6A4F; font-weight:700;">{step_pct}% completed</span></div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    cal_pct = min(int((calories / 1500) * 100), 100)
    st.markdown(f"""
    <div class="health-card">
        <div class="card-title">🔥 Calories Burned</div>
        <div class="big-num" style="color:#40916C;">{calories:,} <span class="unit-text">kcal</span></div>
        <div class="progress-bg"><div class="progress-fill" style="width:{cal_pct}%;"></div></div>
        <div class="sub-text">Active: {int(calories * 0.67)} kcal • Resting: {int(calories * 0.33)} kcal</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    hr_pct = min(max(int(((heart_rate - 45) / 85) * 100), 10), 100)
    st.markdown(f"""
    <div class="health-card">
        <div class="card-title">💚 Heart Rate</div>
        <div class="big-num">{heart_rate} <span class="unit-text">bpm</span></div>
        <div class="progress-bg"><div class="progress-fill" style="width:{hr_pct}%;"></div></div>
        <div class="sub-text">Resting: 60 bpm • Peak: 110 bpm</div>
    </div>
    """, unsafe_allow_html=True)

# 7. ROW 2: Weekly Activity Wave Chart + Sleep & Hydration/Challenge
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="health-card"><div class="card-title">📈 Activity Overview (Weekly)</div>', unsafe_allow_html=True)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    steps_trend = [720, 1320, 910, 1850, 1240, 1810, 1020]
    active_mins = [380, 740, 480, 1150, 810, 1420, 1730]

    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=days, y=steps_trend, mode='lines+markers', name='Steps (x10)',
        line=dict(color='#52B788', width=3, shape='spline'),
        fill='tozeroy', fillcolor='rgba(82, 183, 136, 0.25)'
    ))
    fig_wave.add_trace(go.Scatter(
        x=days, y=active_mins, mode='lines+markers', name='Active Minutes',
        line=dict(color='#1B4332', width=3, shape='spline'),
        fill='tozeroy', fillcolor='rgba(27, 67, 50, 0.12)'
    ))
    fig_wave.update_layout(
        height=275, margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(color='#1B4332', size=12),
        legend=dict(orientation="h", y=1.15, x=0.15, font=dict(color='#1B4332')),
        xaxis=dict(tickfont=dict(color='#1B4332')),
        yaxis=dict(gridcolor='#E6EFEA', tickfont=dict(color='#1B4332'))
    )
    st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown(f'<div class="health-card"><div class="card-title">😴 Sleep Quality &nbsp; <span style="background:#1B4332;color:white;padding:3px 10px;border-radius:8px;font-size:12px;">{sleep_hrs} hrs</span></div>', unsafe_allow_html=True)
    sleep_bars = [5.5, 7.2, 4.8, 7.8, 6.5, 4.5, 7.6, 5.8, sleep_hrs]
    bar_colors = ['#95D5B2'] * 8 + ['#1B4332']
    fig_sleep = go.Figure(go.Bar(
        x=[f"D{i+1}" for i in range(9)], y=sleep_bars,
        marker_color=bar_colors, width=0.55
    ))
    fig_sleep.update_layout(
        height=115, margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, showgrid=False)
    )
    st.plotly_chart(fig_sleep, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        st.markdown('<div class="health-card"><div class="card-title">💧 Hydration Goal</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            values=[water_glasses, max(8 - water_glasses, 0)],
            hole=0.72, marker_colors=['#40916C', '#E0ECE6'],
            textinfo='none', hoverinfo='none'
        ))
        fig_donut.update_layout(
            height=110, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
            paper_bgcolor='#FFFFFF',
            annotations=[dict(text=f"<b>{water_glasses}/8</b>", x=0.5, y=0.5, font=dict(size=16, color='#1B4332'), showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<p style='color:#1B4332; font-size:13px; font-weight:600; text-align:center; margin:4px 0 0 0;'>{water_glasses}/8 Glasses • Track Water</p></div>", unsafe_allow_html=True)

    with sub_c2:
        st.markdown("""
        <div class="health-card">
            <div class="card-title">🏆 Upcoming Challenge</div>
            <div style="background:#1B4332; color:white; padding:12px; border-radius:10px; font-weight:600; margin:12px 0; text-align:center;">
                📅 Weekend Walk
            </div>
        </div>
        """, unsafe_allow_html=True)
        if "joined" not in st.session_state:
            st.session_state.joined = False
        if st.button("✅ Joined! (10k Steps)" if st.session_state.joined else "Join Challenge", key="join_btn", use_container_width=True):
            st.session_state.joined = not st.session_state.joined
            st.rerun()
