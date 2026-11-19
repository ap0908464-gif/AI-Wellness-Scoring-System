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
<div class="health-card" style="background: linear-gradient(135deg, #1B4332, #2D6A4F); border:none;">
    <div style="color:#D8F3DC !important; font-size:13px; font-weight:600;">PROFILE: {display_name.upper()} • AGE {age} • BMI {bmi} ({bmi_status.upper()})</div>
    <div style="color:#FFFFFF !important; font-size:34px; font-weight:800; margin-top:4px;">🧠 AI Wellness Score: {ai_score} / 100</div>
    <div style="color:#95D5B2 !important; font-size:15px; font-weight:700; margin-top:2px;">Status: {status_label}</div>
</div>
""", unsafe_allow_html=True)

# 7. Interactive Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💡 AI Advisor & Report", "🧠 ML Model Insights"])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        step_pct = min(int((steps / 10000) * 100), 100)
        st.markdown(f"""
        <div class="health-card">
            <div class="card-title">🏃 Daily Steps</div>
            <div class="big-num">{steps:,}</div>
            <div class="progress-bg"><div class="progress-fill" style="width:{step_pct}%;"></div></div>
            <div class="sub-text">Goal: 10,000 • <b>{step_pct}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        cal_pct = min(int((calories / 1600) * 100), 100)
        st.markdown(f"""
        <div class="health-card">
            <div class="card-title">🔥 Calories Burned</div>
            <div class="big-num" style="color:#40916C !important;">{calories:,} <span class="unit-text">kcal</span></div>
            <div class="progress-bg"><div class="progress-fill" style="width:{cal_pct}%;"></div></div>
            <div class="sub-text">Active: {int(calories * 0.68)} kcal</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        hr_pct = min(max(int(((heart_rate - 45) / 85) * 100), 10), 100)
        st.markdown(f"""
        <div class="health-card">
            <div class="card-title">💚 Heart Rate</div>
            <div class="big-num">{heart_rate} <span class="unit-text">bpm</span></div>
            <div class="progress-bg"><div class="progress-fill" style="width:{hr_pct}%;"></div></div>
            <div class="sub-text">Normal: 60–80 bpm</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        stress_pct = stress_level * 10
        st.markdown(f"""
        <div class="health-card">
            <div class="card-title">🧘 Stress Index</div>
            <div class="big-num">{stress_level}<span class="unit-text">/10</span></div>
            <div class="progress-bg"><div class="progress-fill" style="width:{stress_pct}%;"></div></div>
            <div class="sub-text">{"Calm State" if stress_level <= 4 else ("Moderate" if stress_level <= 7 else "High Stress")}</div>
        </div>
        """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown('<div class="health-card"><div class="card-title">📈 Weekly Activity Trend</div>', unsafe_allow_html=True)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        steps_trend = [6800, 8200, 7100, 9400, 7900, 10200, steps]
        active_mins = [35, 52, 40, 65, 48, 75, int(steps / 130)]

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=days, y=steps_trend, mode='lines+markers', name='Steps',
            line=dict(color='#52B788', width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(82, 183, 136, 0.22)'
        ))
        fig_wave.add_trace(go.Scatter(
            x=days, y=[m * 100 for m in active_mins], mode='lines+markers', name='Activity Index',
            line=dict(color='#1B4332', width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(27, 67, 50, 0.10)'
        ))
        fig_wave.update_layout(
            height=260, margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
            font=dict(color='#1B4332', size=12),
            legend=dict(orientation="h", y=1.15, x=0.1, font=dict(color='#1B4332')),
            xaxis=dict(tickfont=dict(color='#1B4332')),
            yaxis=dict(gridcolor='#E6EFEA', tickfont=dict(color='#1B4332'))
        )
        st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown(f'<div class="health-card"><div class="card-title">😴 Sleep Quality • {sleep_hrs} hrs Today</div>', unsafe_allow_html=True)
        sleep_bars = [6.0, 7.2, 5.5, 7.8, 6.8, 8.0, sleep_hrs]
        bar_colors = ['#95D5B2'] * 6 + ['#1B4332']
        fig_sleep = go.Figure(go.Bar(x=days, y=sleep_bars, marker_color=bar_colors, width=0.5))
        fig_sleep.update_layout(
            height=120, margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
            xaxis=dict(tickfont=dict(color='#1B4332')),
            yaxis=dict(showticklabels=False, showgrid=False)
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
                height=105, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                paper_bgcolor='#FFFFFF',
                annotations=[dict(text=f"<b>{water_glasses}/8</b>", x=0.5, y=0.5, font=dict(size=16, color='#1B4332'), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f"<p style='color:#1B4332; font-size:12px; font-weight:600; text-align:center; margin:2px 0 0 0;'>{water_glasses}/8 Glasses Target</p></div>", unsafe_allow_html=True)

        with sub_c2:
            st.markdown("""
            <div class="health-card">
                <div class="card-title">🏆 Daily Challenge</div>
                <div style="background:#1B4332; color:#FFFFFF !important; padding:10px; border-radius:10px; font-weight:600; margin:10px 0; text-align:center; font-size:13px;">
                    🚶 10,000 Steps Sprint
                </div>
            </div>
            """, unsafe_allow_html=True)
            if "joined" not in st.session_state:
                st.session_state.joined = False
            if st.button("✅ Challenge Active" if st.session_state.joined else "Join Challenge", key="join_btn", use_container_width=True):
                st.session_state.joined = not st.session_state.joined
                st.rerun()

# ==================== TAB 2: AI ADVISOR & REPORT ====================
with tab2:
    adv_col1, adv_col2 = st.columns([1.3, 1])
    with adv_col1:
        st.markdown('<div class="health-card"><div class="card-title">🩺 Personalized AI Health Recommendations</div>', unsafe_allow_html=True)
        tips = []
        if sleep_hrs < 7.0:
            tips.append("😴 **Sleep Optimization:** Your sleep is below 7 hours. Aim for 7–8 hours for full cognitive and physical recovery.")
        if water_glasses < 8:
            tips.append(f"💧 **Hydration Boost:** You logged {water_glasses}/8 glasses. Drink {8 - water_glasses} more glasses today.")
        if steps < 8000:
            tips.append(f"🚶 **Movement Target:** Add {8000 - steps:,} more steps (approx. a 20-minute brisk walk) to reach the optimal activity zone.")
        if stress_level >= 6:
            tips.append("🧘 **Stress Management:** Elevated stress detected. Try 10 minutes of deep breathing or a short screen break.")
        if not (18.5 <= bmi <= 24.9):
            tips.append(f"🥗 **BMI Guidance:** Your BMI is {bmi} ({bmi_status}). Balanced nutrition and regular cardio are recommended.")
        if not (60 <= heart_rate <= 85):
            tips.append("❤️ **Heart Rate Check:** Resting heart rate is outside the typical 60–85 bpm range. Monitor after rest.")

        if not tips:
            st.markdown('<div class="tip-box">🌟 <b>Excellent Balance!</b> All your vital and lifestyle parameters are within optimal ranges.</div>', unsafe_allow_html=True)
        else:
            for t in tips:
                st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with adv_col2:
        st.markdown('<div class="health-card"><div class="card-title">📄 Export & Session Log</div>', unsafe_allow_html=True)
        report_text = f"""AI-BASED WELLNESS SCORING SYSTEM — HEALTH REPORT
Generated On: {datetime.now().strftime('%d %b %Y, %I:%M %p')}
====================================================
Participant: {display_name} | Age: {age}
Height: {height_cm} cm | Weight: {weight_kg} kg | BMI: {bmi} ({bmi_status})
----------------------------------------------------
Daily Steps: {steps:,}
Calories Burned: {calories:,} kcal
Resting Heart Rate: {heart_rate} bpm
Sleep Duration: {sleep_hrs} hrs
Water Intake: {water_glasses}/8 glasses
Stress Level: {stress_level}/10
====================================================
AI PREDICTED WELLNESS SCORE: {ai_score} / 100 ({status_label})
====================================================
"""
        st.download_button(
            label="📥 Download Wellness Report (.txt)",
            data=report_text,
            file_name=f"{display_name.replace(' ', '_')}_Wellness_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

        if "history_log" not in st.session_state:
            st.session_state.history_log = []

        if st.button("➕ Save Reading to Session Log", use_container_width=True):
            st.session_state.history_log.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "User": display_name,
                "Steps": steps,
                "Sleep (h)": sleep_hrs,
                "AI Score": ai_score
            })

        if st.session_state.history_log:
            st.dataframe(pd.DataFrame(st.session_state.history_log), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 3: ML MODEL INSIGHTS ====================
with tab3:
    m_col1, m_col2 = st.columns([1.2, 1])
    with m_col1:
        st.markdown('<div class="health-card"><div class="card-title">📊 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        fig_imp = px.bar(
            feature_imp_df, x='Importance (%)', y='Parameter', orientation='h',
            color_discrete_sequence=['#2D6A4F']
        )
        fig_imp.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
            font=dict(color='#1B4332')
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="health-card">
            <div class="card-title">⚙️ Model Architecture & Accuracy</div>
            <p class="sub-text"><b>Algorithm:</b> Random Forest Regressor (120 Estimators)</p>
            <p class="sub-text"><b>Training Samples:</b> 1,500 Synthetic Clinical-Wellness Records</p>
            <p class="sub-text"><b>Test R² Accuracy Score:</b> <span style="color:#2D6A4F !important; font-weight:800; font-size:16px;">{model_accuracy}%</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="health-card"><div class="card-title">🗂️ Sample Training Dataset</div>', unsafe_allow_html=True)
        st.dataframe(sample_dataset, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
