import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="AI Wellness Scoring System", page_icon="🩺", layout="wide")

if "steps_val" not in st.session_state:
    st.session_state.steps_val = 7800
if "water_val" not in st.session_state:
    st.session_state.water_val = 6
if "joined" not in st.session_state:
    st.session_state.joined = False
if "history_log" not in st.session_state:
    st.session_state.history_log = []

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F4F7F6 !important; color: #1B4332 !important;
    }
    label, p, h1, h2, h3, h4, span { color: #1B4332 !important; }
    .stTextInput input, .stNumberInput input {
        background-color: #FFFFFF !important; color: #1B4332 !important;
        border: 1px solid #B7D4C5 !important; border-radius: 10px !important;
    }
    .health-card {
        background-color: #FFFFFF; padding: 16px 18px; border-radius: 16px;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.06);
        border: 1px solid #DCE8E2; margin-bottom: 12px;
    }
    .card-title { font-size: 13px; font-weight: 700; color: #2D6A4F !important; text-transform: uppercase; margin-bottom: 6px; }
    .big-num { font-size: 30px; font-weight: 800; color: #1B4332 !important; line-height: 1.1; }
    .sub-text { font-size: 13px; color: #4A6358 !important; margin-top: 8px; }
    .progress-bg { background-color: #E0ECE6; border-radius: 10px; height: 8px; width: 100%; margin-top: 10px; }
    .progress-fill { background: linear-gradient(90deg, #52B788, #1B4332); height: 8px; border-radius: 10px; }
    .tip-box {
        background-color: #FFFFFF; border-left: 4px solid #2D6A4F;
        padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 14px;
    }
    .stTabs [data-baseweb="tab"] { background-color: #E3EDE8; border-radius: 10px; padding: 8px 14px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1B4332 !important; color: #FFFFFF !important; }
    .stTabs [aria-selected="true"] p { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

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

    score = np.clip(
        np.where((sleep >= 7) & (sleep <= 9), 20, np.where(sleep >= 5.5, 12, 5)) +
        np.where(glasses >= 8, 15, glasses * 1.6) +
        np.where(steps >= 8000, 20, (steps / 8000) * 18) +
        np.where((hr >= 60) & (hr <= 80), 15, 7) +
        np.where((bmi >= 18.5) & (bmi <= 24.9), 15, 8) +
        np.where(calories >= 1100, 8, 4) +
        (11 - stress) * 0.7 + np.random.normal(0, 1.2, n), 0, 100
    )

    df = pd.DataFrame({
        'Age': age, 'BMI': bmi, 'Sleep_Hrs': sleep, 'Water_Glasses': glasses,
        'Daily_Steps': steps, 'Heart_Rate': hr, 'Calories': calories, 'Stress_Level': stress
    })
    X_train, X_test, y_train, y_test = train_test_split(df, score, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    acc = round(r2_score(y_test, model.predict(X_test)) * 100, 2)
    imp = pd.DataFrame({'Parameter': df.columns, 'Importance (%)': np.round(model.feature_importances_ * 100, 1)}).sort_values('Importance (%)')
    return model, acc, imp, df.head(8)

model, model_accuracy, feature_imp_df, sample_dataset = train_wellness_model()

st.markdown("<h2 style='margin-bottom:2px;'>🩺 AI-Based Wellness Scoring System</h2>", unsafe_allow_html=True)
st.caption("Interactive Health Analytics, Live Trackers & Predictive Wellness Engine")

with st.expander("✏️ Enter / Edit Your Health Parameters (Tap Here)", expanded=True):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    user_name = c1.text_input("Your Name", value="", placeholder="Enter your name")
    age = c2.number_input("Age", 12, 100, 24)
    height_cm = c3.number_input("Height (cm)", 120, 220, 170)
    weight_kg = c4.number_input("Weight (kg)", 30, 180, 65)

    s1, s2, s3 = st.columns(3)
    steps = s1.slider("Daily Steps", 0, 16000, key="steps_val", step=100)
    sleep_hrs = s1.slider("Sleep (hrs)", 3.0, 11.0, 7.5, 0.5)
    calories = s2.slider("Calories (kcal)", 300, 2500, 1250, 50)
    water_glasses = s2.slider("Water (Glasses)", 0, 14, key="water_val", step=1)
    heart_rate = s3.slider("Heart Rate (bpm)", 50, 130, 72)
    stress_level = s3.slider("Stress (1-10)", 1, 10, 4)

display_name = user_name.strip() if user_name.strip() else "Guest User"
bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
bmi_status = "Healthy" if 18.5 <= bmi <= 24.9 else ("Underweight" if bmi < 18.5 else "Overweight")

input_df = pd.DataFrame([{'Age': age, 'BMI': bmi, 'Sleep_Hrs': sleep_hrs, 'Water_Glasses': water_glasses, 'Daily_Steps': steps, 'Heart_Rate': heart_rate, 'Calories': calories, 'Stress_Level': stress_level}])
ai_score = round(float(model.predict(input_df)[0]), 1)
status_label = "Optimal Wellness" if ai_score >= 80 else ("Moderate Wellness" if ai_score >= 60 else "Needs Attention")

banner_html = (
    f'<div class="health-card" style="background: linear-gradient(135deg, #1B4332, #2D6A4F); border:none;">'
    f'<div style="color:#D8F3DC !important; font-size:13px; font-weight:600;">PROFILE: {display_name.upper()} • AGE {age} • BMI {bmi} ({bmi_status.upper()})</div>'
    f'<div style="color:#FFFFFF !important; font-size:32px; font-weight:800; margin-top:4px;">🧠 AI Wellness Score: {ai_score} / 100</div>'
    f'<div style="color:#95D5B2 !important; font-size:15px; font-weight:700;">Status: {status_label}</div>'
    f'</div>'
)
st.markdown(banner_html, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔔 Step & Water Trackers", "💡 AI Advisor & Report", "🧠 ML Insights"])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    step_pct = min(int((steps / 10000) * 100), 100)
    cal_pct = min(int((calories / 1600) * 100), 100)
    hr_pct = min(max(int(((heart_rate - 45) / 85) * 100), 10), 100)
    stress_pct = stress_level * 10

    m1.markdown(f'<div class="health-card"><div class="card-title">🏃 Daily Steps</div><div class="big-num">{steps:,}</div><div class="progress-bg"><div class="progress-fill" style="width:{step_pct}%;"></div></div><div class="sub-text">Goal: 10,000 ({step_pct}%)</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="health-card"><div class="card-title">🔥 Calories</div><div class="big-num">{calories:,} kcal</div><div class="progress-bg"><div class="progress-fill" style="width:{cal_pct}%;"></div></div><div class="sub-text">Active: {int(calories*0.68)} kcal</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="health-card"><div class="card-title">💚 Heart Rate</div><div class="big-num">{heart_rate} bpm</div><div class="progress-bg"><div class="progress-fill" style="width:{hr_pct}%;"></div></div><div class="sub-text">Normal: 60–80 bpm</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="health-card"><div class="card-title">🧘 Stress</div><div class="big-num">{stress_level}/10</div><div class="progress-bg"><div class="progress-fill" style="width:{stress_pct}%;"></div></div><div class="sub-text">{"Calm" if stress_level<=4 else "Elevated"}</div></div>', unsafe_allow_html=True)

    g1, g2 = st.columns([1.2, 1])
    with g1:
        st.markdown('<div class="health-card"><div class="card-title">📈 Weekly Activity Trend</div>', unsafe_allow_html=True)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig_wave = go.Figure()
        fig_wave.add_scatter(x=days, y=[6800, 8200, 7100, 9400, 7900, 10200, steps], mode='lines+markers', name='Steps', line=dict(color='#52B788', width=3, shape='spline'), fill='tozeroy')
        fig_wave.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
        st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown(f'<div class="health-card"><div class="card-title">😴 Sleep Quality ({sleep_hrs} hrs)</div>', unsafe_allow_html=True)
        fig_sleep = go.Figure(go.Bar(x=days, y=[6.0, 7.2, 5.5, 7.8, 6.8, 8.0, sleep_hrs], marker_color=['#95D5B2']*6 + ['#1B4332']))
        fig_sleep.update_layout(height=110, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
        st.plotly_chart(fig_sleep, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f'<div class="health-card"><div class="card-title">💧 Hydration</div><div class="big-num">{water_glasses}/8</div><div class="sub-text">Glasses Today</div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown('<div class="health-card"><div class="card-title">🏆 Challenge</div><div class="sub-text">10k Steps Sprint</div></div>', unsafe_allow_html=True)
            if st.button("✅ Active" if st.session_state.joined else "Join Challenge", use_container_width=True):
                st.session_state.joined = not st.session_state.joined
                st.rerun()

with tab2:
    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="health-card"><div class="card-title">💧 Water Logger & Reminder</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("💧 +1 Glass", use_container_width=True):
            st.session_state.water_val = min(st.session_state.water_val + 1, 14)
            st.rerun()
        if b2.button("🔄 Reset Water", use_container_width=True):
            st.session_state.water_val = 0
            st.rerun()

        components.html("""
        <div style="font-family:sans-serif;background:#F0F7F4;padding:12px;border-radius:10px;border:1px solid #B7D4C5;">
            <select id="intSel" style="width:100%;padding:8px;border-radius:8px;margin-bottom:8px;">
                <option value="10000">Every 10 Seconds (Viva Demo)</option>
                <option value="1800000">Every 30 Minutes</option>
                <option value="3600000">Every 60 Minutes</option>
            </select>
            <button onclick="startRem()" style="width:100%;background:#1B4332;color:white;border:none;padding:10px;border-radius:8px;font-weight:700;cursor:pointer;">🔔 Enable Water Notification</button>
            <p id="rStat" style="font-size:12px;color:#2D6A4F;margin-top:8px;font-weight:600;">Status: Off</p>
        </div>
        <script>
        let tid = null;
        function beep(){try{const c=new(window.AudioContext||window.webkitAudioContext)(),o=c.createOscillator();o.frequency.value=700;o.connect(c.destination);o.start();o.stop(c.currentTime+0.3);}catch(e){}}
        function fire(){beep();document.getElementById("rStat").innerText="💧 ALERT! Drink 1 glass of water ("+new Date().toLocaleTimeString()+")";if("Notification"in window&&Notification.permission==="granted")new Notification("💧 Water Reminder",{body:"Time to drink 1 glass of water!"});}
        function startRem(){const ms=parseInt(document.getElementById("intSel").value);if("Notification"in window&&Notification.permission!=="granted")Notification.requestPermission();if(tid)clearInterval(tid);document.getElementById("rStat").innerText="✅ Active ("+(ms/1000)+"s timer)";tid=setInterval(fire,ms);}
        </script>
        """, height=155)
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="health-card"><div class="card-title">🚶 Live Step Counter</div>', unsafe_allow_html=True)
        sb1, sb2, sb3 = st.columns(3)
        if sb1.button("➕ 250", use_container_width=True):
            st.session_state.steps_val = min(st.session_state.steps_val + 250, 16000)
            st.rerun()
        if sb2.button("➕ 1000", use_container_width=True):
            st.session_state.steps_val = min(st.session_state.steps_val + 1000, 16000)
            st.rerun()
        if sb3.button("🔄 Reset", use_container_width=True):
            st.session_state.steps_val = 0
            st.rerun()

        components.html("""
        <div style="font-family:sans-serif;background:#F0F7F4;padding:12px;border-radius:10px;border:1px solid #B7D4C5;">
            <div style="font-size:22px;font-weight:800;color:#1B4332;margin-bottom:6px;">Live Steps: <span id="sc">0</span></div>
            <button onclick="startPed()" style="width:100%;background:#2D6A4F;color:white;border:none;padding:10px;border-radius:8px;font-weight:700;cursor:pointer;">🏃 Start Phone Step Sensor</button>
            <p id="pStat" style="font-size:12px;color:#4A6358;margin-top:8px;">Walk with phone in hand to count steps.</p>
        </div>
        <script>
        let ls=0,lt=0;
        function mov(e){const a=e.accelerationIncludingGravity;if(!a)return;const m=Math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z),n=Date.now();if(m>12.2&&(n-lt)>380){ls++;lt=n;document.getElementById("sc").innerText=ls;}}
        function startPed(){if(typeof DeviceMotionEvent!=='undefined'&&typeof DeviceMotionEvent.requestPermission==='function'){DeviceMotionEvent.requestPermission().then(s=>{if(s==='granted'){window.addEventListener('devicemotion',mov);document.getElementById("pStat").innerText="✅ Sensor Active!";}});}else if('ondevicemotion'in window){window.addEventListener('devicemotion',mov);document.getElementById("pStat").innerText="✅ Sensor Active!";}else{document.getElementById("pStat").innerText="⚠️ Use buttons on desktop.";}}
        </script>
        """, height=155)
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    a1, a2 = st.columns([1.3, 1])
    with a1:
        st.markdown('<div class="health-card"><div class="card-title">🩺 AI Health Recommendations</div>', unsafe_allow_html=True)
        tips = []
        if sleep_hrs < 7.0: tips.append("😴 **Sleep:** Aim for 7–8 hours of sleep for better recovery.")
        if water_glasses < 8: tips.append(f"💧 **Hydration:** Drink {8 - water_glasses} more glasses today.")
        if steps < 8000: tips.append(f"🚶 **Activity:** Add {8000 - steps:,} more steps today.")
        if stress_level >= 6: tips.append("🧘 **Stress:** Try 10 minutes of deep breathing.")
        if not (18.5 <= bmi <= 24.9): tips.append(f"🥗 **BMI ({bmi}):** Maintain a balanced diet & cardio routine.")
        for t in (tips or ["🌟 **Optimal:** All parameters are in healthy ranges!"]):
            st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="health-card"><div class="card-title">📄 Report & History</div>', unsafe_allow_html=True)
        rep = f"WELLNESS REPORT\nUser: {display_name} | Age: {age} | BMI: {bmi} ({bmi_status})\nSteps: {steps} | Sleep: {sleep_hrs}h | Water: {water_glasses}/8 | HR: {heart_rate} | Stress: {stress_level}/10\nAI SCORE: {ai_score}/100 ({status_label})"
        st.download_button("📥 Download Report (.txt)", rep, f"{display_name}_Report.txt", use_container_width=True)
        if st.button("➕ Log Current Score", use_container_width=True):
            st.session_state.history_log.append({"Time": datetime.now().strftime("%H:%M:%S"), "User": display_name, "Steps": steps, "Water": water_glasses, "Score": ai_score})
        if st.session_state.history_log:
            st.dataframe(pd.DataFrame(st.session_state.history_log), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="health-card"><div class="card-title">📊 Feature Importance</div>', unsafe_allow_html=True)
        fig_imp = px.bar(feature_imp_df, x='Importance (%)', y='Parameter', orientation='h', color_discrete_sequence=['#2D6A4F'])
        fig_imp.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="health-card"><div class="card-title">⚙️ Model Info</div><p class="sub-text"><b>Model:</b> Random Forest Regressor<br><b>R² Accuracy:</b> {model_accuracy}%</p></div>', unsafe_allow_html=True)
        st.dataframe(sample_dataset, use_container_width=True, hide_index=True)
