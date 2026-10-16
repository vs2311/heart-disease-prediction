from datetime import datetime
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
BASE = Path(__file__).parent
 
# ------------------------------------------------------------------ #
# Page config + styling
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
 
    .hero {
        background: linear-gradient(120deg, #ff4b4b 0%, #c2185b 55%, #6a1b9a 100%);
        padding: 1.8rem 2rem; border-radius: 18px; color: white;
        box-shadow: 0 8px 24px rgba(194, 24, 91, 0.25); margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: white;}
    .hero p {margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem;}
 
    .verdict {
        padding: 1.1rem 1.4rem; border-radius: 14px; font-size: 1.15rem;
        font-weight: 600; margin-bottom: .8rem;
    }
    .verdict.high {background: rgba(255, 75, 75, .12); border-left: 6px solid #ff4b4b; color: #d32f2f;}
    .verdict.low  {background: rgba(46, 160, 67, .12); border-left: 6px solid #2ea043; color: #2e7d32;}
 
    .chip {
        display: inline-block; padding: .3rem .8rem; margin: .2rem .3rem .2rem 0;
        border-radius: 999px; font-size: .88rem; font-weight: 500;
    }
    .chip.warn {background: rgba(255, 170, 0, .16); color: #b26a00; border: 1px solid rgba(255,170,0,.5);}
    .chip.ok   {background: rgba(46, 160, 67, .14); color: #2e7d32; border: 1px solid rgba(46,160,67,.4);}
 
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08); padding: .8rem 1rem;
        border-radius: 12px; border: 1px solid rgba(128,128,128,.15);
    }
    .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ------------------------------------------------------------------ #
# Load artifacts
# ------------------------------------------------------------------ #
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "knn_heart_model.pkl")
    scaler = joblib.load(BASE / "heart_scaler.pkl")
    columns = joblib.load(BASE / "heart_columns.pkl")
    return model, scaler, list(columns)
 
 
@st.cache_data
def load_data():
    try:
        return pd.read_csv(BASE / "heart.csv")
    except Exception:
        return None
 
 
model, scaler, expected_columns = load_artifacts()
data = load_data()
 
CAT_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUM_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
 
LABELS = {
    "ChestPainType": {"ATA": "Atypical Angina", "NAP": "Non-Anginal Pain",
                      "TA": "Typical Angina", "ASY": "Asymptomatic"},
    "Sex": {"M": "Male", "F": "Female"},
    "ExerciseAngina": {"Y": "Yes", "N": "No"},
}
 
 
def encode(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode and align to the exact columns used in training."""
    enc = pd.get_dummies(raw_df, columns=CAT_COLS)
    return enc.reindex(columns=expected_columns, fill_value=0).astype(float)
 
 
def predict_proba(raw_df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(encode(raw_df)))[:, 1]
 
 
# ------------------------------------------------------------------ #
# Session state defaults + presets
# ------------------------------------------------------------------ #
DEFAULT = dict(age=40, sex="M", cp="ATA", bp=120, chol=200, fbs=0,
               ecg="Normal", hr=150, angina="Y", oldpeak=1.0, slope="Flat")
PRESETS = {
    "healthy": dict(age=35, sex="F", cp="ATA", bp=110, chol=180, fbs=0,
                    ecg="Normal", hr=175, angina="N", oldpeak=0.0, slope="Up"),
    "risk": dict(age=62, sex="M", cp="ASY", bp=150, chol=260, fbs=1,
                 ecg="ST", hr=110, angina="Y", oldpeak=3.0, slope="Flat"),
    "default": DEFAULT,
}
 
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)
 
 
def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v
 
 
# ------------------------------------------------------------------ #
# Sidebar – patient input
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 🧑‍⚕️ Patient Details")
    st.caption("The prediction updates live as you change values.")
 
    c1, c2, c3 = st.columns(3)
    c1.button("🟢 Healthy", on_click=apply_preset, args=("healthy",), use_container_width=True)
    c2.button("🔴 At-risk", on_click=apply_preset, args=("risk",), use_container_width=True)
    c3.button("↺ Reset", on_click=apply_preset, args=("default",), use_container_width=True)
 
    st.markdown("---")
    st.markdown("##### 👤 Demographics")
    age = st.slider("Age", 18, 100, key="age")
    sex = st.radio("Sex", ["M", "F"], key="sex", horizontal=True,
                   format_func=lambda x: LABELS["Sex"][x])
 
    st.markdown("##### 🩺 Clinical Measurements")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], key="cp",
                              format_func=lambda x: f"{x} – {LABELS['ChestPainType'][x]}")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, key="bp")
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, key="chol")
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1], key="fbs",
                          horizontal=True, format_func=lambda x: "Yes" if x else "No")
 
    st.markdown("##### 📈 ECG & Exercise Test")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key="ecg")
    max_hr = st.slider("Maximum Heart Rate", 60, 220, key="hr")
    exercise_angina = st.radio("Exercise-Induced Angina", ["Y", "N"], key="angina",
                               horizontal=True, format_func=lambda x: LABELS["ExerciseAngina"][x])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, step=0.1, key="oldpeak")
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="slope")
 
# ------------------------------------------------------------------ #
# Build patient + prediction
# ------------------------------------------------------------------ #
patient = dict(
    Age=age, Sex=sex, ChestPainType=chest_pain, RestingBP=resting_bp,
    Cholesterol=cholesterol, FastingBS=fasting_bs, RestingECG=resting_ecg,
    MaxHR=max_hr, ExerciseAngina=exercise_angina, Oldpeak=oldpeak, ST_Slope=st_slope,
)
patient_df = pd.DataFrame([patient])
patient_scaled = scaler.transform(encode(patient_df))
prediction = int(model.predict(patient_scaled)[0])
prob = float(model.predict_proba(patient_scaled)[0][1])
k = getattr(model, "n_neighbors", 5)
 
# ------------------------------------------------------------------ #
# Helpers for charts
# ------------------------------------------------------------------ #
def gauge(p: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Estimated Risk", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#222", "thickness": 0.25},
            "steps": [
                {"range": [0, 35], "color": "#2ea043"},
                {"range": [35, 65], "color": "#f0b429"},
                {"range": [65, 100], "color": "#ff4b4b"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "value": p * 100},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig
 
 
def risk_flags(p: dict):
    flags = []
    if p["RestingBP"] >= 140:
        flags.append(("warn", f"High blood pressure ({p['RestingBP']} mm Hg)"))
    if p["Cholesterol"] >= 240:
        flags.append(("warn", f"High cholesterol ({p['Cholesterol']} mg/dL)"))
    if p["FastingBS"] == 1:
        flags.append(("warn", "Elevated fasting blood sugar"))
    if p["ExerciseAngina"] == "Y":
        flags.append(("warn", "Exercise-induced angina"))
    if p["Oldpeak"] >= 2:
        flags.append(("warn", f"Significant ST depression ({p['Oldpeak']:.1f})"))
    if p["ST_Slope"] in ("Flat", "Down"):
        flags.append(("warn", f"{p['ST_Slope']} ST slope"))
    if p["ChestPainType"] == "ASY":
        flags.append(("warn", "Asymptomatic chest pain type"))
    if p["MaxHR"] < 0.7 * (220 - p["Age"]):
        flags.append(("warn", "Low max heart rate for age"))
    if p["RestingECG"] != "Normal":
        flags.append(("warn", f"Abnormal resting ECG ({p['RestingECG']})"))
    if not flags:
        flags.append(("ok", "No common risk markers flagged"))
    return flags
 
 
def build_report() -> str:
    lines = [
        "HEART DISEASE RISK REPORT",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "-" * 40,
        *[f"{key}: {val}" for key, val in patient.items()],
        "-" * 40,
        f"Model: K-Nearest Neighbors (k={k})",
        f"Prediction: {'Higher risk' if prediction == 1 else 'Lower risk'}",
        f"Neighbour-vote risk score: {prob * 100:.1f}%",
        "",
        "For educational/demo purposes only. Not a medical diagnosis.",
    ]
    return "\n".join(lines)
 
 
# ------------------------------------------------------------------ #
# Hero
# ------------------------------------------------------------------ #
st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Risk Predictor</h1>
        <p>An interactive machine-learning app powered by K-Nearest Neighbors.
        Adjust the patient profile in the sidebar and watch the risk update in real time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info("🎓 Educational/demo project only — not a medical diagnosis.")
 
tab_pred, tab_whatif, tab_nn, tab_data, tab_about = st.tabs(
    ["🩺 Prediction", "🔬 What-If Explorer", "👥 Similar Patients", "📊 Data Insights", "ℹ️ About"]
)
 
# ------------------------------------------------------------------ #
# TAB 1 – Prediction
# ------------------------------------------------------------------ #
with tab_pred:
    left, right = st.columns([1, 1.2], gap="large")
 
    with left:
        st.plotly_chart(gauge(prob), use_container_width=True)
 
    with right:
        if prediction == 1:
            st.markdown('<div class="verdict high">⚠️ The model predicts a HIGHER risk of heart disease.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict low">✅ The model predicts a LOWER risk of heart disease.</div>',
                        unsafe_allow_html=True)
 
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk score", f"{prob * 100:.0f}%")
        m2.metric("Neighbours (k)", k)
        m3.metric("Voting 'disease'", f"{round(prob * k)} / {k}")
 
        st.markdown("**Risk markers in this profile**")
        chips = "".join(f'<span class="chip {t}">{txt}</span>' for t, txt in risk_flags(patient))
        st.markdown(chips, unsafe_allow_html=True)
        st.caption("Markers are simple clinical rules of thumb shown for context; "
                   "they are separate from the model's decision.")
 
    st.markdown("---")
 
    if data is not None:
        st.markdown("#### Patient profile vs. dataset averages")
        rng = data[NUM_COLS].agg(["min", "max"])
        norm = lambda s: ((s - rng.loc["min"]) / (rng.loc["max"] - rng.loc["min"])).clip(0, 1)
        healthy_avg = norm(data[data["HeartDisease"] == 0][NUM_COLS].mean())
        disease_avg = norm(data[data["HeartDisease"] == 1][NUM_COLS].mean())
        patient_norm = norm(pd.Series({c: patient[c] for c in NUM_COLS}))
 
        radar = go.Figure()
        for name, vals, color in [
            ("Avg. healthy", healthy_avg, "#2ea043"),
            ("Avg. heart disease", disease_avg, "#ff4b4b"),
            ("This patient", patient_norm, "#1f77ff"),
        ]:
            radar.add_trace(go.Scatterpolar(
                r=list(vals.values) + [vals.values[0]],
                theta=NUM_COLS + [NUM_COLS[0]],
                name=name, fill="toself" if name == "This patient" else None,
                line=dict(color=color, width=3 if name == "This patient" else 2),
                opacity=0.85,
            ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            height=420, margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(radar, use_container_width=True)
 
    st.download_button("📄 Download report", build_report(),
                       file_name="heart_risk_report.txt", mime="text/plain")
 
# ------------------------------------------------------------------ #
# TAB 2 – What-if
# ------------------------------------------------------------------ #
with tab_whatif:
    st.markdown("#### How does risk change if one value changes?")
    st.caption("Every other value stays as set in the sidebar. The red dot is the current patient.")
 
    sweeps = {
        "Age": (18, 100, 1),
        "RestingBP": (80, 200, 2),
        "Cholesterol": (100, 600, 5),
        "MaxHR": (60, 220, 2),
        "Oldpeak": (0.0, 6.0, 0.1),
    }
    feat = st.selectbox("Feature to vary", list(sweeps.keys()))
    lo, hi, step = sweeps[feat]
    grid = np.arange(lo, hi + step, step)
 
    sweep_df = pd.concat([patient_df] * len(grid), ignore_index=True)
    sweep_df[feat] = grid
    probs = predict_proba(sweep_df) * 100
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=probs, mode="lines", name="Risk",
                             line=dict(color="#c2185b", width=3, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(194,24,91,.12)"))
    fig.add_trace(go.Scatter(x=[patient[feat]], y=[prob * 100], mode="markers",
                             name="Current patient",
                             marker=dict(size=14, color="#ff4b4b", line=dict(color="white", width=2))))
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% decision line")
    fig.update_layout(height=420, xaxis_title=feat, yaxis_title="Risk score (%)",
                      yaxis=dict(range=[0, 105]), margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("#### Categorical comparison")
    cat_feat = st.selectbox("Category to vary", ["ChestPainType", "ST_Slope", "RestingECG", "ExerciseAngina", "Sex"])
    options = {"ChestPainType": ["ATA", "NAP", "TA", "ASY"], "ST_Slope": ["Up", "Flat", "Down"],
               "RestingECG": ["Normal", "ST", "LVH"], "ExerciseAngina": ["Y", "N"], "Sex": ["M", "F"]}[cat_feat]
    cat_df = pd.concat([patient_df] * len(options), ignore_index=True)
    cat_df[cat_feat] = options
    cat_probs = predict_proba(cat_df) * 100
    bar = px.bar(x=options, y=cat_probs, labels={"x": cat_feat, "y": "Risk score (%)"},
                 color=cat_probs, color_continuous_scale=["#2ea043", "#f0b429", "#ff4b4b"],
                 range_color=[0, 100], text=[f"{v:.0f}%" for v in cat_probs])
    bar.update_layout(height=340, coloraxis_showscale=False, yaxis=dict(range=[0, 110]),
                      margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(bar, use_container_width=True)
 
# ------------------------------------------------------------------ #
# TAB 3 – Similar patients (explainability for KNN)
# ------------------------------------------------------------------ #
with tab_nn:
    st.markdown("#### Why did the model say that?")
    st.write(
        f"KNN decides by looking at the **{k} most similar patients** and taking a majority vote. "
        "Here they are, found in the dataset:"
    )
    if data is None:
        st.warning("`heart.csv` was not found next to app.py, so neighbours can't be shown.")
    else:
        feats = data.drop(columns=["HeartDisease"])
        all_scaled = scaler.transform(encode(feats))
        dist = np.linalg.norm(all_scaled - patient_scaled, axis=1)
        idx = np.argsort(dist)[:k]
        nn = data.iloc[idx].copy()
        nn["Distance"] = dist[idx].round(3)
        nn["Outcome"] = nn["HeartDisease"].map({1: "❤️‍🩹 Heart disease", 0: "✅ Healthy"})
        nn = nn.drop(columns=["HeartDisease"]).reset_index(drop=True)
        nn.index = nn.index + 1
 
        n_dis = int((data.iloc[idx]["HeartDisease"] == 1).sum())
        c1, c2 = st.columns(2)
        c1.metric("Neighbours with heart disease", f"{n_dis} / {k}")
        c2.metric("Neighbours healthy", f"{k - n_dis} / {k}")
        st.progress(n_dis / k, text=f"{n_dis} of {k} neighbours had heart disease")
        st.dataframe(nn, use_container_width=True)
 
        st.caption("Distance is measured in the scaled feature space — smaller means more similar.")
 
# ------------------------------------------------------------------ #
# TAB 4 – Data insights
# ------------------------------------------------------------------ #
with tab_data:
    if data is None:
        st.warning("`heart.csv` was not found next to app.py.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Patients in dataset", f"{len(data):,}")
        d2.metric("Features", data.shape[1] - 1)
        d3.metric("Heart disease rate", f"{data['HeartDisease'].mean() * 100:.0f}%")
 
        st.markdown("#### Feature distribution")
        col_pick = st.selectbox("Choose a feature", NUM_COLS, key="dist_feature")
        plot_df = data.assign(Outcome=data["HeartDisease"].map({1: "Heart disease", 0: "Healthy"}))
        hist = px.histogram(plot_df, x=col_pick, color="Outcome", barmode="overlay", opacity=0.65,
                            nbins=30, color_discrete_map={"Heart disease": "#ff4b4b", "Healthy": "#2ea043"})
        hist.add_vline(x=patient[col_pick], line_width=3, line_dash="dash", line_color="#1f77ff",
                       annotation_text="This patient", annotation_position="top")
        hist.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(hist, use_container_width=True)
 
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Class balance")
            pie = px.pie(plot_df, names="Outcome", hole=0.55, color="Outcome",
                         color_discrete_map={"Heart disease": "#ff4b4b", "Healthy": "#2ea043"})
            pie.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pie, use_container_width=True)
        with g2:
            st.markdown("#### Correlation heatmap")
            corr = data[NUM_COLS + ["FastingBS", "HeartDisease"]].corr()
            heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
            heat.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(heat, use_container_width=True)
 
# ------------------------------------------------------------------ #
# TAB 5 – About
# ------------------------------------------------------------------ #
with tab_about:
    st.markdown("#### Model performance")
    a1, a2, a3 = st.columns(3)
    a1.metric("Accuracy", "88.59%")
    a2.metric("F1 Score", "89.86%")
    a3.metric("Algorithm", f"KNN (k={k})")
 
    st.markdown("#### How it works")
    st.markdown(
        """
        1. **Input** – clinical features are collected from the sidebar.
        2. **Encoding** – categorical values are one-hot encoded and aligned to the training columns.
        3. **Scaling** – features are standardised with the same scaler used in training
           (essential for distance-based models like KNN).
        4. **Prediction** – the model finds the nearest patients and takes a majority vote.
        5. **Explanation** – similar patients, what-if curves and dataset comparisons make the result transparent.
        """
    )
 
    with st.expander("📋 Feature guide"):
        st.markdown(
            """
            - **Chest Pain Type** – ATA: atypical angina, NAP: non-anginal pain, TA: typical angina, ASY: asymptomatic
            - **Resting ECG** – Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy)
            - **Oldpeak** – ST depression induced by exercise relative to rest
            - **ST Slope** – slope of the peak-exercise ST segment (Up / Flat / Down)
            - **Fasting BS** – whether fasting blood sugar is above 120 mg/dL
            """
        )
 
    st.markdown("#### Tech stack")
    st.markdown("`Python` · `scikit-learn` · `pandas` · `NumPy` · `Plotly` · `Streamlit`")
    st.caption("⚠️ This tool is for learning and demonstration only. It is not a substitute for professional medical advice.")
    st.caption("Built by Vijay Saroj")
 
