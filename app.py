# ============================================================
# MULTIMODAL HEART DISEASE PREDICTION — STREAMLIT APP
# Agentic AI (LangChain) + Generative AI (Gemini)
# Group: Horizon | MIT Academy of Engineering
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
import hashlib
from PIL import Image
from collections import Counter

from agentic_ai import (
    MODELS, TRAINED_ACC,
    create_heart_disease_agent,
    direct_predict_tabular,
    direct_predict_ecg,
    direct_predict_image,
    direct_predict_symptoms,
)

try:
    import google.generativeai as google_genai
    GENAI_AVAILABLE = True
except:
    GENAI_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Prediction AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff9f43);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .sub-title {
        text-align: center;
        color: #aaaaaa;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #333355;
        margin: 4px;
    }
    .agent-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #4a90d944;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .gemini-box {
        background: linear-gradient(135deg, #1e1e2e, #2d1b4e);
        border: 1px solid #9b59b644;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        line-height: 1.8;
    }
    .result-positive {
        background: linear-gradient(135deg, #2d1b1b, #4a1a1a);
        border: 1px solid #ff4b4b66;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #1b2d1b, #1a4a1a);
        border: 1px solid #26de8166;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .risk-high   { background:linear-gradient(90deg,#ff4b4b,#c0392b);
                   color:white;padding:6px 18px;border-radius:20px;
                   font-weight:bold;display:inline-block; }
    .risk-medium { background:linear-gradient(90deg,#ff9f43,#e67e22);
                   color:white;padding:6px 18px;border-radius:20px;
                   font-weight:bold;display:inline-block; }
    .risk-low    { background:linear-gradient(90deg,#26de81,#20bf6b);
                   color:white;padding:6px 18px;border-radius:20px;
                   font-weight:bold;display:inline-block; }
    .section-divider {
        border:none;height:2px;
        background:linear-gradient(90deg,transparent,#ff4b4b,transparent);
        margin:20px 0;
    }
    .model-row {
        background:#111122;border-radius:8px;
        padding:10px 16px;margin:4px 0;
        border-left:3px solid #4a90d9;
    }
    .best-model-row {
        background:#112211;border-radius:8px;
        padding:10px 16px;margin:4px 0;
        border-left:3px solid #26de81;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #ff9f43);
        color: white;border: none;border-radius: 10px;
        padding: 12px 30px;font-size: 1.05rem;
        font-weight: bold;width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# API KEY
# ============================================================
DEFAULT_GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or "AIzaSyAGEPX3wfCwwZybaMqkUw-PiAIskdc4VsI"
)
GEMINI_API_KEY = st.sidebar.text_input(
    "Gemini API Key",
    value=DEFAULT_GEMINI_API_KEY,
    type="password",
    help="Paste your new Google AI Studio key here. This also refreshes the cached LangChain client.",
)
GEMINI_MODEL = "gemini-2.5-flash"

gemini_key_hash = hashlib.sha256(GEMINI_API_KEY.encode()).hexdigest()
if st.session_state.get("gemini_key_hash") != gemini_key_hash:
    st.session_state["gemini_key_hash"] = gemini_key_hash
    st.session_state.pop("gemini_quota_exhausted", None)
    st.session_state.pop("gemini_quota_error", None)
    st.session_state.pop("last_langchain_error", None)


def is_gemini_quota_error(error):
    """Return True for Gemini quota/rate-limit exceptions from LangChain or SDK."""
    error_text = str(error).lower()
    quota_markers = (
        "429",
        "resource_exhausted",
        "resource exhausted",
        "quota exceeded",
        "rate limit",
        "ratelimit",
        "too many requests",
        "retryinfo",
    )
    return any(marker in error_text for marker in quota_markers)


def is_gemini_auth_error(error):
    error_text = str(error).lower()
    auth_markers = (
        "api key not valid",
        "invalid api key",
        "permission_denied",
        "permission denied",
        "unauthenticated",
        "403",
        "401",
    )
    return any(marker in error_text for marker in auth_markers)


def is_gemini_network_error(error):
    error_text = str(error).lower()
    network_markers = (
        "dns",
        "name resolution",
        "failed to resolve",
        "connection error",
        "connection reset",
        "timeout",
        "deadline exceeded",
        "503",
        "unavailable",
    )
    return any(marker in error_text for marker in network_markers)


def mark_gemini_quota_exhausted(error=None):
    st.session_state["gemini_quota_exhausted"] = True
    if error is not None:
        st.session_state["gemini_quota_error"] = str(error)


def record_langchain_fallback(error):
    st.session_state["last_langchain_error"] = str(error)
    if is_gemini_quota_error(error):
        mark_gemini_quota_exhausted(error)


def can_use_gemini():
    return gemini_client is not None

# ============================================================
# SETUP GEMINI
# ============================================================
@st.cache_resource
def setup_gemini(api_key):
    if not GENAI_AVAILABLE or not api_key:
        return None
    try:
        google_genai.configure(api_key=api_key)
        model = google_genai.GenerativeModel(GEMINI_MODEL)
        return model
    except:
        return None

@st.cache_resource
def setup_langchain_agent(api_key):
    if not api_key:
        return None
    try:
        agent = create_heart_disease_agent(api_key)
        return agent
    except Exception as e:
        return None

gemini_client  = setup_gemini(GEMINI_API_KEY)
langchain_agent = setup_langchain_agent(GEMINI_API_KEY)

# ============================================================
# GEMINI EXPLANATION
# ============================================================
def get_gemini_explanation(input_type, prediction,
                            risk_level, confidence,
                            patient_info, model_results=None):
    model_info = ""
    if model_results:
        model_info = "\nModel Results:\n"
        for name, res in model_results.items():
            if isinstance(res, dict):
                result_val = res.get('result', res.get('class', ''))
                conf_val   = res.get('confidence', 0) * 100
                trained    = res.get('trained_accuracy',
                              res.get('trained_acc', 0))
                model_info += (f"  {name}: {result_val} "
                               f"({conf_val:.1f}% conf, "
                               f"{trained}% trained)\n")

    prompt = f"""
You are a Medical AI assistant. A patient was analyzed by
Multimodal Heart Disease Prediction System.

RESULTS:
- Input Type : {input_type}
- Prediction : {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}
- Risk Level : {risk_level}
- Confidence : {confidence*100:.1f}%
- Patient    : {patient_info}
{model_info}

Provide:
1. 🔍 EXPLANATION: Plain English (2-3 sentences)
2. ⚠️ KEY RISK FACTORS: 3 factors from the data
3. 💊 RECOMMENDATIONS: 3 health recommendations
4. 🏥 NEXT STEPS: What to do next

Keep it clear and non-alarming.
Always recommend consulting a real doctor.
"""
    if can_use_gemini():
        for attempt in range(3):
            try:
                response = gemini_client.generate_content(prompt)
                return response.text
            except Exception as e:
                if is_gemini_quota_error(e):
                    break
                break

    # Fallback
    return _fallback_explanation(prediction, risk_level, confidence)


def _fallback_explanation(prediction, risk_level, confidence):
    if prediction == 1:
        return f"""
🔍 **EXPLANATION:**
Heart disease indicators detected with {confidence*100:.1f}% confidence.
Risk assessed as **{risk_level}** based on multimodal AI analysis.

⚠️ **KEY RISK FACTORS:**
- Elevated cardiovascular indicators detected
- {risk_level} classification from K-Means clustering
- Multiple risk parameters above normal range

💊 **RECOMMENDATIONS:**
- Schedule cardiologist appointment immediately
- Monitor blood pressure and cholesterol regularly
- Adopt heart-healthy diet and regular exercise

🏥 **NEXT STEPS:**
Consult a qualified healthcare professional urgently.
This AI prediction should be confirmed by a medical doctor.
"""
    else:
        return f"""
🔍 **EXPLANATION:**
No strong heart disease indicators found ({confidence*100:.1f}% confidence).
Risk assessed as **{risk_level}**.

⚠️ **PREVENTIVE FACTORS:**
- Maintain regular annual health checkups
- Monitor cholesterol and blood pressure
- Stay aware of any new symptoms

💊 **RECOMMENDATIONS:**
- Balanced diet rich in vegetables and fruits
- 30 minutes of exercise daily
- Avoid smoking and limit alcohol

🏥 **NEXT STEPS:**
Continue regular health monitoring.
Consult a doctor if new symptoms appear.
"""


def get_local_recommendations(prediction, risk_level, confidence, input_type):
    confidence_text = f"{confidence * 100:.1f}%"
    if prediction == 1:
        return f"""
**Summary**
- Prediction: Heart disease indicators detected
- Risk level: {risk_level}
- Confidence: {confidence_text}
- Analysis source: {input_type.title()} specialist model route

**Key Risk Factors**
- The selected model route found cardiovascular risk signals in the submitted data.
- The case was categorized as {risk_level}, so it should be reviewed carefully.
- Model confidence is {confidence_text}; use this as a screening signal, not a diagnosis.

**Recommendations**
- Book a consultation with a qualified doctor or cardiologist.
- Monitor blood pressure, cholesterol, blood sugar, and symptoms regularly.
- Follow a heart-healthy routine: balanced diet, regular activity, sleep, and no smoking.

**Next Steps**
- Share these results with a healthcare professional.
- Seek urgent medical help if there is chest pain, severe breathlessness, fainting, or sweating.
- Repeat testing with clinical evaluation if symptoms continue.
"""

    return f"""
**Summary**
- Prediction: No strong heart disease indicators found
- Risk level: {risk_level}
- Confidence: {confidence_text}
- Analysis source: {input_type.title()} specialist model route

**Preventive Factors**
- Current submitted data did not show strong disease signals.
- Continue watching for new symptoms or changes in health.
- Keep routine checkups, especially for blood pressure, cholesterol, and sugar.

**Recommendations**
- Maintain regular exercise and a balanced diet.
- Avoid smoking and limit alcohol.
- Continue annual health screening or earlier review if symptoms appear.

**Next Steps**
- Treat this as an educational screening result.
- Consult a doctor if you have chest pain, shortness of breath, dizziness, or fatigue.
- Keep tracking lifestyle and clinical parameters over time.
"""


def get_local_follow_up_answer(question, prediction, risk_level, confidence):
    result_text = "heart disease indicators were detected" if prediction == 1 else "no strong heart disease indicators were found"
    question_text = question.lower()

    if any(word in question_text for word in ["food", "eat", "diet", "meal", "avoid"]):
        return f"""
Based on your result, {result_text} with {confidence * 100:.1f}% confidence and the risk level is {risk_level}.
For heart health, choose vegetables, fruits, whole grains, beans, lentils, nuts, seeds, low-fat dairy, and lean proteins such as fish or skinless chicken.
Use healthy fats like olive oil, and limit fried foods, processed snacks, sugary drinks, excess salt, red meat, butter, and packaged foods high in trans fat.
For Indian meals, good options include dal, sprouts, vegetable sabzi with less oil, oats, brown rice, roti, curd, salad, and grilled or steamed foods.
Please confirm diet changes with a doctor or dietitian, especially if you have diabetes, kidney disease, high BP, or cholesterol issues.
"""

    if any(word in question_text for word in ["exercise", "walk", "workout", "activity"]):
        return f"""
Based on your result, {result_text} with {confidence * 100:.1f}% confidence and the risk level is {risk_level}.
A safe general target is regular walking or light-to-moderate activity, building toward about 30 minutes most days if your doctor says it is safe.
Avoid sudden heavy exercise if you have chest pain, breathlessness, dizziness, or fainting.
Consult a doctor before starting a new exercise plan, especially with {risk_level}.
"""

    if any(word in question_text for word in ["medicine", "tablet", "drug", "aspirin", "statin"]):
        return f"""
Based on your result, {result_text} with {confidence * 100:.1f}% confidence and the risk level is {risk_level}.
Do not start or stop heart medicines based only on this app.
A doctor may consider tests like BP, ECG, lipid profile, sugar testing, or further cardiac evaluation before prescribing treatment.
If you already take prescribed medicine, continue it as directed and ask your doctor before making changes.
"""

    return f"""
Based on the current result, {result_text} with {confidence * 100:.1f}% confidence and the risk level is {risk_level}.
For your question, "{question}", the safest guidance is to use this AI result as a screening aid and discuss it with a qualified doctor.
Focus on heart-healthy habits such as regular exercise, balanced diet, blood pressure and cholesterol monitoring, good sleep, and avoiding smoking.
If symptoms like chest pain, shortness of breath, fainting, heavy sweating, or severe fatigue appear, seek urgent medical care.
"""

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ❤️ Heart Disease AI")
    st.markdown("---")

    st.markdown("### 🤖 System Status")
    total = len(MODELS)
    if total >= 8:
        st.success(f"✅ {total} models loaded")
    else:
        st.warning(f"⚠️ {total} models loaded")

    if gemini_client:
        st.success("✅ Gemini AI Connected")
    else:
        st.warning("⚠️ Gemini unavailable")

    if langchain_agent:
        st.success("✅ LangChain Agent Ready")
    else:
        st.warning("⚠️ LangChain unavailable")

    st.markdown("---")
    st.markdown("### 📊 Models Loaded")
    model_display = [
        ("Logistic Regression", "lr"),
        ("Random Forest",       "rf"),
        ("SVM",                 "svm"),
        ("KNN",                 "knn"),
        ("Decision Tree",       "dt"),
        ("ANN",                 "ann"),
        ("CNN 1D",              "cnn1d"),
        ("LSTM",                "lstm"),
        ("RNN/GRU",             "rnn"),
        ("ResNet50",            "resnet"),
        ("K-Means",             "kmeans"),
        ("NLP",                 "nlp"),
    ]
    for name, key in model_display:
        icon = "✅" if key in MODELS else "❌"
        st.markdown(f"{icon} {name}")

    st.markdown("---")
    st.markdown("### 🧠 Agentic AI Routing")
    st.markdown("""
    - 📋 Clinical → **6 Models**
    - 📈 ECG → **CNN1D+LSTM+RNN**
    - 🖼️ X-ray → **ResNet50**
    - 💬 Symptoms → **NLP**
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Agent Mode")
    use_langchain = st.toggle(
        "Use LangChain Agent",
        value=False,
        help="ON=Full LangChain reasoning, OFF=Fast direct prediction"
    )
    if use_langchain:
        st.info("🐢 Slower but shows full reasoning")
    else:
        st.info("⚡ Fast direct prediction mode")

    st.markdown("---")
    st.markdown("### 👥 Group: Horizon")
    st.markdown("MIT Academy of Engineering")
    st.markdown("Predictive Analytics | Sem VI")

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown(
    '<p class="main-title">❤️ Multimodal Heart Disease Prediction</p>',
    unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">LangChain Agentic AI + Google Gemini | '
    'Group Horizon | MIT Academy of Engineering</p>',
    unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── How It Works ──
with st.expander("🤖 How the System Works", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""**1️⃣ Input**
- Clinical data
- ECG signal
- X-ray image
- Symptoms text""")
    with c2:
        st.markdown("""**2️⃣ Agent Decides**
- LangChain ReAct
- Detects input type
- Selects tools
- Reasons step-by-step""")
    with c3:
        st.markdown("""**3️⃣ Multi-Model**
- Runs ALL models
- Compares results
- Majority voting
- Best model picked""")
    with c4:
        st.markdown("""**4️⃣ Gemini Explains**
- Plain English
- Risk factors
- Recommendations
- Q&A chat""")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("## 📥 Patient Input")
st.info("💡 Select input type. Agentic AI automatically routes to best models.")

input_mode = st.radio(
    "🔀 Select Input Type:",
    ["📋 Clinical Data (6 Models)",
     "📈 ECG Signal (CNN1D + LSTM + RNN)",
     "🖼️ Chest X-ray (ResNet50)",
     "💬 Symptoms Text (NLP)"],
    horizontal=True
)

st.markdown("---")

tabular_input = None
ecg_input     = None
image_input   = None
symptom_input = ""
image_path    = None
patient_info  = {}

# ── CLINICAL DATA ──
if input_mode == "📋 Clinical Data (6 Models)":
    st.markdown("### 📋 Patient Clinical Data")
    st.caption("*Runs: Logistic Regression, Random Forest, "
               "SVM, KNN, Decision Tree, ANN*")

    c1, c2, c3 = st.columns(3)
    with c1:
        age         = st.slider("Age", 20, 90, 50)
        resting_bp  = st.slider("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs  = st.selectbox("Fasting Blood Sugar > 120?",
                                   [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
    with c2:
        max_hr          = st.slider("Max Heart Rate", 60, 220, 150)
        oldpeak         = st.slider("Oldpeak", 0.0, 6.0, 1.0, 0.1)
        sex             = st.selectbox("Sex", ["Male", "Female"])
        exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
    with c3:
        chest_pain  = st.selectbox("Chest Pain Type",
                                   ["ATA", "NAP", "ASY", "TA"])
        resting_ecg = st.selectbox("Resting ECG",
                                   ["Normal", "ST", "LVH"])
        st_slope    = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    tabular_input = {
        'Age': age, 'RestingBP': resting_bp,
        'Cholesterol': cholesterol, 'FastingBS': fasting_bs,
        'MaxHR': max_hr, 'Oldpeak': oldpeak, 'Sex': sex,
        'ChestPainType': chest_pain, 'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina, 'ST_Slope': st_slope,
    }
    patient_info = {
        'Age': age, 'BP': resting_bp,
        'Cholesterol': cholesterol, 'MaxHR': max_hr,
        'Sex': sex, 'ChestPain': chest_pain
    }

# ── ECG SIGNAL ──
elif input_mode == "📈 ECG Signal (CNN1D + LSTM + RNN)":
    st.markdown("### 📈 ECG Signal Data")
    st.caption("*Runs: CNN1D (98.62%), LSTM (81.11%), RNN/GRU (87.39%)*")

    ecg_file   = st.file_uploader("Upload ECG CSV (187 values)",
                                   type=["csv"])
    use_sample = st.checkbox("✅ Use sample ECG signal for demo")

    if ecg_file is not None:
        ecg_df    = pd.read_csv(ecg_file, header=None)
        ecg_input = ecg_df.iloc[0, :187].values.astype(float)
        st.success(f"✅ ECG loaded! Shape: {ecg_input.shape}")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(ecg_input, color='tomato', linewidth=1.2)
        ax.set_title('Uploaded ECG Signal', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif use_sample:
        t         = np.linspace(0, 4*np.pi, 187)
        ecg_input = (np.sin(t) + 0.3*np.sin(3*t) +
                     0.1*np.random.randn(187)).astype(float)
        st.success("✅ Sample ECG loaded!")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(ecg_input, color='steelblue', linewidth=1.2)
        ax.set_title('Sample ECG Signal (Demo)', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    patient_info['input_type'] = 'ECG Signal'

# ── CHEST X-RAY ──
elif input_mode == "🖼️ Chest X-ray (ResNet50)":
    st.markdown("### 🖼️ Chest X-ray Image")
    st.caption("*Runs: ResNet50 Transfer Learning (87.50%)*")

    uploaded = st.file_uploader("Upload Chest X-ray (JPG/PNG)",
                                 type=["jpg", "jpeg", "png"])
    if uploaded:
        image_input = Image.open(uploaded)
        # Save temp file for LangChain agent
        temp_path   = r"C:\Users\ABHINAV\Desktop\Heart_PA\temp_xray.jpg"
        image_input.save(temp_path)
        image_path  = temp_path

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(image_input, caption="Uploaded X-ray",
                     use_column_width=True)
        with c2:
            st.success("✅ X-ray uploaded!")
            st.info(f"📐 Size: {image_input.size}")
            st.info("🤖 ResNet50 will analyze this image")
        patient_info['input_type'] = 'Chest X-ray'

# ── SYMPTOMS TEXT ──
elif input_mode == "💬 Symptoms Text (NLP)":
    st.markdown("### 💬 Symptom Description")
    st.caption("*Runs: NLP (TF-IDF + Logistic Regression)*")

    symptom_input = st.text_area(
        "Describe symptoms in plain English:",
        placeholder="Example: chest pain, shortness of breath, "
                    "fatigue, dizziness, sweating...",
        height=150
    )
    if symptom_input:
        st.success(f"✅ {len(symptom_input.split())} words detected")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**High Risk:**\n• Chest pain\n• Shortness of breath\n• Sweating")
    with c2:
        st.markdown("**Medium Risk:**\n• Mild fatigue\n• Occasional dizziness\n• Palpitations")
    with c3:
        st.markdown("**Low Risk:**\n• No symptoms\n• Normal checkup\n• Feeling fine")
    patient_info['Symptoms'] = symptom_input

# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
_, cbtn, _ = st.columns([1, 2, 1])
if 'results_visible' not in st.session_state:
    st.session_state.results_visible = False

with cbtn:
    if st.button("🔍 Analyze & Predict", key="main_predict"):
        st.session_state.results_visible = True
        st.session_state.last_follow_up_question = ""
        st.session_state.last_follow_up_answer = ""
        st.session_state.last_follow_up_source = ""

predict_clicked = st.session_state.results_visible

# ============================================================
# RESULTS
# ============================================================
if predict_clicked:

    # Validate input
    has_input = (
        (tabular_input is not None and input_mode.startswith("📋")) or
        (ecg_input is not None and input_mode.startswith("📈")) or
        (image_input is not None and input_mode.startswith("🖼️")) or
        (symptom_input.strip() and input_mode.startswith("💬"))
    )

    if not has_input:
        st.error("⚠️ Please provide input before predicting!")
        st.stop()

    st.markdown("---")
    st.markdown("## 🤖 Agentic AI — Processing...")

    # ── Determine input type ──
    if input_mode.startswith("📋"):
        input_type = "tabular"
        model_used = "LR + RF + SVM + KNN + DT + ANN"
    elif input_mode.startswith("📈"):
        input_type = "ecg"
        model_used = "CNN1D + LSTM + RNN/GRU"
    elif input_mode.startswith("🖼️"):
        input_type = "image"
        model_used = "ResNet50 (Transfer Learning)"
    else:
        input_type = "text"
        model_used = "NLP (TF-IDF + Logistic Regression)"

    patient_info['model_used'] = model_used

    # ── Show Agent Decision ──
    st.markdown(f"""
    <div class="agent-box">
        <h4>🤖 LangChain Agent Decision</h4>
        <p>📥 <b>Input Detected :</b> {input_type.upper()}</p>
        <p>🧠 <b>Models Selected:</b> {model_used}</p>
        <p>🔧 <b>Framework     :</b> LangChain ReAct Agent + Google Gemini</p>
        <p>⚡ <b>Status        :</b> Routing to specialist models...</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Run Prediction ──
    prediction     = None
    confidence     = None
    risk_level     = None
    model_results  = {}
    agent_response = None

    if use_langchain and langchain_agent:
        # ── LangChain Full Agent ──
        with st.spinner("🧠 LangChain Agent reasoning..."):
            try:
                if input_type == 'tabular':
                    ti = tabular_input
                    query = (
                        f"Analyze patient: "
                        f"{ti['Age']},{ti['RestingBP']},{ti['Cholesterol']},"
                        f"{ti['FastingBS']},{ti['MaxHR']},{ti['Oldpeak']},"
                        f"{ti['Sex']},{ti['ChestPainType']},{ti['RestingECG']},"
                        f"{ti['ExerciseAngina']},{ti['ST_Slope']}. "
                        f"Compare all 6 models and recommend best."
                    )
                elif input_type == 'ecg':
                    query = "Analyze ECG signal from sample. Compare CNN1D LSTM RNN."
                elif input_type == 'image':
                    query = f"Analyze chest X-ray at: {image_path}"
                else:
                    query = (f"Patient symptoms: {symptom_input}. "
                             f"Analyze and predict risk level.")

                result         = langchain_agent.invoke({"input": query})
                agent_response = result.get('output', '')

            except Exception as e:
                record_langchain_fallback(e)
                use_langchain = False

    # ── Direct Prediction ──
    if not use_langchain or prediction is None:
        with st.spinner(f"Running {model_used}..."):
            time.sleep(0.5)

            if input_type == 'tabular':
                prediction, confidence, risk_level, model_results = \
                    direct_predict_tabular(tabular_input)

            elif input_type == 'ecg':
                prediction, confidence, risk_level, model_results = \
                    direct_predict_ecg(ecg_input)

            elif input_type == 'image':
                prediction, confidence, risk_level = \
                    direct_predict_image(image_input)

            elif input_type == 'text':
                prediction, confidence, risk_level = \
                    direct_predict_symptoms(symptom_input)

    # Fix confidence for LangChain mode
    if prediction is None:
        prediction, confidence, risk_level = 0, 0.5, 'Medium Risk'
    if confidence is None:
        confidence = 0.5

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    result_text  = "❤️ Heart Disease Detected" if prediction == 1 else "💚 No Heart Disease"
    result_color = "#ff4b4b" if prediction == 1 else "#26de81"
    risk_class   = {'High Risk'  :'risk-high',
                    'Medium Risk':'risk-medium',
                    'Low Risk'   :'risk-low'}.get(risk_level, 'risk-medium')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <h4 style="color:{result_color}">{result_text}</h4>
            <p style="color:#aaa">Final Prediction</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <span class="{risk_class}">{risk_level}</span>
            <p style="color:#aaa;margin-top:8px">Risk Level</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <h4 style="color:#ff9f43">{confidence*100:.1f}%</h4>
            <p style="color:#aaa">Confidence</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-box">
            <h4 style="color:#4a90d9;font-size:0.8rem">{model_used}</h4>
            <p style="color:#aaa">Models Used</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Confidence Score")
    st.progress(float(confidence))
    st.markdown(f"Confidence: **{confidence*100:.1f}%**")

    # ── Multi-Model Comparison (Tabular) ──
    if input_type == 'tabular' and model_results:
        st.markdown("---")
        st.markdown("## 📊 All 6 Models Comparison")
        st.caption("*Majority voting determines final prediction*")

        # Table
        rows = []
        for name, res in model_results.items():
            rows.append({
                'Model'           : name,
                'Prediction'      : res['result'],
                'Confidence'      : f"{res['confidence']*100:.1f}%",
                'Trained Accuracy': f"{res['trained_accuracy']}%",
                'Best?'           : '⭐' if name == max(
                    model_results.items(),
                    key=lambda x: x[1]['trained_accuracy'])[0] else ''
            })

        df_results = pd.DataFrame(rows)
        st.dataframe(df_results, use_container_width=True,
                     hide_index=True)

        # Bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        names  = list(model_results.keys())
        accs   = [r['trained_accuracy'] for r in model_results.values()]
        confs  = [r['confidence']*100  for r in model_results.values()]
        colors = ['#26de81' if r['prediction'] == 0 else '#ff4b4b'
                  for r in model_results.values()]

        axes[0].barh(names, accs,
                     color='steelblue', edgecolor='white')
        axes[0].set_title('Trained Accuracy per Model',
                           fontweight='bold')
        axes[0].set_xlabel('Accuracy (%)')
        axes[0].set_xlim(70, 100)
        for i, v in enumerate(accs):
            axes[0].text(v + 0.2, i, f'{v}%',
                         va='center', fontweight='bold')

        axes[1].barh(names, confs,
                     color=colors, edgecolor='white')
        axes[1].set_title('Current Confidence per Model\n'
                           '(Green=No Disease, Red=Disease)',
                           fontweight='bold')
        axes[1].set_xlabel('Confidence (%)')
        for i, v in enumerate(confs):
            axes[1].text(v + 0.2, i, f'{v:.1f}%',
                         va='center', fontweight='bold')

        plt.suptitle('All 6 Models — Comparison',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Voting summary
        votes     = [r['prediction'] for r in model_results.values()]
        disease   = votes.count(1)
        no_disease = votes.count(0)
        st.markdown(f"""
        **🗳️ Majority Voting:**
        - Heart Disease: **{disease}/{len(votes)}** models
        - No Disease: **{no_disease}/{len(votes)}** models
        - Final Decision: **{'Heart Disease' if disease > no_disease else 'No Disease'}**
        """)

    # ── ECG Model Comparison ──
    elif input_type == 'ecg' and model_results:
        st.markdown("---")
        st.markdown("## 📊 ECG Models Comparison")
        st.caption("*CNN1D + LSTM + RNN/GRU — Majority voting*")

        cols = st.columns(len(model_results))
        for col, (mname, res) in zip(cols, model_results.items()):
            color = "#ff4b4b" if res['class'] != 'Normal' else "#26de81"
            with col:
                st.markdown(f"""<div class="metric-box">
                    <h4 style="color:#4a90d9">{mname}</h4>
                    <h3 style="color:{color}">{res['class']}</h3>
                    <p style="color:#ff9f43">
                        {res['confidence']*100:.1f}% confidence
                    </p>
                    <p style="color:#aaa">
                        Trained: {res['trained_acc']}%
                    </p>
                </div>""", unsafe_allow_html=True)

        # ECG comparison chart
        fig, ax = plt.subplots(figsize=(8, 4))
        mnames  = list(model_results.keys())
        confs   = [r['confidence']*100 for r in model_results.values()]
        accs    = [r['trained_acc'] for r in model_results.values()]
        x       = np.arange(len(mnames))
        bars1   = ax.bar(x - 0.2, accs, 0.4,
                         label='Trained Accuracy', color='steelblue')
        bars2   = ax.bar(x + 0.2, confs, 0.4,
                         label='Current Confidence', color='tomato')
        ax.set_title('ECG Models — Accuracy vs Confidence',
                     fontweight='bold')
        ax.set_ylabel('%')
        ax.set_xticks(x)
        ax.set_xticklabels(mnames)
        ax.legend()
        ax.set_ylim(0, 115)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{bar.get_height():.1f}%',
                    ha='center', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{bar.get_height():.1f}%',
                    ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── LangChain Agent Response ──
    if agent_response:
        st.markdown("---")
        st.markdown("## 🤖 LangChain Agent Full Response")
        st.markdown(f"""<div class="agent-box">
            <pre style="white-space:pre-wrap;color:#e0e0e0;
                        font-size:0.85rem">{agent_response}</pre>
        </div>""", unsafe_allow_html=True)

    # ── Recommendations and Next Steps ──
    st.markdown("---")
    st.markdown("## 🩺 Recommendations & Next Steps")
    local_recommendations = get_local_recommendations(
        prediction, risk_level, confidence, input_type
    )
    st.markdown(f"""<div class="gemini-box">
        {local_recommendations.replace(chr(10), '<br>')}
    </div>""", unsafe_allow_html=True)

    # ── Gemini Explanation ──
    st.markdown("---")
    st.markdown("## 🧠 Generative AI Explanation (Gemini)")

    with st.spinner("Gemini generating explanation..."):
        explanation = get_gemini_explanation(
            input_type, prediction, risk_level,
            confidence, patient_info, model_results
        )

    st.markdown(f"""<div class="gemini-box">
        {explanation.replace(chr(10), '<br>')}
    </div>""", unsafe_allow_html=True)

    # ── Follow-up Chat ──
    st.markdown("---")
    st.markdown("## 💬 Ask Gemini Follow-up Questions")

    if 'follow_up' not in st.session_state:
        st.session_state.follow_up = ""
    if 'last_follow_up_question' not in st.session_state:
        st.session_state.last_follow_up_question = ""
    if 'last_follow_up_answer' not in st.session_state:
        st.session_state.last_follow_up_answer = ""
    if 'last_follow_up_source' not in st.session_state:
        st.session_state.last_follow_up_source = ""
    if st.session_state.pop("clear_follow_up_input", False):
        st.session_state.follow_up = ""

    follow_up = st.text_input(
        "Ask anything about your results:",
        key="follow_up",
        placeholder="What foods should I avoid? "
                    "How can I improve my heart health?"
    )

    if st.button("📨 Ask Gemini", key="ask_gemini"):
        current_question = st.session_state.follow_up.strip()
        if current_question and can_use_gemini():
            with st.spinner("Gemini thinking..."):
                ctx_prompt = f"""
Patient: {patient_info}
Prediction: {'Heart Disease' if prediction == 1 else 'No Disease'}
Risk: {risk_level}
Question: {current_question}
Answer in 3-4 sentences. Always recommend consulting a doctor.
"""
                for attempt in range(3):
                    try:
                        resp = gemini_client.generate_content(ctx_prompt)
                        st.session_state.last_follow_up_question = current_question
                        st.session_state.last_follow_up_answer = resp.text
                        st.session_state.last_follow_up_source = "Gemini"
                        st.session_state.clear_follow_up_input = True
                        st.rerun()
                        break
                    except Exception as e:
                        st.session_state.last_follow_up_question = current_question
                        st.session_state.last_follow_up_answer = get_local_follow_up_answer(
                            current_question, prediction, risk_level, confidence
                        )
                        st.session_state.last_follow_up_source = "AI Assistant"
                        st.session_state.clear_follow_up_input = True
                        st.rerun()
                        break
        else:
            st.warning("Type a question and ensure Gemini is connected.")

    if st.session_state.last_follow_up_answer:
        answer_source = st.session_state.last_follow_up_source or "AI Assistant"
        st.markdown(f"""<div class="gemini-box">
            <b>You:</b> {st.session_state.last_follow_up_question}<br><br>
            <b>🤖 {answer_source}:</b><br><br>
            {st.session_state.last_follow_up_answer.replace(chr(10), '<br>')}
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("---")
    st.warning("""
    ⚠️ **Medical Disclaimer:** This system is for educational purposes only
    as part of an academic project at MIT Academy of Engineering.
    It is NOT a substitute for professional medical advice.
    Always consult a qualified healthcare professional.
    """)
