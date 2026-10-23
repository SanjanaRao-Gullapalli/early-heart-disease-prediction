# a1.py
"""
Streamlit UI for Cleveland dataset. Loads best_model.pkl and scaler.pkl produced by m1.py.
Run: streamlit run a.py OR streamlit run a1.py
"""

import streamlit as st
import numpy as np
import joblib
import os

# ================= CONFIG =================
MODEL_FILE = "best_logistic_model.pkl"
DECISION_THRESHOLD = 0.40   # recall-priority for medical safety
# =========================================

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center; color: #c0392b;'>❤️ Heart Disease Risk Prediction</h1>
    <p style='text-align: center; font-size: 16px;'>
    Machine Learning–based decision support using the Cleveland Heart Disease Dataset
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------- MODEL LOAD ----------
if not os.path.exists(MODEL_FILE):
    st.error("❌ Trained model not found. Please run the training script first.")
    st.stop()

model = joblib.load(MODEL_FILE)

# ---------- INPUT FORM ----------
st.subheader("🧍 Patient Clinical Inputs")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 18, 120, 54)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ][x]
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 246)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with col2:
        restecg = st.selectbox(
            "Resting ECG Results",
            [0, 1, 2],
            format_func=lambda x: [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ][x]
        )
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            0.0, 6.0, 1.0, step=0.1
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x]
        )
        ca = st.selectbox("Number of Major Vessels Blocked (ca)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia (thal)",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Fixed Defect",
                3: "Reversible Defect"
            }[x]
        )

    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

# ---------- PREDICTION ----------
if submitted:
    sex_v = 1 if sex == "Male" else 0
    fbs_v = 1 if fbs == "Yes" else 0
    exang_v = 1 if exang == "Yes" else 0

    X = np.array([
        age, sex_v, cp, trestbps, chol, fbs_v, restecg,
        thalach, exang_v, oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    # Model handles scaling internally (Pipeline-safe)
    try:
        prob = float(model.predict_proba(X)[0][1])
    except:
        prob = 0.5

    pred = 1 if prob >= DECISION_THRESHOLD else 0

    # ---------- RESULTS ----------
    st.divider()
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error("⚠️ **Heart Disease Risk: PRESENT**")
    else:
        st.success("✅ **Heart Disease Risk: ABSENT**")

    st.metric(
        label="Estimated Risk Likelihood (ML-based)",
        value=f"{prob:.2f}"
    )

    # Risk category
    if prob >= 0.75:
        risk = "High Risk"
    elif prob >= 0.40:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    st.write("**Risk Category:**", risk)
    st.caption("Decision threshold = 0.40 (chosen to reduce false negatives)")

    # ---------- PATIENT SUMMARY ----------
    st.subheader("🧾 Patient Summary")
    st.json({
        "Age": age,
        "Sex": sex,
        "Resting BP": trestbps,
        "Cholesterol": chol,
        "Fasting Blood Sugar >120": fbs,
        "Max Heart Rate": thalach,
        "Exercise Angina": exang
    })

    # ---------- LIFESTYLE SUGGESTIONS ----------
    st.subheader("🍎 Lifestyle & Clinical Suggestions")

    tips = []

    if pred == 1:
        tips.append("Consult a cardiologist for further clinical evaluation.")
    if chol > 240:
        tips.append("Reduce saturated fats and trans fats to manage cholesterol.")
    if trestbps > 140:
        tips.append("Monitor blood pressure and reduce sodium intake.")
    if fbs == "Yes":
        tips.append("Maintain strict blood sugar control with medical guidance.")
    if exang == "Yes" or oldpeak > 1.0:
        tips.append("Avoid unsupervised intense exercise; follow a supervised cardiac plan.")

    tips.extend([
        "Engage in at least 30 minutes of moderate exercise most days.",
        "Adopt a heart-healthy diet rich in fruits, vegetables, and whole grains.",
        "Quit smoking and limit alcohol intake.",
        "Manage stress and ensure adequate sleep."
    ])

    for t in dict.fromkeys(tips):
        st.markdown("• " + t)

    # ---------- FOOTER ----------
    st.divider()
    st.caption(
        "⚠️ This system is intended for academic and decision-support purposes only. "
        "It does not replace professional medical diagnosis or treatment."
    )
    st.caption(f"Model used: {type(model).__name__}")
