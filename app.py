"""
MediAssist - Temporary Demo Interface
======================================
A barebones Streamlit application for testing the bridge.py API before the
production frontend is built.

FOR FRONTEND DEVELOPERS
-----------------------
This file is a working, annotated reference implementation showing exactly
how to integrate with bridge.py.  The core integration is one function call:

    from bridge import get_prediction
    result = get_prediction(patient_data)   # patient_data is a plain dict

Your production frontend can be built in any language or framework.  Collect
the same 11 inputs listed below, POST them to a thin backend endpoint (or call
bridge.py directly if staying in Python), and parse the returned dict.

See README.md → "Using bridge.py (Frontend Integration)" for the full input/
output specification including allowed ranges and types.

Usage
-----
    streamlit run app.py
"""

import streamlit as st

from bridge import get_prediction

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediAssist — Demo",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("MediAssist")
st.markdown("**Cardiovascular Risk Assessment — Temporary Demo Interface**")
st.warning(
    "This is a barebones testing interface for the backend API. "
    "It is **not** the production frontend and is **not** for clinical use."
)

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Patient Information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demographics & Measurements**")
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=45, step=1,
        )
        gender = st.selectbox(
            "Gender",
            options=[1, 2],
            format_func=lambda x: "Female (1)" if x == 1 else "Male (2)",
        )
        height = st.number_input(
            "Height (cm)", min_value=100, max_value=220, value=170, step=1,
        )
        weight = st.number_input(
            "Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5,
        )
        ap_hi = st.number_input(
            "Systolic BP (mmHg)", min_value=60, max_value=250, value=120, step=1,
        )
        ap_lo = st.number_input(
            "Diastolic BP (mmHg)", min_value=40, max_value=160, value=80, step=1,
        )

    with col2:
        st.markdown("**Clinical & Lifestyle Indicators**")
        LEVEL_LABELS = {1: "1 — Normal", 2: "2 — Above Normal", 3: "3 — Well Above Normal"}
        cholesterol = st.selectbox(
            "Cholesterol Level",
            options=[1, 2, 3],
            format_func=lambda x: LEVEL_LABELS[x],
        )
        gluc = st.selectbox(
            "Glucose Level",
            options=[1, 2, 3],
            format_func=lambda x: LEVEL_LABELS[x],
        )
        smoke = st.radio(
            "Smoker?",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            horizontal=True,
        )
        alco = st.radio(
            "Alcohol intake?",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            horizontal=True,
        )
        active = st.radio(
            "Physically active?",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            horizontal=True,
        )

    submitted = st.form_submit_button(
        "Assess Cardiovascular Risk",
        use_container_width=True,
        type="primary",
    )

# ---------------------------------------------------------------------------
# Run prediction when form is submitted
# ---------------------------------------------------------------------------
if submitted:

    # -----------------------------------------------------------------
    # CORE INTEGRATION — replicate this block in your frontend
    #
    # Build the 11-key dict with the correct types, then call
    # get_prediction().  All validation and scaling happens inside
    # bridge.py; you do not need to compute BMI or any derived features.
    # -----------------------------------------------------------------
    patient_data = {
        "age":         int(age),
        "gender":      int(gender),
        "height":      int(height),
        "weight":      float(weight),
        "ap_hi":       int(ap_hi),
        "ap_lo":       int(ap_lo),
        "cholesterol": int(cholesterol),
        "gluc":        int(gluc),
        "smoke":       int(smoke),
        "alco":        int(alco),
        "active":      int(active),
    }

    try:
        result = get_prediction(patient_data)
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run `python main.py` first to train the model, "
            "then relaunch this app."
        )
        st.stop()
    except ValueError as exc:
        st.error(f"Input validation error: {exc}")
        st.stop()

    # -----------------------------------------------------------------
    # Parse the response dict
    #
    # result keys:
    #   prediction      (int)   0 = no disease, 1 = disease
    #   probability     (float) raw model probability (0.0 – 1.0)
    #   threshold_used  (float) the F2-optimised threshold that was applied
    #   risk_assessment (dict)  rule-based engine output
    #     .risk_level   (str)   "High Risk" / "Moderate Risk" / "Low Risk"
    #     .risk_factors (list)  human-readable strings for each flag triggered
    #     .flag_count   (int)   number of clinical rules triggered
    # -----------------------------------------------------------------
    prediction     = result["prediction"]
    probability    = result["probability"]
    threshold_used = result["threshold_used"]
    risk_level     = result["risk_assessment"]["risk_level"]
    risk_factors   = result["risk_assessment"]["risk_factors"]

    st.divider()
    st.subheader("Result")

    # --- Primary outcome ---
    if prediction == 1:
        st.error("Cardiovascular disease risk **detected** by the ML model.")
    else:
        st.success("No cardiovascular disease risk detected by the ML model.")

    # --- Metric cards ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Disease Probability", f"{probability:.1%}")
    m2.metric("Rule-Based Risk Level", risk_level)
    m3.metric("Classification Threshold", f"{threshold_used:.4f}")

    # --- Risk factors ---
    if risk_factors:
        st.markdown("**Clinical risk factors flagged by the knowledge engine:**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    else:
        st.info("No clinical risk factors flagged by the rule-based knowledge engine.")

    # --- Derived values shown for transparency ---
    height_m = patient_data["height"] / 100.0
    bmi = round(patient_data["weight"] / (height_m ** 2), 1)
    pulse_pressure = patient_data["ap_hi"] - patient_data["ap_lo"]
    st.caption(
        f"Computed internally by bridge.py — BMI: {bmi} kg/m²  |  "
        f"Pulse Pressure: {pulse_pressure} mmHg  |  "
        f"Age × BMI: {round(patient_data['age'] * bmi, 1)}"
    )

    # --- Raw API response (developer reference) ---
    with st.expander("Raw API response — developer reference"):
        st.json(result)
        st.caption(
            "This is the exact dict returned by `get_prediction()` in bridge.py.  "
            "Your frontend should parse these same top-level keys: "
            "`prediction`, `probability`, `threshold_used`, `risk_assessment`."
        )

    # --- Input echo (developer reference) ---
    with st.expander("Input dict sent to bridge.py — developer reference"):
        st.json(patient_data)
        st.caption(
            "Replicate this exact structure from your frontend. Types and ranges "
            "are enforced by bridge.py — see README.md for the specification."
        )
