"""
MediAssist - Knowledge-Based Diagnosis Engine
Implements rule-based clinical reasoning to flag cardiovascular risk factors
and assign an overall risk level based on established medical guidelines.

Rules are derived from:
  - ESC/ESH 2018 Hypertension Guidelines
  - WHO BMI Classification
  - ACC/AHA 2019 Cardiovascular Risk Guidelines
  - Framingham Heart Study risk factors
"""


def assess_risk(patient: dict) -> dict:
    """
    Evaluate cardiovascular risk using evidence-based clinical heuristics.

    Parameters
    ----------
    patient : dict
        Must contain keys: ap_hi, ap_lo, bmi, cholesterol, gluc, age, smoke,
        alco, active. Values must be in their original (unscaled) form.

    Returns
    -------
    dict
        {
            "risk_level"  : "High Risk" | "Moderate Risk" | "Low Risk",
            "risk_factors": [str, ...],   # human-readable flag descriptions
            "flag_count"  : int           # number of rules triggered
        }
    """
    risk_factors = []

    ap_hi  = patient.get("ap_hi", 0)
    ap_lo  = patient.get("ap_lo", 0)
    bmi    = patient.get("bmi", 0)
    chol   = patient.get("cholesterol", 1)
    gluc   = patient.get("gluc", 1)
    age    = patient.get("age", 0)
    smoke  = patient.get("smoke", 0)
    alco   = patient.get("alco", 0)
    active = patient.get("active", 1)

    # ------------------------------------------------------------------
    # 1. Blood pressure (ESC/ESH 2018)
    #    Stage 2 hypertension ≥160/100 is a stronger flag than Stage 1
    # ------------------------------------------------------------------
    if ap_hi >= 160 or ap_lo >= 100:
        risk_factors.append(
            f"Stage 2 hypertension (systolic: {ap_hi}, diastolic: {ap_lo})"
        )
    elif ap_hi >= 140 or ap_lo >= 90:
        risk_factors.append(
            f"Stage 1 hypertension (systolic: {ap_hi}, diastolic: {ap_lo})"
        )
    elif ap_hi >= 130 or ap_lo >= 80:
        risk_factors.append(
            f"Elevated blood pressure (systolic: {ap_hi}, diastolic: {ap_lo})"
        )

    # Pulse pressure > 60 mmHg — independent predictor of CV events
    pulse_pressure = ap_hi - ap_lo
    if pulse_pressure > 60:
        risk_factors.append(
            f"Wide pulse pressure ({pulse_pressure} mmHg) — arterial stiffness indicator"
        )

    # ------------------------------------------------------------------
    # 2. BMI (WHO classification)
    # ------------------------------------------------------------------
    if bmi >= 35:
        risk_factors.append(f"Severe obesity (BMI: {bmi:.1f} — Class II/III)")
    elif bmi >= 30:
        risk_factors.append(f"Obesity (BMI: {bmi:.1f} — Class I)")
    elif bmi >= 25:
        risk_factors.append(f"Overweight (BMI: {bmi:.1f})")

    # ------------------------------------------------------------------
    # 3. Cholesterol (dataset encoding: 1=normal, 2=above normal, 3=well above)
    # ------------------------------------------------------------------
    if chol == 3:
        risk_factors.append("Cholesterol well above normal — hypercholesterolaemia risk")
    elif chol == 2:
        risk_factors.append("Cholesterol above normal — borderline high")

    # ------------------------------------------------------------------
    # 4. Glucose (dataset encoding: 1=normal, 2=above normal, 3=well above)
    # ------------------------------------------------------------------
    if gluc == 3:
        risk_factors.append("Glucose well above normal — possible diabetes/hyperglycaemia")
    elif gluc == 2:
        risk_factors.append("Glucose above normal — pre-diabetic range")

    # ------------------------------------------------------------------
    # 5. Smoking — independent CV risk factor at any age (Framingham)
    # ------------------------------------------------------------------
    if smoke == 1:
        risk_factors.append("Active smoker — independent cardiovascular risk factor")

    # ------------------------------------------------------------------
    # 6. Alcohol intake
    # ------------------------------------------------------------------
    if alco == 1:
        risk_factors.append("Regular alcohol intake — associated with hypertension and cardiomyopathy")

    # ------------------------------------------------------------------
    # 7. Physical inactivity
    # ------------------------------------------------------------------
    if active == 0:
        risk_factors.append("Physically inactive — sedentary lifestyle increases CV risk")

    # ------------------------------------------------------------------
    # 8. Age-compounded lifestyle risk (Framingham — stronger above 55)
    # ------------------------------------------------------------------
    if age >= 55 and (smoke == 1 or active == 0):
        compounding = []
        if smoke == 1:
            compounding.append("smoking")
        if active == 0:
            compounding.append("sedentary lifestyle")
        risk_factors.append(
            f"Age-compounded risk: age {age} with {' and '.join(compounding)} "
            f"carries significantly higher CV risk"
        )

    # ------------------------------------------------------------------
    # 9. Metabolic syndrome proxy (3+ of: central obesity, raised BP,
    #    raised glucose, raised cholesterol)
    # ------------------------------------------------------------------
    metabolic_flags = sum([
        bmi >= 30,
        ap_hi >= 130 or ap_lo >= 80,
        gluc >= 2,
        chol >= 2,
    ])
    if metabolic_flags >= 3:
        risk_factors.append(
            f"Metabolic syndrome indicators present ({metabolic_flags}/4 criteria met: "
            f"obesity, elevated BP, raised glucose, raised cholesterol)"
        )

    # ------------------------------------------------------------------
    # Risk stratification
    # High:     3+ flags (or any Stage 2 hypertension / severe obesity)
    # Moderate: 1-2 flags
    # Low:      0 flags
    # ------------------------------------------------------------------
    flag_count = len(risk_factors)
    has_severe = (ap_hi >= 160 or ap_lo >= 100 or bmi >= 35)

    if flag_count >= 3 or has_severe:
        risk_level = "High Risk"
    elif flag_count >= 1:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "flag_count": flag_count,
    }
