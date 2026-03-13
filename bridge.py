"""
MediAssist - Bridge Module
Provides the get_prediction() function as the integration surface
for frontend applications. Loads the trained model and scaler from
.pkl files, preprocesses raw patient data, and returns a prediction
along with a knowledge-based risk assessment.
"""

import os
from pathlib import Path

import joblib
import numpy as np

from src.knowledge_engine import assess_risk

_BASE_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _BASE_DIR / "models" / "final_model.pkl"
_SCALER_PATH = _BASE_DIR / "models" / "scaler.pkl"
_FEATURE_ORDER_PATH = _BASE_DIR / "models" / "feature_names.pkl"
_THRESHOLD_PATH = _BASE_DIR / "models" / "threshold.pkl"

# Lazy-loaded singletons
_model = None
_scaler = None
_feature_names = None
_threshold = None

REQUIRED_KEYS = {
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
}

VALUE_BOUNDS = {
    "age": (1, 120),
    "gender": (1, 2),
    "height": (100, 220),
    "weight": (30, 200),
    "ap_hi": (60, 250),
    "ap_lo": (40, 160),
    "cholesterol": (1, 3),
    "gluc": (1, 3),
    "smoke": (0, 1),
    "alco": (0, 1),
    "active": (0, 1),
}


def _load_artifacts():
    """Load model, scaler, feature names, and threshold from disk (once)."""
    global _model, _scaler, _feature_names, _threshold

    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {_MODEL_PATH}")
        _model = joblib.load(_MODEL_PATH)

    if _scaler is None:
        if not _SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found: {_SCALER_PATH}")
        _scaler = joblib.load(_SCALER_PATH)

    if _feature_names is None:
        if not _FEATURE_ORDER_PATH.exists():
            raise FileNotFoundError(f"Feature names file not found: {_FEATURE_ORDER_PATH}")
        _feature_names = joblib.load(_FEATURE_ORDER_PATH)

    if _threshold is None:
        if not _THRESHOLD_PATH.exists():
            raise FileNotFoundError(f"Threshold file not found: {_THRESHOLD_PATH}")
        _threshold = joblib.load(_THRESHOLD_PATH)


def _validate_input(patient_data: dict) -> None:
    """Validate that all required keys are present and values are within sane ranges."""
    missing = REQUIRED_KEYS - set(patient_data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    for key, (lo, hi) in VALUE_BOUNDS.items():
        val = patient_data[key]
        if not isinstance(val, (int, float)):
            raise TypeError(f"Expected numeric value for '{key}', got {type(val).__name__}")
        if val < lo or val > hi:
            raise ValueError(f"Value for '{key}' ({val}) out of expected range [{lo}, {hi}]")


def _preprocess(patient_data: dict) -> np.ndarray:
    """
    Transform raw patient data into a scaled feature vector matching
    the training pipeline's format, including all engineered features.
    """
    # Primary derived features
    height_m = patient_data["height"] / 100.0
    bmi = patient_data["weight"] / (height_m ** 2)
    pulse_pressure = patient_data["ap_hi"] - patient_data["ap_lo"]
    age_bmi = patient_data["age"] * bmi
    bp_hypertension = 1 if (patient_data["ap_hi"] >= 140 or patient_data["ap_lo"] >= 90) else 0
    cholesterol_age = patient_data["cholesterol"] * patient_data["age"]

    # Build feature dict with all engineered features
    features = {
        **patient_data,
        "bmi": round(bmi, 2),
        "pulse_pressure": round(pulse_pressure, 2),
        "age_bmi": round(age_bmi, 2),
        "bp_hypertension": int(bp_hypertension),
        "cholesterol_age": round(cholesterol_age, 2),
    }

    # Order features to match training
    feature_vector = [features[name] for name in _feature_names]
    feature_array = np.array(feature_vector).reshape(1, -1)

    # Scale
    feature_scaled = _scaler.transform(feature_array)
    return feature_scaled


def get_prediction(patient_data: dict) -> dict:
    """
    Generate a cardiovascular disease prediction for a single patient.

    Parameters
    ----------
    patient_data : dict
        Raw patient features. Required keys:
            age (int)      : Patient age in years (1-120)
            gender (int)   : 1 = female, 2 = male
            height (int)   : Height in cm (100-220)
            weight (float) : Weight in kg (30-200)
            ap_hi (int)    : Systolic blood pressure (60-250)
            ap_lo (int)    : Diastolic blood pressure (40-160)
            cholesterol (int) : 1 = normal, 2 = above normal, 3 = well above normal
            gluc (int)     : 1 = normal, 2 = above normal, 3 = well above normal
            smoke (int)    : 0 = no, 1 = yes
            alco (int)     : 0 = no, 1 = yes
            active (int)   : 0 = no, 1 = yes

    Returns
    -------
    dict
        {
            "prediction": int        (0 = no disease, 1 = disease),
            "probability": float     (probability of disease, 0.0-1.0),
            "risk_assessment": dict  (from knowledge engine)
        }
    """
    _validate_input(patient_data)
    _load_artifacts()

    # ML prediction using the tuned classification threshold
    feature_scaled = _preprocess(patient_data)
    y_prob = float(_model.predict_proba(feature_scaled)[0][1])
    prediction = int(y_prob >= _threshold)
    probability = round(y_prob, 4)

    # Knowledge-based risk assessment (uses unscaled values)
    height_m = patient_data["height"] / 100.0
    bmi = patient_data["weight"] / (height_m ** 2)
    risk_input = {**patient_data, "bmi": bmi}
    risk_assessment = assess_risk(risk_input)

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold_used": round(_threshold, 4),
        "risk_assessment": risk_assessment,
    }
