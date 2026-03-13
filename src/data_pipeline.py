"""
MediAssist - Data Pipeline Module
Handles loading, cleaning, feature engineering, and normalization
of the Cardiovascular Disease dataset.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Biological plausibility bounds
OUTLIER_BOUNDS = {
    "height": (100, 220),
    "weight": (30, 200),
    "ap_hi": (60, 250),
    "ap_lo": (40, 160),
}


def load_data(path: str) -> pd.DataFrame:
    """Load the Cardiovascular Disease CSV and drop the surrogate id column."""
    df = pd.read_csv(path, sep=";")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df


def convert_age(df: pd.DataFrame) -> pd.DataFrame:
    """Convert age from days to whole years."""
    df = df.copy()
    df["age"] = (df["age"] / 365.25).astype(int)
    logger.info("Converted age from days to years (range: %d-%d)", df["age"].min(), df["age"].max())
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove biologically implausible records based on clinical thresholds."""
    initial_count = len(df)
    df = df.copy()

    for col, (lo, hi) in OUTLIER_BOUNDS.items():
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    # Systolic pressure must exceed diastolic
    df = df[df["ap_hi"] > df["ap_lo"]]

    removed = initial_count - len(df)
    logger.info("Removed %d outlier rows (%.1f%%), %d remaining", removed, 100 * removed / initial_count, len(df))
    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing values."""
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing values", dropped)
    return df


def calculate_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Add a BMI column: weight (kg) / (height (m))^2."""
    df = df.copy()
    height_m = df["height"] / 100.0
    df["bmi"] = np.round(df["weight"] / (height_m ** 2), 2)
    logger.info("Calculated BMI (mean: %.1f, std: %.1f)", df["bmi"].mean(), df["bmi"].std())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed derived features to improve predictive power.

    Features added:
    - pulse_pressure  : Systolic - Diastolic BP (ap_hi - ap_lo).
                        A pulse pressure > 60 mmHg is an independent predictor of
                        cardiovascular events, reflecting arterial stiffness not
                        captured by either BP value alone.
    - age_bmi         : Interaction term (age * bmi).
                        Captures the compounding metabolic-aging risk that neither
                        variable encodes alone — obesity in older patients carries
                        disproportionately higher cardiovascular risk.
    - bp_hypertension : Binary flag (1 if systolic ≥ 140 OR diastolic ≥ 90).
                        Stage 2 hypertension by ACC/AHA 2017 guidelines is one of
                        the strongest modifiable predictors of cardiovascular disease.
                        Encoding it as a binary feature sharpens the decision boundary
                        and raises the precision-recall curve ceiling.
    - cholesterol_age : Interaction term (cholesterol * age).
                        Elevated cholesterol is more damaging in older patients;
                        this interaction captures the compounding effect that a
                        linear combination of the two features cannot represent.
    """
    df = df.copy()
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["age_bmi"] = df["age"] * df["bmi"]
    df["bp_hypertension"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
    df["cholesterol_age"] = df["cholesterol"] * df["age"]
    logger.info(
        "Engineered features: pulse_pressure (mean=%.1f), age_bmi (mean=%.1f), "
        "bp_hypertension (positive rate=%.2f), cholesterol_age (mean=%.1f)",
        df["pulse_pressure"].mean(),
        df["age_bmi"].mean(),
        df["bp_hypertension"].mean(),
        df["cholesterol_age"].mean(),
    )
    return df


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def run_pipeline(path: str) -> dict:
    """
    Execute the full preprocessing pipeline.

    Returns a dict with keys:
        X_train, X_val, X_test           - feature arrays (70 / 15 / 15 split)
        y_train, y_val, y_test           - label arrays
        scaler                           - fitted StandardScaler
        feature_names                    - list of feature column names
        stats                            - dict of pipeline statistics

    The validation split is reserved exclusively for classification threshold
    tuning so that test-set metrics remain unbiased.
    """
    stats = {}

    # Load and clean
    df = load_data(path)
    stats["rows_raw"] = len(df)

    df = convert_age(df)
    df = handle_missing_values(df)
    stats["rows_after_missing"] = len(df)

    df = remove_outliers(df)
    stats["rows_after_outliers"] = len(df)

    df = calculate_bmi(df)
    df = engineer_features(df)
    stats["engineered_features"] = ["pulse_pressure", "age_bmi", "bp_hypertension", "cholesterol_age"]

    # Separate features and target
    target_col = "cardio"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    feature_names = feature_cols

    stats["num_features"] = len(feature_names)
    stats["positive_rate"] = float(np.mean(y))

    # Stratified 70 / 15 / 15 split: train / validation / test.
    # The validation split is used exclusively for threshold tuning so that
    # test-set metrics are never contaminated by the threshold search.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    stats["train_size"] = len(X_train)
    stats["val_size"] = len(X_val)
    stats["test_size"] = len(X_test)

    # Normalize — scaler fitted on training data only, applied to all three splits
    X_train, X_test, scaler = normalize_features(X_train, X_test)
    X_val = scaler.transform(X_val)

    logger.info(
        "Pipeline complete: %d train / %d val / %d test samples, %d features",
        len(X_train), len(X_val), len(X_test), len(feature_names),
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "stats": stats,
    }
