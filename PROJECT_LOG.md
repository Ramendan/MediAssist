# MediAssist - Project Technical Report

**Generated:** 2026-03-13 15:36:44
**Module:** AI-Powered Medical Diagnosis Support System
**Dataset:** Cardiovascular Disease (Kaggle - sulianova/cardiovascular-disease, 70,000 patients)

---

## Abstract

MediAssist is a cardiovascular disease risk assessment system that combines three
machine learning models with a rule-based clinical knowledge engine. The system
is optimised for **Recall (Sensitivity)**, the medically correct primary metric
for disease screening, where missing a sick patient is far more harmful than a
false alarm.

| Component | Detail |
|-----------|--------|
| **Dataset** | 70,000 patient records, 12 features + 3 engineered features |
| **Models trained** | Logistic Regression, Random Forest (tuned), LightGBM (tuned) |
| **Selected model** | Random Forest |
| **Recall at default threshold** | 0.7034 |
| **Recall after threshold tuning** | 0.9819 |
| **Tuned threshold** | 0.1226 (F2-score optimised on validation set) |
| **ROC AUC** | 0.7871 |
| **Primary metric** | Recall: minimises false negatives (missed diagnoses) |

---

## 1. Dataset Overview

- **Source:** Kaggle - Cardiovascular Disease Dataset (sulianova/cardiovascular-disease)
- **Original rows:** 70,000
- **Rows after handling missing values:** 70,000
- **Rows after outlier removal:** 68,636
- **Rows removed (outliers):** 1,364 (1.9%)
- **Features used:** 14 (age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi, pulse_pressure, age_bmi)
- **Target variable:** cardio (0 = no disease, 1 = disease)
- **Positive class rate:** 49.47%
- **Train/Val/Test split:** 48,045 / 10,295 / 10,296 (70/15/15, stratified)
- **Validation split purpose:** Threshold tuning only; never used for model training or final test-set reporting

---

## 2. Preprocessing Steps

| Step | Description |
|------|-------------|
| Age conversion | Converted from days to years (`age / 365.25`) |
| Missing value handling | Dropped rows with any NaN values |
| Outlier removal | Removed biologically implausible rows: height 100-220 cm, weight 30-200 kg, systolic BP 60-250 mmHg, diastolic BP 40-160 mmHg, and enforced systolic > diastolic |
| BMI calculation | `weight / (height / 100)^2` |
| Feature engineering | Added `pulse_pressure` and `age_bmi` interaction term (see Section 3) |
| Normalization | `StandardScaler` fitted on training data only, then applied to all three splits using training statistics (no leakage) |

---

## 3. Feature Engineering

Two clinically-motivated derived features were added to improve predictive power:

| Feature | Formula | Clinical Rationale |
|---------|---------|-------------------|
| `pulse_pressure` | `ap_hi - ap_lo` | A pulse pressure above 60 mmHg is an independent predictor of cardiovascular events, particularly in older adults. It reflects arterial stiffness, which is not captured by either systolic or diastolic BP alone. |
| `age_bmi` | `age * bmi` | An interaction term that captures compounded metabolic-aging risk. Obesity in older patients carries disproportionately higher cardiovascular risk than in younger patients, and a linear model cannot capture this without the explicit interaction. |

---

## 4. Model Comparison (Default Threshold = 0.50)

Three models were trained and compared. All use `class_weight='balanced'` to
counteract the approximately 50/50 class distribution and favor recall.

| Metric | Logistic Regression | Random Forest | LightGBM |
|--------|---: | ---: | ---:|
| **Accuracy** | 0.7237 | 0.7247 | 0.7306 |
| **Precision** | 0.7422 | 0.7303 | 0.7436 |
| **Recall** | 0.6765 | 0.7034 | 0.6951 |
| **F1 Score** | 0.7078 | 0.7166 | 0.7185 |
| **ROC AUC** | 0.7886 | 0.7871 | 0.7991 |

**Selected model:** Random Forest (primary criterion: Recall)

---

## 5. Threshold Tuning

### Methodology

After model selection, the classification probability threshold was optimized
using the **F-beta score (beta=2)**. This metric weights recall twice as heavily
as precision, reflecting the clinical cost asymmetry in medical screening:

> A missed cardiovascular disease case (false negative) is far more harmful than
> an unnecessary follow-up consultation (false positive).

The precision-recall curve was computed on a **dedicated validation split**
(15% of the dataset, held out from both training and the final test set) across
all candidate thresholds. The threshold maximizing F2 was selected and persisted.
Using a separate validation set ensures the reported test-set metrics are unbiased
and reflect true generalisation performance.

### Before vs. After Threshold Tuning (Random Forest)

| Metric | Default (0.50) | Tuned (0.1226) | Change |
|--------|---------------:|----------------:|--------|
| **Recall** | 0.7034 | 0.9819 | +0.2785 |
| **Precision** | 0.7303 | 0.5227 | -0.2076 |
| **F1 Score** | 0.7166 | 0.6822 | -0.0344 |
| **Accuracy** | 0.7247 | 0.5474 | -0.1773 |

**Optimal threshold:** 0.1226

---

## 6. Classification Reports

### Logistic Regression
```
precision    recall  f1-score   support

           0       0.71      0.77      0.74      5202
           1       0.74      0.68      0.71      5094

    accuracy                           0.72     10296
   macro avg       0.73      0.72      0.72     10296
weighted avg       0.73      0.72      0.72     10296
```

### Random Forest
```
precision    recall  f1-score   support

           0       0.72      0.75      0.73      5202
           1       0.73      0.70      0.72      5094

    accuracy                           0.72     10296
   macro avg       0.72      0.72      0.72     10296
weighted avg       0.72      0.72      0.72     10296
```

### LightGBM
```
precision    recall  f1-score   support

           0       0.72      0.77      0.74      5202
           1       0.74      0.70      0.72      5094

    accuracy                           0.73     10296
   macro avg       0.73      0.73      0.73     10296
weighted avg       0.73      0.73      0.73     10296
```

---

## 7. Technical Decisions and Methodology Justification

### Why Recall, Not Accuracy

Accuracy measures the proportion of all predictions that are correct. It treats
a missed diagnosis (false negative) and an unnecessary follow-up (false positive)
as identical errors. In cardiovascular screening, they are not:

| Error Type | What it means | Clinical consequence |
|------------|---------------|----------------------|
| **False Negative** | Model predicts healthy; patient has disease | Patient goes untreated, potentially fatal |
| **False Positive** | Model predicts disease; patient is healthy | Patient receives a follow-up consultation, inconvenient but not dangerous |

**Recall (Sensitivity) = TP / (TP + FN)** directly measures the proportion of
actual disease cases the model identifies. A high Recall means fewer missed
diagnoses. This is the medically correct primary metric for any screening tool.

### Why F2-Score for Threshold Tuning

F1-score weights Precision and Recall equally (beta=1). Since we deliberately
accept more false positives to reduce false negatives, **F2-score (beta=2)**
weights Recall twice as heavily as Precision. It is the correct optimisation
target when the cost of a false negative exceeds the cost of a false positive.

```
F2 = 5 * precision * recall / (4 * precision + recall)
```

### Why a Separate Validation Split for Threshold Tuning

If the threshold is tuned on the test set, the test metrics no longer reflect
unseen data; the threshold has effectively seen the test set and the reported
Recall is optimistically biased. By tuning on a dedicated 15% validation split,
the 15% test set remains truly held-out, so reported metrics accurately reflect
what would be observed in deployment.

### Other Decisions

- **Three-model comparison:** Logistic Regression (interpretable baseline), Random Forest
  (ensemble method), and LightGBM (gradient boosting) were trained and compared.
  LightGBM uses leaf-wise tree growth and is typically the strongest performer on
  tabular clinical data.
- **Hyperparameter search:** `RandomizedSearchCV`, 30 iterations, 5-fold cross-validation,
  scored on Recall for both Random Forest and LightGBM.
- **`class_weight='balanced'`:** Applied to all models to counteract class imbalance
  without oversampling, biasing each model toward correct positive detection.
- **Feature engineering:** `pulse_pressure` (arterial stiffness proxy) and `age_bmi`
  (aging-obesity interaction) added after BMI calculation, before normalization.
- **No data leakage:** `StandardScaler` is fitted exclusively on the 70% training
  split and applied to all other splits using training statistics.
- **Reproducibility:** `random_state=42` used across all stochastic operations.
- **Outlier removal:** Based on clinically accepted biological ranges to prevent
  the model from learning from data-entry errors.

---

## 8. Ethical Limitations and Disclaimers

- **Not a clinical diagnostic tool.** This system is designed for educational and
  decision-support purposes only. It must not be used as a substitute for professional
  medical evaluation, diagnosis, or treatment.
- **Dataset bias:** The dataset originates from a specific geographic and demographic
  context. Model performance may not generalise across populations with different age
  distributions, ethnic backgrounds, or healthcare access patterns.
- **No temporal validation:** The model has not been validated on prospective or
  time-shifted data. Performance in real-world deployment may differ from reported metrics.
- **Limited feature set:** Important cardiovascular risk factors such as family history,
  LDL/HDL levels, troponin, ECG data, and medication history are not present in the dataset.
- **Binary classification simplification:** Cardiovascular disease is a spectrum.
  The binary (0/1) output oversimplifies clinical reality.
- **Recall vs. precision trade-off:** Optimizing for recall increases false positives,
  which could cause unnecessary patient anxiety and healthcare resource consumption.

---

## 9. Generated Artifacts

| File | Description |
|------|-------------|
| `models/final_model.pkl` | Trained Random Forest model |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/feature_names.pkl` | Ordered feature name list (includes engineered features) |
| `models/threshold.pkl` | Optimal classification threshold (0.1226) |
| `plots/learning_curves.png` | Recall vs. training set size (bias-variance analysis) |
| `plots/confusion_matrix.png` | Confusion matrix at tuned threshold (0.1226) |
| `plots/feature_importance.png` | Feature importance ranked bar chart |
| `plots/precision_recall_curve.png` | PR curve with default and optimal threshold annotated |
| `plots/roc_curve.png` | ROC curve for all three models with tuned threshold marked |

---

## 10. Evaluation Plots

### Learning Curves: Recall vs. Training Set Size

Shows how training and cross-validation Recall change as more data is used.
Converging curves indicate the model has learned adequately; a large gap
indicates overfitting.

![Learning Curves](plots/learning_curves.png)

---

### Confusion Matrix (Tuned Threshold = 0.1226)

Shows the counts of correct and incorrect predictions at the operating threshold.
At the tuned threshold, the model prioritises minimising False Negatives (bottom-left
cell) at the cost of more False Positives (top-right cell), the clinically correct
trade-off.

- **True Negative (top-left):** Healthy patients correctly identified as healthy
- **False Positive (top-right):** Healthy patients flagged for follow-up (acceptable)
- **False Negative (bottom-left):** Sick patients missed; the error we minimise
- **True Positive (bottom-right):** Sick patients correctly detected

![Confusion Matrix](plots/confusion_matrix.png)

---

### Feature Importance: Random Forest

Ranks each feature by its contribution to the model's predictions.
Engineered features (`pulse_pressure`, `age_bmi`) are included and ranked
relative to the original dataset features.

![Feature Importance](plots/feature_importance.png)

---

### Precision-Recall Curve

The full trade-off curve between Precision and Recall across all probability
thresholds. The star marks the F2-optimal operating point (threshold = 0.1226).
The diamond marks the standard 0.50 operating point for comparison.
Higher area under the curve (AP score) indicates better overall performance.

![Precision-Recall Curve](plots/precision_recall_curve.png)

---

### ROC Curve: All Three Models

Plots True Positive Rate (Recall) against False Positive Rate for all three models.
The star marks the tuned threshold operating point on the selected model.
AUC = 1.0 is perfect; AUC = 0.5 is random. All three models are shown for comparison.

![ROC Curve](plots/roc_curve.png)
