"""
=============================================================================
Telecom Churn Prediction Pipeline — Morocco Adaptation
=============================================================================
Base dataset : CSV1 (Kaggle Telco Churn / similar schema)
Enrichment   : Synthetic Moroccan context columns added to csv1
Target       : Compare Logistic Regression, Decision Tree, Random Forest,
               XGBoost, LightGBM on the same cleaned feature matrix
=============================================================================
Usage:
    python churn_pipeline.py --csv path/to/your_csv1.csv

If no CSV is provided, the script generates a realistic synthetic dataset
that mirrors the csv1 schema + Moroccan enrichment for testing.
=============================================================================
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import shap

# ─── Output folder ────────────────────────────────────────────────────────────
OUTPUT_DIR = "churn_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# =============================================================================
# 1. SYNTHETIC DATA GENERATOR  (used when no CSV is provided)
# =============================================================================

def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset that mirrors CSV1 schema and
    adds Moroccan-context columns. Churn logic is seeded from real telecom
    research so distributions are plausible.
    """
    rng = np.random.default_rng(SEED)

    # ── Core CSV1 columns ────────────────────────────────────────────────────
    # US State / Area code will be DROPPED — replaced by Moroccan Region
    intl_plan   = rng.choice(["yes", "no"], n, p=[0.1, 0.9])
    vm_plan     = rng.choice(["yes", "no"], n, p=[0.26, 0.74])
    n_vmail     = np.where(vm_plan == "yes", rng.integers(0, 50, n), 0)
    acct_len    = rng.integers(1, 200, n)

    day_min     = rng.uniform(0, 350, n)
    day_calls   = rng.integers(0, 150, n)
    day_charge  = day_min * 0.17

    eve_min     = rng.uniform(0, 350, n)
    eve_calls   = rng.integers(0, 150, n)
    eve_charge  = eve_min * 0.085

    night_min   = rng.uniform(0, 350, n)
    night_calls = rng.integers(0, 150, n)
    night_charge = night_min * 0.045

    intl_min    = rng.uniform(0, 20, n)
    intl_calls  = rng.integers(0, 20, n)
    intl_charge = intl_min * 0.27

    cs_calls    = rng.integers(0, 9, n)

    # ── Moroccan enrichment columns ──────────────────────────────────────────
    moroccan_regions = [
        "Casablanca", "Rabat", "Marrakech", "Fès",
        "Tanger", "Agadir", "Oujda", "Meknès"
    ]
    region = rng.choice(moroccan_regions, n, p=[0.30, 0.15, 0.12, 0.10,
                                                  0.10, 0.08, 0.08, 0.07])

    contract_type = rng.choice(["Prepaid", "Postpaid", "Hybrid"], n,
                                p=[0.55, 0.35, 0.10])

    # Network quality — correlated with region (urban areas better)
    urban_mask = np.isin(region, ["Casablanca", "Rabat", "Tanger"])
    base_speed = np.where(urban_mask, rng.uniform(30, 100, n),
                                      rng.uniform(5, 40, n))
    avg_download_speed = base_speed.clip(1, 100)

    outage_min  = rng.exponential(scale=15, size=n).clip(0, 300).astype(int)
    dropped_calls = (outage_min / 20 + rng.integers(0, 5, n)).clip(0, 30).astype(int)

    # Support CSAT (1–5) — lower for high cs_calls
    csat_base   = 4.5 - cs_calls * 0.4 + rng.uniform(-0.5, 0.5, n)
    csat_score  = csat_base.clip(1, 5).round(1)

    # Late payments — correlated with Prepaid
    late_pay    = np.where(
        contract_type == "Prepaid",
        rng.integers(0, 4, n),
        rng.integers(0, 2, n)
    )

    # Monthly charges in MAD (convert from USD charge basis ~×10)
    monthly_charges_mad = (day_charge + eve_charge + night_charge) * 3.5

    # ── Churn label construction (realistic signal weighting) ─────────────────
    # Build a churn score from known telecom drivers, then threshold
    churn_score = (
          0.35 * (cs_calls >= 4).astype(float)          # ≥4 service calls
        + 0.25 * (intl_plan == "yes").astype(float)      # intl plan friction
        + 0.20 * (dropped_calls >= 10).astype(float)     # bad network
        + 0.20 * (outage_min >= 60).astype(float)        # outages
        + 0.15 * (late_pay >= 2).astype(float)           # payment stress
        + 0.15 * (csat_score <= 2.5).astype(float)       # low satisfaction
        + 0.10 * (contract_type == "Prepaid").astype(float)
        + 0.10 * (monthly_charges_mad > monthly_charges_mad.mean() * 1.4).astype(float)
        + rng.uniform(0, 0.15, n)                         # noise
    )
    churn_prob = churn_score / churn_score.max()
    churn = (churn_prob > rng.uniform(0.45, 0.65, n)).astype(bool)

    df = pd.DataFrame({
        # CSV1 core (State / Area code dropped — replaced by Region)
        "Account_Length":          acct_len,
        "International_Plan":      intl_plan,
        "Voice_Mail_Plan":         vm_plan,
        "Num_Vmail_Messages":      n_vmail,
        "Total_Day_Minutes":       day_min.round(1),
        "Total_Day_Calls":         day_calls,
        "Total_Day_Charge":        day_charge.round(2),
        "Total_Eve_Minutes":       eve_min.round(1),
        "Total_Eve_Calls":         eve_calls,
        "Total_Eve_Charge":        eve_charge.round(2),
        "Total_Night_Minutes":     night_min.round(1),
        "Total_Night_Calls":       night_calls,
        "Total_Night_Charge":      night_charge.round(2),
        "Total_Intl_Minutes":      intl_min.round(2),
        "Total_Intl_Calls":        intl_calls,
        "Total_Intl_Charge":       intl_charge.round(2),
        "Customer_Service_Calls":  cs_calls,
        # Moroccan enrichment
        "Region":                  region,
        "Contract_Type":           contract_type,
        "Avg_Download_Speed_Mbps": avg_download_speed.round(1),
        "Network_Outage_Minutes":  outage_min,
        "Dropped_Calls_Count":     dropped_calls,
        "Support_CSAT_Score":      csat_score,
        "Late_Payment_Count":      late_pay,
        "Monthly_Charges_MAD":     monthly_charges_mad.round(2),
        # Label
        "Churn":                   churn,
    })
    return df


# =============================================================================
# 2. INGESTION  (load real CSV1 or generate synthetic)
# =============================================================================

def load_csv1(path: str) -> pd.DataFrame:
    """
    Load real CSV1 (Kaggle Telco / standard churn schema).
    Normalises column names and drops US-specific columns.
    """
    df = pd.read_csv(path)
    # Normalise column names → snake_case, no spaces
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(" ", "_")
          .str.replace(".", "", regex=False)
    )

    # Drop US geographic columns — not meaningful for Morocco
    us_cols = [c for c in df.columns if c.lower() in {"state", "area_code"}]
    if us_cols:
        df.drop(columns=us_cols, inplace=True)
        print(f"[INFO] Dropped US-specific columns: {us_cols}")

    # Standardise churn column to bool
    churn_col = next((c for c in df.columns if "churn" in c.lower()), None)
    if churn_col and churn_col != "Churn":
        df.rename(columns={churn_col: "Churn"}, inplace=True)
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"True": True, "False": False,
                                        "Yes": True, "No": False,
                                        True: True, False: False})

    return df


def add_moroccan_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append synthetic Moroccan-context columns to a real CSV1 dataframe.
    Distributions are seeded from churn signals already present in the data.
    """
    rng = np.random.default_rng(SEED)
    n = len(df)

    # Infer cs_calls column name
    cs_col = next(
        (c for c in df.columns if "service" in c.lower() and "call" in c.lower()),
        None
    )
    cs_arr = df[cs_col].values if cs_col else rng.integers(0, 5, n)

    moroccan_regions = [
        "Casablanca", "Rabat", "Marrakech", "Fès",
        "Tanger", "Agadir", "Oujda", "Meknès"
    ]
    df["Region"] = rng.choice(moroccan_regions, n,
                               p=[0.30,0.15,0.12,0.10,0.10,0.08,0.08,0.07])

    df["Contract_Type"] = rng.choice(
        ["Prepaid", "Postpaid", "Hybrid"], n, p=[0.55, 0.35, 0.10]
    )
    urban = np.isin(df["Region"].values, ["Casablanca", "Rabat", "Tanger"])
    df["Avg_Download_Speed_Mbps"] = np.where(
        urban, rng.uniform(30, 100, n), rng.uniform(5, 40, n)
    ).clip(1, 100).round(1)

    df["Network_Outage_Minutes"] = (
        rng.exponential(15, n).clip(0, 300).astype(int)
    )
    df["Dropped_Calls_Count"] = (
        df["Network_Outage_Minutes"].values / 20 + rng.integers(0, 5, n)
    ).clip(0, 30).astype(int)

    csat = (4.5 - cs_arr * 0.4 + rng.uniform(-0.5, 0.5, n)).clip(1, 5).round(1)
    df["Support_CSAT_Score"] = csat

    df["Late_Payment_Count"] = np.where(
        df["Contract_Type"] == "Prepaid",
        rng.integers(0, 4, n),
        rng.integers(0, 2, n)
    )
    # Monthly charges MAD derived from day charge column if present
    day_chg_col = next(
        (c for c in df.columns if "day" in c.lower() and "charge" in c.lower()),
        None
    )
    if day_chg_col:
        df["Monthly_Charges_MAD"] = (df[day_chg_col].values * 3.5 * 3).round(2)
    else:
        df["Monthly_Charges_MAD"] = rng.uniform(50, 500, n).round(2)

    return df


# =============================================================================
# 3. CLEANING
# =============================================================================

def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    original_shape = df.shape

    # 3.1 Duplicates
    dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    # 3.2 Nulls — impute rather than drop
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            fill_val = df[col].median()
            df[col].fillna(fill_val, inplace=True)
        else:
            fill_val = df[col].mode()[0]
            df[col].fillna(fill_val, inplace=True)
        if verbose:
            print(f"  [CLEAN] '{col}': imputed {null_count} nulls with {fill_val}")

    # 3.3 Clip extreme outliers (IQR × 3.0 fence — conservative)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Churn" in num_cols:
        num_cols.remove("Churn")
    for col in num_cols:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr     = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if clipped > 0:
            df[col] = df[col].clip(lower, upper)
            if verbose:
                print(f"  [CLEAN] '{col}': clipped {clipped} outliers")

    if verbose:
        print(f"\n[CLEAN] {original_shape} → {df.shape} "
              f"(removed {dupes} duplicates)\n")

    return df


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive predictive signals from raw columns.
    All new features are prefixed fe_ to make them identifiable.
    """

    # ── Usage ratios ──────────────────────────────────────────────────────────
    total_calls = (
        df.get("Total_Day_Calls", 0)
      + df.get("Total_Eve_Calls", 0)
      + df.get("Total_Night_Calls", 0)
    )
    df["fe_total_calls"] = total_calls

    total_minutes = (
        df.get("Total_Day_Minutes", 0)
      + df.get("Total_Eve_Minutes", 0)
      + df.get("Total_Night_Minutes", 0)
    )
    df["fe_total_minutes"] = total_minutes

    total_charge = (
        df.get("Total_Day_Charge", 0)
      + df.get("Total_Eve_Charge", 0)
      + df.get("Total_Night_Charge", 0)
    )
    df["fe_total_charge"] = total_charge

    # Charge per minute (efficiency of plan)
    df["fe_charge_per_minute"] = np.where(
        total_minutes > 0, total_charge / total_minutes, 0
    )

    # Day usage share (peak usage indicator)
    df["fe_day_usage_share"] = np.where(
        total_minutes > 0,
        df.get("Total_Day_Minutes", 0) / (total_minutes + 1),
        0
    )

    # ── International signals ─────────────────────────────────────────────────
    intl_col = next(
        (c for c in df.columns if "intl" in c.lower() and "plan" in c.lower()),
        None
    )
    if intl_col:
        df["fe_intl_plan_flag"] = (df[intl_col].str.lower() == "yes").astype(int)
    else:
        df["fe_intl_plan_flag"] = 0

    intl_min_col = next(
        (c for c in df.columns if "intl" in c.lower() and "minute" in c.lower()),
        None
    )
    if intl_min_col:
        df["fe_high_intl_usage"] = (
            df[intl_min_col] > df[intl_min_col].quantile(0.75)
        ).astype(int)

    # ── Customer service stress ───────────────────────────────────────────────
    cs_col = next(
        (c for c in df.columns if "service" in c.lower() and "call" in c.lower()),
        None
    )
    if cs_col:
        df["fe_high_cs_calls"] = (df[cs_col] >= 4).astype(int)
        df["fe_cs_calls_sq"]   = df[cs_col] ** 2  # captures non-linear jump

    # ── Moroccan enrichment-derived features ──────────────────────────────────
    if "Network_Outage_Minutes" in df.columns:
        df["fe_outage_flag"] = (df["Network_Outage_Minutes"] >= 60).astype(int)

    if "Dropped_Calls_Count" in df.columns:
        df["fe_bad_network"] = (df["Dropped_Calls_Count"] >= 10).astype(int)

    if "Support_CSAT_Score" in df.columns:
        df["fe_low_csat"] = (df["Support_CSAT_Score"] <= 2.5).astype(int)

    if "Late_Payment_Count" in df.columns:
        df["fe_payment_stress"] = (df["Late_Payment_Count"] >= 2).astype(int)

    if "Monthly_Charges_MAD" in df.columns:
        q75 = df["Monthly_Charges_MAD"].quantile(0.75)
        df["fe_high_bill"] = (df["Monthly_Charges_MAD"] > q75).astype(int)

    # Composite risk score (readable by CRM agents; not used in ML directly)
    risk_components = [c for c in df.columns if c.startswith("fe_") and
                       c in ["fe_high_cs_calls", "fe_bad_network",
                              "fe_outage_flag", "fe_low_csat",
                              "fe_payment_stress", "fe_intl_plan_flag"]]
    if risk_components:
        df["fe_composite_risk"] = df[risk_components].sum(axis=1)

    return df


# =============================================================================
# 5. PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, scale numerics, build X/y.
    Returns X, y, feature_names.
    """
    target = "Churn"
    y = df[target].astype(int)

    # Drop label + any identifier-style columns
    drop_cols = [target]
    X = df.drop(columns=drop_cols)

    # Encode all remaining categoricals
    cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_names = X.columns.tolist()
    return X, y, feature_names


# =============================================================================
# 6. CLASS IMBALANCE  (SMOTE)
# =============================================================================

def apply_smote(X_train, y_train):
    churn_rate = y_train.mean()
    print(f"\n[IMBALANCE] Training churn rate: {churn_rate:.1%}")
    if churn_rate < 0.30 or churn_rate > 0.70:
        smote = SMOTE(random_state=SEED, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Resampled from {len(y_train)} → {len(y_res)} samples, "
              f"new churn rate: {y_res.mean():.1%}")
        return X_res, y_res
    print("[IMBALANCE] Classes balanced enough — skipping SMOTE.")
    return X_train, y_train


# =============================================================================
# 7. MODEL DEFINITIONS
# =============================================================================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.1, solver="lbfgs",
            class_weight="balanced", random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=3,  # compensates class imbalance
            n_jobs=-1, random_state=SEED, verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True,
            n_jobs=-1, random_state=SEED, verbose=-1
        ),
    }


# =============================================================================
# 8. EVALUATION
# =============================================================================

THRESHOLD = 0.38  # optimised for recall over precision (missed churner > FP)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train, cross-validate, and evaluate a single model."""
    # Cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(model, X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1)

    # Train on full training set
    model.fit(X_train, y_train)

    # Predict with custom threshold
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= THRESHOLD).astype(int)
    else:
        proba = model.decision_function(X_test)
        preds = model.predict(X_test)

    results = {
        "Model":     name,
        "CV AUC":    f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}",
        "Accuracy":  accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall":    recall_score(y_test, preds, zero_division=0),
        "F1":        f1_score(y_test, preds, zero_division=0),
        "ROC AUC":   roc_auc_score(y_test, proba),
    }
    return results, model, proba


def print_results_table(all_results):
    df = pd.DataFrame(all_results)
    df = df.set_index("Model")
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    for c in numeric_cols:
        df[c] = df[c].apply(lambda x: f"{x:.3f}")
    print("\n" + "=" * 75)
    print("MODEL COMPARISON RESULTS")
    print("=" * 75)
    print(df.to_string())
    print("=" * 75)
    print(f"\n⚠  Classification threshold: {THRESHOLD} (tuned for recall)")
    print("   Key metric for churn: Recall + ROC AUC\n")


# =============================================================================
# 9. VISUALISATIONS
# =============================================================================

def plot_confusion_matrices(models_dict, X_test, y_test):
    """One confusion matrix per model in a single figure."""
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, (model, proba)) in zip(axes, models_dict.items()):
        preds = (proba >= THRESHOLD).astype(int)
        cm    = confusion_matrix(y_test, preds)
        disp  = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")

    plt.suptitle("Confusion Matrices — all models", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_roc_curves(models_dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#378ADD", "#639922", "#BA7517", "#A32D2D", "#7F77DD"]

    for (name, (model, proba)), color in zip(models_dict.items(), colors):
        auc = roc_auc_score(y_test, proba)
        RocCurveDisplay.from_predictions(
            y_test, proba, ax=ax, color=color,
            name=f"{name}  (AUC={auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_feature_importance(model, feature_names, model_name="XGBoost", top_n=20):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        imp = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    else:
        return

    imp = imp.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    colors_bar = ["#378ADD" if "fe_" in n else "#9BBFDE" for n in imp.index]
    imp.plot(kind="barh", ax=ax, color=colors_bar)
    ax.set_title(f"Feature Importance — {model_name} (top {top_n})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance score")
    # Annotate engineered features
    ax.text(
        0.98, 0.02, "■ Engineered (fe_)  □ Raw",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="gray"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feature_importance_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_shap_summary(model, X_test, feature_names, model_name="XGBoost"):
    """SHAP beeswarm summary — global feature importance with direction."""
    try:
        explainer = shap.TreeExplainer(model)
        # Use a sample to keep it fast
        sample = X_test.sample(min(500, len(X_test)), random_state=SEED)
        shap_values = explainer.shap_values(sample)

        # For classifiers that return array of shape (n, features, classes)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, sample,
            feature_names=feature_names,
            max_display=20,
            show=False,
            plot_size=(10, 8)
        )
        plt.title(f"SHAP Summary — {model_name}", fontsize=12, fontweight="bold")
        path = os.path.join(OUTPUT_DIR, f"shap_summary_{model_name.replace(' ', '_').lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Saved: {path}")
    except Exception as e:
        print(f"[SHAP] Could not generate SHAP plot for {model_name}: {e}")


def plot_churn_distribution(df: pd.DataFrame):
    """EDA: churn rate by key segments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Overall churn rate
    churn_counts = df["Churn"].value_counts()
    axes[0].pie(churn_counts.values,
                labels=["Active", "Churned"],
                colors=["#9BBFDE", "#D85A30"],
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Overall Churn Rate", fontweight="bold")

    # 2. Churn by contract type
    if "Contract_Type" in df.columns:
        ct = df.groupby("Contract_Type")["Churn"].mean().sort_values(ascending=False)
        ct.plot(kind="bar", ax=axes[1], color=["#D85A30", "#378ADD", "#639922"],
                rot=0, edgecolor="white")
        axes[1].set_title("Churn Rate by Contract Type", fontweight="bold")
        axes[1].set_ylabel("Churn Rate")
        axes[1].set_ylim(0, 1)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        for bar in axes[1].patches:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                         f"{h:.1%}", ha="center", va="bottom", fontsize=9)

    # 3. Churn by region
    if "Region" in df.columns:
        rg = df.groupby("Region")["Churn"].mean().sort_values(ascending=True)
        rg.plot(kind="barh", ax=axes[2], color="#378ADD", edgecolor="white")
        axes[2].set_title("Churn Rate by Moroccan Region", fontweight="bold")
        axes[2].set_xlabel("Churn Rate")
        axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_churn_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


# =============================================================================
# 10. EXPORT
# =============================================================================

def export_predictions(model, X_test, y_test, feature_names, model_name="XGBoost"):
    """Save a predictions CSV with churn probability + risk band."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)

    def risk_band(p):
        if p >= 0.70:   return "High"
        elif p >= 0.40: return "Medium"
        else:           return "Low"

    out = X_test.copy()
    out.columns = feature_names
    out["actual_churn"]      = y_test.values
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"]  = preds
    out["risk_band"]         = [risk_band(p) for p in proba]

    path = os.path.join(OUTPUT_DIR, f"predictions_{model_name.replace(' ', '_').lower()}.csv")
    out.to_csv(path, index=False)
    print(f"[EXPORT] Predictions saved: {path}")
    print(f"         High risk: {(out['risk_band']=='High').sum():,}  |  "
          f"Medium: {(out['risk_band']=='Medium').sum():,}  |  "
          f"Low: {(out['risk_band']=='Low').sum():,}")


def export_metrics(all_results):
    df = pd.DataFrame(all_results)
    path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)
    print(f"[EXPORT] Metrics table saved: {path}")


# =============================================================================
# 11. MAIN PIPELINE
# =============================================================================

def run_pipeline(csv_path: str = None):
    print("\n" + "=" * 65)
    print("  TELECOM CHURN PREDICTION — MOROCCO PIPELINE")
    print("=" * 65)

    # ── Step 1: Data ingestion ────────────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        print(f"\n[STEP 1] Loading CSV: {csv_path}")
        df = load_csv1(csv_path)
        print(f"         Loaded {len(df):,} rows × {len(df.columns)} columns")
        print("[STEP 1] Adding Moroccan enrichment columns...")
        df = add_moroccan_enrichment(df)
    else:
        print("\n[STEP 1] No CSV provided — generating synthetic Morocco dataset")
        df = generate_synthetic_data(n=5000)
    print(f"         Dataset shape: {df.shape}")
    print(f"         Churn rate: {df['Churn'].mean():.1%}")

    # ── Step 2: EDA plot ──────────────────────────────────────────────────────
    print("\n[STEP 2] Generating EDA visualisations...")
    plot_churn_distribution(df)

    # ── Step 3: Clean ─────────────────────────────────────────────────────────
    print("\n[STEP 3] Cleaning data...")
    df = clean(df, verbose=True)

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    print("\n[STEP 4] Engineering features...")
    df = engineer_features(df)
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    print(f"         Created {len(fe_cols)} engineered features: {fe_cols}")

    # ── Step 5: Preprocess ────────────────────────────────────────────────────
    print("\n[STEP 5] Preprocessing (encode + prepare X/y)...")
    X, y, feature_names = preprocess(df)
    print(f"         X shape: {X.shape}  |  churn rate: {y.mean():.1%}")

    # ── Step 6: Train/test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\n[STEP 6] Split: {len(X_train):,} train / {len(X_test):,} test "
          f"(stratified, 80/20)")

    # ── Step 7: SMOTE ─────────────────────────────────────────────────────────
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Convert to DataFrames for SHAP compatibility
    X_train_res = pd.DataFrame(X_train_res, columns=feature_names)
    X_test      = pd.DataFrame(X_test,      columns=feature_names)

    # ── Step 8: Train & evaluate all models ───────────────────────────────────
    print("\n[STEP 8] Training & evaluating all models...")
    models = get_models()
    all_results    = []
    trained_models = {}   # name → (model, proba)

    for name, model in models.items():
        print(f"         → {name}...", end=" ", flush=True)
        results, trained, proba = evaluate_model(
            name, model, X_train_res, y_train_res, X_test, y_test
        )
        all_results.append(results)
        trained_models[name] = (trained, proba)
        print(f"AUC={results['ROC AUC']:.3f}  Recall={results['Recall']:.3f}")

    # ── Step 9: Results table ─────────────────────────────────────────────────
    print_results_table(all_results)
    export_metrics(all_results)

    # ── Step 10: Visualisations ───────────────────────────────────────────────
    print("[STEP 10] Generating visualisations...")
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)

    # Feature importance for tree models
    for name in ["XGBoost", "Random Forest", "LightGBM"]:
        if name in trained_models:
            plot_feature_importance(
                trained_models[name][0], feature_names, model_name=name
            )

    # SHAP for XGBoost (primary model)
    if "XGBoost" in trained_models:
        print("[STEP 10] Computing SHAP values for XGBoost...")
        plot_shap_summary(
            trained_models["XGBoost"][0], X_test,
            feature_names, model_name="XGBoost"
        )

    # ── Step 11: Export predictions (best model by ROC AUC) ──────────────────
    best_name = max(
        {n: r["ROC AUC"] for n, r in
         zip([r["Model"] for r in all_results], all_results)}.items(),
        key=lambda x: x[1]
    )[0]
    # Rebuild: name → auc
    name_to_auc = {r["Model"]: r["ROC AUC"] for r in all_results}
    best_name   = max(name_to_auc, key=name_to_auc.get)
    print(f"\n[STEP 11] Best model by ROC AUC: {best_name}")
    export_predictions(
        trained_models[best_name][0], X_test, y_test,
        feature_names, model_name=best_name
    )

    print(f"\n{'=' * 65}")
    print(f"  PIPELINE COMPLETE — outputs in ./{OUTPUT_DIR}/")
    print(f"{'=' * 65}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telecom Churn Prediction — Morocco Pipeline"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV1-schema dataset. If omitted, synthetic data is generated."
    )
    args = parser.parse_args()
    run_pipeline(csv_path=args.csv)