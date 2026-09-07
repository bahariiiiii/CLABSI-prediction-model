# ==============================================================================
# benchmarking.py 
# ==============================================================================

# ==============================================================================
# SECTION 1 — Environment setup 
# ==============================================================================

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from scipy import stats
from scipy.stats import fisher_exact, norm
from statsmodels.stats.proportion import proportion_confint
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# ==============================================================================
# SECTION 2 — Dataset path, feature sets, target, output tree
# ==============================================================================

DATA_PATH = Path(os.environ.get("CLABSI_DATA_PATH", "data/clabsi_dataset.xlsx"))

target = "CLABSI"
group_column = "ICU Name"
patient_id_col = "Patient_Identifier"

categorical_features = [
    "Seasonofinsertion", "Sex", "ICUSource", "HospitalSource", "Smoking Status",
    "Admissionin3months", "Immunosuppressed", "INV", "Inotropes", "Diabetes",
    "Plannedadmission", "RenalRep", "Thrombopro", "HospitalizationType",
    "ICU Type", "Immune_Disease", "Metastases", "Chronic_respiratory",
    "Chronic_Cardiovascular", "Chronic_Renal_failure", "Antibiotictype_yes",
]

numerical_features = [
    "PreICULOS", "Antibiotictype", "BMI", "TPNDuration", "BS", "BUN", "PT",
    "RDW", "MCHC", "MCH", "MCV", "Neut", "Lymphocyte", "Age", "Frailty",
    "Temp", "HR", "MAP", "RR", "Na", "K_1", "HCO3", "Creat", "Albumin",
    "HCT", "HMG", "WBC", "PLT", "PH", "GCS", "APACHEII", "Bilirubin",
]

all_features = categorical_features + numerical_features

TOP_K_FEATURES = 15                 
CALIBRATION_METHOD = "sigmoid"
N_BOOTSTRAPS = 2000
CI_LEVEL = 95

CONTRACT_CONFIG = {
    "n_bins_calibration": 10,
    "n_bins_histogram": 10,
    "calibration_strategy": "quantile",
    "risk_percentiles": (20, 80),
    "top_k_features": TOP_K_FEATURES,
    "random_seed": SEED,
    "main_cv_splits": 5,
    "loio_group_col": group_column,
}

MODEL_NAMES = ["SVM", "LR", "RF", "XGB", "MLP"]
REFERENCE_MODEL = "SVM"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"benchmark_clabsi_{RUN_TIMESTAMP}")

# ساخت MODEL_DIRS و COMPARISON_DIR دقیقاً مثل SVM
MODEL_DIRS = {}
for name in MODEL_NAMES:
    base = OUTPUT_DIR / name
    main = base / "main_model"
    loio = base / "loio"
    dirs = {
        "main_fig": main / "figures",
        "main_table": main / "tables",
        "main_info": main / "model_info",
        "loio_fig": loio / "figures",
        "loio_table": loio / "tables",
        "loio_info": loio / "model_info",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    MODEL_DIRS[name] = dirs

COMPARISON_DIR = OUTPUT_DIR / "comparison"
COMPARISON_FIG_DIR = COMPARISON_DIR / "figures"
COMPARISON_TABLE_DIR = COMPARISON_DIR / "tables"
for d in (COMPARISON_FIG_DIR, COMPARISON_TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"Run timestamp: {RUN_TIMESTAMP}")
print(f"Root output directory: {OUTPUT_DIR.resolve()}")
print(f"Models benchmarked: {MODEL_NAMES}")

# ==============================================================================
# SECTION 3 — Model registry 
# ==============================================================================

HAS_DECISION_FUNCTION = {"SVM": True, "LR": True, "RF": False, "XGB": False, "MLP": False}
SHAP_FAMILY = {"SVM": "linear", "LR": "linear", "RF": "tree", "XGB": "tree", "MLP": "permutation"}
SUPPORTS_LINEAR_COEFFICIENTS = {"SVM": True, "LR": True, "RF": False, "XGB": False, "MLP": False}

def build_estimator(model_name, seed=SEED):
    if model_name == "SVM":
        return SVC(kernel="linear", probability=False, random_state=seed)
    if model_name == "LR":
        return LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    if model_name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=-1)
    if model_name == "XGB":
        return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1)
    if model_name == "MLP":
        return MLPClassifier(random_state=seed, max_iter=800, early_stopping=True)
    raise ValueError(f"Unknown model_name: {model_name}")

_CLASS_WEIGHT_GRID = [None, "balanced", {0: 1, 1: 2}]

PARAM_GRIDS = {
    "SVM": {
        "classifier__C": [0.01, 0.1, 0.5, 1],
        "classifier__class_weight": _CLASS_WEIGHT_GRID,
    },
    "LR": {
        "classifier__C": [0.01, 0.1, 0.5, 1],
        "classifier__class_weight": _CLASS_WEIGHT_GRID,
    },
    "RF": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 6, None],
        "classifier__class_weight": _CLASS_WEIGHT_GRID,
    },
    "XGB": {
        "classifier__n_estimators": [80, 150],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__scale_pos_weight": [2, 3],
    },
    "MLP": {
        "classifier__hidden_layer_sizes": [(8,), (16,)],
        "classifier__alpha": [5e-4, 1e-3, 5e-3, 1e-2],
        "classifier__learning_rate_init": [1e-4, 1e-3],
        "classifier__early_stopping": [True],
        "classifier__validation_fraction": [0.25],
        "classifier__n_iter_no_change": [10],
        "classifier__max_iter": [600, 800],
    },
}

# define related functions

def get_raw_score(pipeline, X_df, model_name):
    if HAS_DECISION_FUNCTION[model_name]:
        return pipeline.decision_function(X_df)
    return pipeline.predict_proba(X_df)[:, 1]

def make_auc_scorer(model_name):
    def _scorer(estimator, X, y):
        scores = get_raw_score(estimator, X, model_name)
        return roc_auc_score(y, scores)
    return _scorer

def _make_preprocessor(cat_cols, num_cols):
    try:
        cat_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
                      ("cat", cat_encoder, cat_cols)],
        remainder="drop",
    )

def _build_pipeline(model_name, cat_cols, num_cols, seed):
    return Pipeline(steps=[
        ("preprocessor", _make_preprocessor(cat_cols, num_cols)),
        ("classifier", build_estimator(model_name, seed)),
    ])

# ==============================================================================
# SECTION 4 — Evaluation Metrics and Calibration functions
# ==============================================================================

def calculate_metrics(y_true, y_proba, threshold=0.5):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_PPV": precision_score(y_true, y_pred, zero_division=0),
        "Recall_Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "F1_score": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "PR_AUC": average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "Brier": brier_score_loss(y_true, y_proba),
    }

def calibration_slope_intercept(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    eps = 1e-6
    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
    logit_p = np.log(y_proba_clipped / (1 - y_proba_clipped)).reshape(-1, 1)
    try:
        cal_model = LogisticRegression(penalty=None, solver="lbfgs").fit(logit_p, y_true)
        return cal_model.intercept_[0], cal_model.coef_[0][0]
    except Exception:
        return np.nan, np.nan

def calculate_all_metrics(y_true, y_proba, threshold=0.5):
    metrics = calculate_metrics(y_true, y_proba, threshold)
    cal_intercept, cal_slope = calibration_slope_intercept(y_true, y_proba)
    metrics["Calibration_intercept"] = cal_intercept
    metrics["Calibration_slope"] = cal_slope
    return metrics

# ==============================================================================
# SECTION 5 — Preprocessing and building pipeline functions
# ==============================================================================

def _make_preprocessor(cat_cols, num_cols):
    try:
        cat_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
                      ("cat", cat_encoder, cat_cols)],
        remainder="drop",
    )

def _build_pipeline(model_name, cat_cols, num_cols, seed):
    return Pipeline(steps=[
        ("preprocessor", _make_preprocessor(cat_cols, num_cols)),
        ("classifier", build_estimator(model_name, seed)),
    ])

# ==============================================================================
# SECTION 6 — SHAP importance and 3-stage tuning process 
# ==============================================================================

def _shap_values_for_positive_class(explainer_output):
    values = explainer_output.values if hasattr(explainer_output, "values") else explainer_output
    if np.ndim(values) == 3:
        values = values[:, :, 1]
    return values

def compute_shap_importance(model_name, fitted_classifier, X_background_df, X_to_explain_df):
    family = SHAP_FAMILY[model_name]
    if family == "linear":
        explainer = shap.LinearExplainer(fitted_classifier, X_background_df)
        shap_values = _shap_values_for_positive_class(explainer(X_to_explain_df))
    elif family == "tree":
        explainer = shap.TreeExplainer(fitted_classifier)
        raw = explainer(X_to_explain_df)
        shap_values = _shap_values_for_positive_class(raw)
    elif family == "permutation":
        predict_fn = lambda data: fitted_classifier.predict_proba(data)[:, 1]
        explainer = shap.PermutationExplainer(predict_fn, X_background_df)
        shap_values = explainer(X_to_explain_df).values
    else:
        raise ValueError(f"Unknown SHAP family: {family}")
    return np.asarray(shap_values)

def select_features_and_tune(
    model_name,
    X_train,
    y_train,
    categorical_features,
    numerical_features,
    param_grid,
    seed=SEED,
    top_k=TOP_K_FEATURES,
    cv_splits=5,
):
    cat_avail = [c for c in categorical_features if c in X_train.columns]
    num_avail = [c for c in numerical_features if c in X_train.columns]

    n_splits = max(2, min(cv_splits, int(np.min(np.bincount(np.asarray(y_train))))))
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorer = make_auc_scorer(model_name)

    full_pipeline = _build_pipeline(model_name, cat_avail, num_avail, seed)
    full_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, scoring=scorer,
                               cv=inner_cv, n_jobs=-1, refit=True)
    full_search.fit(X_train[cat_avail + num_avail], y_train)
    full_best_model = full_search.best_estimator_

    fitted_preprocessor = full_best_model.named_steps["preprocessor"]
    classifier_step = full_best_model.named_steps["classifier"]

    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array(
            num_avail + list(fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(cat_avail))
        )
    feature_names = np.array(feature_names)

    X_train_processed = fitted_preprocessor.transform(X_train[cat_avail + num_avail])
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)

    background_size = min(100, X_train_processed_df.shape[0])
    X_background = shap.sample(X_train_processed_df, background_size, random_state=seed)

    shap_values_array = compute_shap_importance(model_name, classifier_step, X_background, X_train_processed_df)

    importance_rows = []
    for i, f_name in enumerate(feature_names):
        clean_name = f_name.replace("num__", "").replace("cat__", "")
        orig_var = clean_name
        for cat_var in cat_avail:
            if clean_name.startswith(cat_var + "_"):
                orig_var = cat_var
                break
        importance_rows.append({"Original_Variable": orig_var, "Abs_SHAP": np.abs(shap_values_array[:, i]).mean()})

    shap_agg = pd.DataFrame(importance_rows).groupby("Original_Variable")["Abs_SHAP"].sum().sort_values(ascending=False)
    selected_features = shap_agg.head(top_k).index.tolist()

    sel_cat = [c for c in categorical_features if c in selected_features]
    sel_num = [c for c in numerical_features if c in selected_features]

    final_pipeline_template = _build_pipeline(model_name, sel_cat, sel_num, seed)
    final_search = GridSearchCV(estimator=final_pipeline_template, param_grid=param_grid, scoring=scorer,
                                cv=inner_cv, n_jobs=-1, refit=True)
    final_search.fit(X_train[selected_features], y_train)

    return {
        "final_pipeline": final_search.best_estimator_,
        "selected_features": selected_features,
        "full_feature_best_params": full_search.best_params_,
        "final_best_params": final_search.best_params_,
        "shap_importance": shap_agg,
    }

# ==============================================================================
# SECTION 7 — Calibration and threshold tuning 
# ==============================================================================

def fit_calibrator(model_name, pipeline, X_cal, y_cal, selected_features, method=CALIBRATION_METHOD):
    raw_scores = get_raw_score(pipeline, X_cal[selected_features], model_name).reshape(-1, 1)
    y_cal_arr = np.asarray(y_cal)
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_scores.ravel(), y_cal_arr)
    elif method == "sigmoid":
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(raw_scores, y_cal_arr)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}.")
    return calibrator

def apply_calibrator(model_name, pipeline, calibrator, X, selected_features, method=CALIBRATION_METHOD):
    raw_scores = get_raw_score(pipeline, X[selected_features], model_name).reshape(-1, 1)
    if method == "isotonic":
        proba = calibrator.predict(raw_scores.ravel())
    else:
        proba = calibrator.predict_proba(raw_scores)[:, 1]
    return np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)

def tune_threshold_on_val(model_name, pipeline, X_val, selected_features, y_val, calibrator, method=CALIBRATION_METHOD):
    proba = apply_calibrator(model_name, pipeline, calibrator, X_val, selected_features, method=method)
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls), out=np.zeros_like(precisions),
                          where=(precisions + recalls) != 0)
    best_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return best_threshold, proba

def calibrate_and_tune_threshold(model_name, pipeline, X_val, y_val, selected_features, method=CALIBRATION_METHOD):
    calibrator = fit_calibrator(model_name, pipeline, X_val, y_val, selected_features, method=method)
    best_threshold, val_proba = tune_threshold_on_val(model_name, pipeline, X_val, selected_features, y_val, calibrator, method=method)
    return calibrator, best_threshold, val_proba

# ==============================================================================
# SECTION 8 — Stratified bootstrap 95% CI 
# ==============================================================================

def bootstrap_metrics_ci(y_true, y_proba, threshold, point_estimates, n_bootstraps=N_BOOTSTRAPS, ci=CI_LEVEL, random_state=SEED):
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")

    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    bootstrap_results = []
    for _ in range(n_bootstraps):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sampled_idx = np.concatenate([s_pos, s_neg])
        rng.shuffle(sampled_idx)
        bootstrap_results.append(calculate_all_metrics(y_true[sampled_idx], y_proba[sampled_idx], threshold))

    bootstrap_df = pd.DataFrame(bootstrap_results)
    lower_p, upper_p = (100 - ci) / 2, 100 - (100 - ci) / 2

    ci_rows = []
    for metric, pt_val in point_estimates.items():
        if metric not in bootstrap_df.columns:
            continue
        vals = bootstrap_df[metric].dropna()
        if vals.empty:
            lo, hi, formatted = np.nan, np.nan, f"{pt_val:.3f} (NA-NA)"
        else:
            lo, hi = np.percentile(vals, lower_p), np.percentile(vals, upper_p)
            formatted = f"{pt_val:.3f} ({lo:.3f}–{hi:.3f})"
        ci_rows.append({"Metric": metric, "Point_Estimate": pt_val, "CI_Lower_95": lo,
                        "CI_Upper_95": hi, "Formatted_Result": formatted, "Valid_Replicates": len(vals)})
    return pd.DataFrame(ci_rows), bootstrap_df

# ==============================================================================
# SECTION 9 — DeLong's test 
# ==============================================================================

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(preds_sorted_by_class, m):
    n = preds_sorted_by_class.shape[1] - m
    positive_examples = preds_sorted_by_class[:, :m]
    negative_examples = preds_sorted_by_class[:, m:]
    k = preds_sorted_by_class.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_by_class[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_roc_test(y_true, proba_a, proba_b):
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    m = int(y_sorted.sum())

    preds = np.vstack([np.asarray(proba_a)[order], np.asarray(proba_b)[order]])
    aucs, cov = _fast_delong(preds, m)

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan, np.nan

    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)

# ==============================================================================
# SECTION 10 — Figure styling and interpretation helpers
# ==============================================================================

STYLE_PALETTE = {
    "primary": "#1f77b4", "secondary": "#ff7f0e", "neutral": "#7f7f7f",
    "danger": "#d62728", "success": "#2ca02c",
    "train": "#56B4E9", "val": "#E69F00", "test": "#009E73",
}

MODEL_COLORS = {
    "SVM": "#1f77b4", "LR": "#2ca02c", "RF": "#9467bd", "XGB": "#d62728", "MLP": "#ff7f0e",
}

def set_publication_style():
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.labelweight": "bold",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.title_fontsize": 9.5,
        "axes.grid": True, "axes.axisbelow": True, "axes.facecolor": "white",
        "figure.facecolor": "white", "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E5E5E5", "grid.linestyle": ":", "grid.linewidth": 0.8, "grid.alpha": 0.7,
        "lines.linewidth": 1.8, "lines.markersize": 5.5,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })

def _fmt(value, digits=3):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"

def _fmt_pct(value, digits=1):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100 * value:.{digits}f}%"

def save_interpretation(interpretation_text, interpretation_dict, filename_stem, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{filename_stem}.txt"
    json_path = output_dir / f"{filename_stem}.json"
    txt_path.write_text(interpretation_text, encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(interpretation_dict, f, indent=4, ensure_ascii=False, default=str)
    return txt_path, json_path

def add_interpretation_box(ax, text, x=0.03, y=0.97, fontsize=8.2):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize, wrap=True,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#A0A0A0", "alpha": 0.92})

set_publication_style()
print("Section 1-10 loaded: registry, shared metrics, tuning, calibration, bootstrap, DeLong, styling.")

# ==============================================================================
# SECTION 11 — Load and validate dataset 
# ==============================================================================

required_columns = all_features + [target, group_column, patient_id_col]

df = pd.read_excel(DATA_PATH)
print(f"Original dataset shape: {df.shape}")

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

df = df[required_columns].copy()

missing_patient_ids = int(df[patient_id_col].isna().sum())
if missing_patient_ids > 0:
    raise ValueError(f"Data Integrity Error: {missing_patient_ids} records have missing '{patient_id_col}'.")

df[patient_id_col] = df[patient_id_col].astype(str).str.strip()
empty_patient_ids = int(df[patient_id_col].isin(["", "nan", "None"]).sum())
if empty_patient_ids > 0:
    raise ValueError(f"Data Integrity Error: {empty_patient_ids} records have empty '{patient_id_col}' values.")

before_drop_shape = df.shape
df = df.dropna().reset_index(drop=True)
after_drop_shape = df.shape
print(f"Shape after column selection: {before_drop_shape}")
print(f"Shape after dropping missing values: {after_drop_shape}")

duplicate_patient_mask = df[patient_id_col].duplicated(keep=False)
if duplicate_patient_mask.any():
    raise ValueError(f"Error: Repeated Patient_Identifier values detected for "
                     f"{df.loc[duplicate_patient_mask, patient_id_col].nunique()} patients.")

print(f"✓ Patient_Identifier uniqueness confirmed: {df[patient_id_col].nunique()} unique identifiers for {len(df)} rows.")
print(f"ICUs available for LOIO: {df[group_column].nunique()} -> {list(df[group_column].unique())}")

data = df.loc[:, required_columns].copy()
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
if data.empty:
    raise ValueError("No records left after cleaning missing and infinite values.")

X = data.loc[:, all_features].copy()
y = data[target].astype(int).copy()
icu_groups = data[group_column].copy()
patient_groups = data[patient_id_col].copy()

metadata_columns = {target, group_column, patient_id_col}
leakage_columns = metadata_columns.intersection(X.columns)
if leakage_columns:
    raise ValueError(f"Data Leakage Risk: Found metadata/target columns inside X: {sorted(leakage_columns)}")

print("✓ Feature matrix (X) and target (y) are ready.")

# ==============================================================================
# SECTION 12 — Patient-level data split:  60/20/20
# ==============================================================================

if y.nunique() != 2:
    raise ValueError(f"Expected binary outcome (0 and 1), but found classes: {sorted(y.unique().tolist())}")
if y.value_counts().min() < 5:
    raise ValueError("Too few positive or negative cases to perform stratified splitting.")

record_indices = np.arange(len(y))
idx_dev, idx_test = train_test_split(record_indices, test_size=0.20, stratify=y, random_state=SEED)
idx_train, idx_val = train_test_split(idx_dev, test_size=0.25, stratify=y.iloc[idx_dev], random_state=SEED)

X_train, y_train = X.iloc[idx_train].copy(), y.iloc[idx_train].copy()
X_val, y_val = X.iloc[idx_val].copy(), y.iloc[idx_val].copy()
X_test, y_test = X.iloc[idx_test].copy(), y.iloc[idx_test].copy()

patient_train = patient_groups.iloc[idx_train].copy()
patient_val = patient_groups.iloc[idx_val].copy()
patient_test = patient_groups.iloc[idx_test].copy()

train_pts, val_pts, test_pts = set(patient_train), set(patient_val), set(patient_test)
if (train_pts & val_pts) or (train_pts & test_pts) or (val_pts & test_pts):
    raise ValueError("Patient Leakage: The same patient appears in more than one split!")

print("✓ Shared Train/Validation/Test split completed and leakage-checked once, for all five models.")

split_summary = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test", "Total"],
    "N": [len(y_train), len(y_val), len(y_test), len(y)],
    "CLABSI_Positive": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum()), int(y.sum())],
    "Event_Rate": [y_train.mean(), y_val.mean(), y_test.mean(), y.mean()],
})
split_summary.to_csv(COMPARISON_TABLE_DIR / "shared_data_split_summary.csv", index=False)

print("Section 11-12 loaded: data pipeline, shared split, and run_main_model() ready.")

# ==============================================================================
# SECTION 13 — Training Pipeline for models
# ==============================================================================

def run_main_model(model_name):
    dirs = MODEL_DIRS[model_name]
    fig_dir, table_dir, info_dir = dirs["main_fig"], dirs["main_table"], dirs["main_info"]
    print(f"\n{'=' * 70}\n[MAIN MODEL] {model_name}\n{'=' * 70}")

    pipeline_result = select_features_and_tune(
        model_name, X_train, y_train, categorical_features, numerical_features,
        PARAM_GRIDS[model_name], seed=SEED, top_k=TOP_K_FEATURES,
        cv_splits=CONTRACT_CONFIG["main_cv_splits"],
    )
    final_pipeline = pipeline_result["final_pipeline"]
    selected_features = pipeline_result["selected_features"]
    shap_importance = pipeline_result["shap_importance"]

    with open(info_dir / "top_features.json", "w") as f:
        json.dump(selected_features, f, indent=4)
    with open(info_dir / "final_best_parameters.json", "w") as f:
        json.dump(pipeline_result["final_best_params"], f, indent=4, default=str)
    print(f"Selected {len(selected_features)} features. Best params: {pipeline_result['final_best_params']}")

    # SHAP plot, calibration, metrics, bootstrap, confusion matrix, ROC/PR, calibration and risk-distribution,
    # split performance, learning curve, DCA, SHAP beeswarm, linear coefficients, risk stratification
   

    print(f"{model_name} main model completed with updated settings.")

    return {"model_name": model_name, "final_pipeline": final_pipeline, "calibrator": None,
            "selected_features": selected_features, "best_threshold": 0.5,
            "train_metrics": {}, "val_metrics": {}, "test_metrics": {}, "test_metrics_ci_df": pd.DataFrame(),
            "y_test_proba": np.array([]), "y_test_true": np.array([]), "risk_table": pd.DataFrame()}

# ==============================================================================
# SECTION 14 — Comparison between models on the held-out test set
# ==============================================================================

def run_all_main_models():
    results = {}
    for name in MODEL_NAMES:
        results[name] = run_main_model(name)
    return results

def build_main_comparison(main_results):
    rows = []
    for name, res in main_results.items():
        row = {"Model": name}
        for _, r in res["test_metrics_ci_df"].iterrows():
            row[r["Metric"]] = r["Formatted_Result"]
        rows.append(row)
    combined_metrics_df = pd.DataFrame(rows)
    combined_metrics_df.to_csv(COMPARISON_TABLE_DIR / "test_set_metrics_all_models.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for name, res in main_results.items():
        fpr, tpr, _ = roc_curve(res["y_test_true"], res["y_test_proba"])
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2.0,
                label=f"{name} (AUROC={res['test_metrics']['ROC_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=STYLE_PALETTE["neutral"], lw=1.2, label="Chance")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="1 - Specificity", ylabel="Sensitivity",
           title="Held-out Test Set: ROC Curves — All Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(COMPARISON_FIG_DIR / "roc_overlay_all_models.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    n_bins = CONTRACT_CONFIG["n_bins_calibration"]
    for name, res in main_results.items():
        y_t, y_p = res["y_test_true"], res["y_test_proba"]
        bins = pd.qcut(pd.Series(y_p).rank(method="first"), q=n_bins, labels=False, duplicates="drop")
        cal = (pd.DataFrame({"y_true": y_t, "y_proba": y_p, "bin": bins})
               .groupby("bin", observed=False).agg(mean_pred=("y_proba", "mean"), obs_rate=("y_true", "mean")))
        ax.plot(cal["mean_pred"], cal["obs_rate"], marker="o", lw=1.8, color=MODEL_COLORS[name], label=name)
    ax.plot([0, 1], [0, 1], "--", color=STYLE_PALETTE["neutral"], lw=1.2, label="Ideal")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean Predicted Probability", ylabel="Observed Event Rate",
           title="Held-out Test Set: Calibration — All Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(COMPARISON_FIG_DIR / "calibration_overlay_all_models.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    ref = main_results[REFERENCE_MODEL]
    delong_rows = []
    for name, res in main_results.items():
        if name == REFERENCE_MODEL:
            continue
        auc_ref, auc_other, z, p = delong_roc_test(ref["y_test_true"], ref["y_test_proba"], res["y_test_proba"])
        delong_rows.append({
            "Comparison": f"{REFERENCE_MODEL} vs {name}", "AUROC_SVM": auc_ref, f"AUROC_{name}": auc_other,
            "Z_statistic": z, "P_value": p, "Significant_at_0.05": (p < 0.05) if pd.notna(p) else False,
        })
    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(COMPARISON_TABLE_DIR / "delong_test_svm_vs_others_testset.csv", index=False)

    print("\n=== Held-out Test Set: DeLong AUROC comparisons vs SVM ===")
    print(delong_df.to_string(index=False))

    return combined_metrics_df, delong_df

# ==============================================================================
# SECTION 15 —  LOIO pipeline for models
# ==============================================================================

unique_icus = sorted(df[group_column].dropna().unique().tolist())
n_icus = len(unique_icus)
LOIO_SCORING = "roc_auc"
LOIO_MAX_INNER_SPLITS = CONTRACT_CONFIG["main_cv_splits"]

loio_fold_splits = {}
for test_icu in unique_icus:
    df_dev = df[df[group_column] != test_icu].copy()
    y_dev_full = df_dev[target].astype(int).to_numpy()
    df_dev_train, df_dev_val, y_dev_train, y_dev_val = train_test_split(
        df_dev, y_dev_full, test_size=0.25, stratify=y_dev_full, random_state=SEED,
    )
    loio_fold_splits[test_icu] = (df_dev_train, df_dev_val, y_dev_train, y_dev_val)

print(f"✓ Precomputed {n_icus} LOIO dev-train/dev-val splits, shared across all five models.")

def run_loio(model_name):
    dirs = MODEL_DIRS[model_name]
    fig_dir, table_dir, info_dir = dirs["loio_fig"], dirs["loio_table"], dirs["loio_info"]
    print(f"\n{'=' * 70}\n[LOIO] {model_name} — {n_icus} ICUs\n{'=' * 70}")

    loio_fold_results = []
    loio_patient_predictions = []

    for fold_idx, test_icu in enumerate(unique_icus, start=1):
        df_dev_train, df_dev_val, y_dev_train, y_dev_val = loio_fold_splits[test_icu]
        df_test = df[df[group_column] == test_icu].copy()
        y_test_icu = df_test[target].astype(int).to_numpy()
        test_n, test_events = len(df_test), int(y_test_icu.sum())

        fold_result = select_features_and_tune(
            model_name, df_dev_train, y_dev_train, categorical_features, numerical_features,
            PARAM_GRIDS[model_name], seed=SEED, top_k=TOP_K_FEATURES, cv_splits=LOIO_MAX_INNER_SPLITS,
        )
        fold_pipeline = fold_result["final_pipeline"]
        selected_cols = fold_result["selected_features"]

        fold_calibrator, best_threshold_fold, _ = calibrate_and_tune_threshold(
            model_name, fold_pipeline, df_dev_val, y_dev_val, selected_cols, method=CALIBRATION_METHOD
        )
        test_probs = apply_calibrator(model_name, fold_pipeline, fold_calibrator, df_test, selected_cols, method=CALIBRATION_METHOD)
        test_preds = (test_probs >= best_threshold_fold).astype(int)

        for orig_idx, y_t, p_val, y_p in zip(df_test.index, y_test_icu, test_probs, test_preds):
            loio_patient_predictions.append({
                "Left_Out_ICU": test_icu, "Fold": fold_idx, "y_true": int(y_t),
                "y_proba": float(p_val), "y_pred": int(y_p),
            })

        fold_metrics = calculate_all_metrics(y_test_icu, test_probs, best_threshold_fold)
        tn, fp, fn, tp = confusion_matrix(y_test_icu, test_preds, labels=[0, 1]).ravel()
        loio_fold_results.append({
            "Left_Out_ICU": test_icu, "Fold": fold_idx, "Test_N": test_n, "Test_Events": test_events,
            "Optimal_Threshold": round(best_threshold_fold, 4),
            **{k: fold_metrics[k] for k in ["Accuracy", "Precision_PPV", "Recall_Sensitivity", "Specificity", "NPV", "F1_score", "ROC_AUC", "PR_AUC", "Brier"]},
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
        print(f"  [{fold_idx}/{n_icus}] {test_icu}: AUROC={fold_metrics['ROC_AUC']:.3f} (n={test_n}, events={test_events})")

    loio_results_df = pd.DataFrame(loio_fold_results)
    loio_predictions_df = pd.DataFrame(loio_patient_predictions)

    pooled_y_true = loio_predictions_df["y_true"].astype(int).to_numpy()
    pooled_y_proba = loio_predictions_df["y_proba"].astype(float).to_numpy()
    pooled_y_pred = loio_predictions_df["y_pred"].astype(int).to_numpy()
    pooled_auc = roc_auc_score(pooled_y_true, pooled_y_proba)
    pooled_prauc = average_precision_score(pooled_y_true, pooled_y_proba)
    pooled_brier = brier_score_loss(pooled_y_true, pooled_y_proba)

    macro_metric_cols = ["Accuracy", "Precision_PPV", "Recall_Sensitivity", "Specificity", "NPV", "F1_score", "ROC_AUC", "PR_AUC", "Brier"]
    macro_rows = []
    for metric in macro_metric_cols:
        values = pd.to_numeric(loio_results_df[metric], errors="coerce").dropna().to_numpy()
        n_valid = len(values)
        if n_valid > 1:
            mean_value = float(values.mean())
            sd_value = float(values.std(ddof=1))
            se_value = sd_value / np.sqrt(n_valid)
            t_crit = stats.t.ppf(0.975, df=n_valid - 1)
            ci_low, ci_high = mean_value - t_crit * se_value, mean_value + t_crit * se_value
        else:
            mean_value = float(values[0]) if n_valid == 1 else np.nan
            ci_low = ci_high = np.nan
        macro_rows.append({"Metric": metric, "Macro_Mean": mean_value, "N_Valid_ICUs": n_valid,
                            "95%_CI_Lower": ci_low, "95%_CI_Upper": ci_high})
    macro_summary_df = pd.DataFrame(macro_rows)

    loio_results_df.to_csv(table_dir / "loio_per_icu_metrics.csv", index=False)
    loio_predictions_df.to_csv(table_dir / "loio_patient_level_predictions.csv", index=False)
    macro_summary_df.to_csv(table_dir / "loio_macro_summary_95ci.csv", index=False)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13.5, 6.0))
    for icu in unique_icus:
        sub = loio_predictions_df[loio_predictions_df["Left_Out_ICU"] == icu]
        if sub["y_true"].nunique() == 2:
            fpr, tpr, _ = roc_curve(sub["y_true"], sub["y_proba"])
            ax_roc.plot(fpr, tpr, color=STYLE_PALETTE["neutral"], alpha=0.35, linewidth=1.0)
    pooled_fpr, pooled_tpr, _ = roc_curve(pooled_y_true, pooled_y_proba)
    ax_roc.plot(pooled_fpr, pooled_tpr, color=STYLE_PALETTE["primary"], linewidth=2.4, label=f"Pooled (AUROC={pooled_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], ":", color="#555555", linewidth=1.1, label="Chance")
    ax_roc.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="1 - Specificity", ylabel="Sensitivity",
               title=f"{model_name}: LOIO ROC Curves")
    ax_roc.legend(loc="lower right")

    for icu in unique_icus:
        sub = loio_predictions_df[loio_predictions_df["Left_Out_ICU"] == icu]
        if sub["y_true"].nunique() == 2:
            prec, rec, _ = precision_recall_curve(sub["y_true"], sub["y_proba"])
            ax_pr.plot(rec, prec, color=STYLE_PALETTE["neutral"], alpha=0.35, linewidth=1.0)
    pooled_prec, pooled_rec, _ = precision_recall_curve(pooled_y_true, pooled_y_proba)
    ax_pr.plot(pooled_rec, pooled_prec, color=STYLE_PALETTE["test"], linewidth=2.4, label=f"Pooled (AUPRC={pooled_prauc:.3f})")
    ax_pr.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Recall", ylabel="Precision",
              title=f"{model_name}: LOIO PR Curves")
    ax_pr.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "loio_pooled_and_icu_roc_pr.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    n_bins = CONTRACT_CONFIG["n_bins_calibration"]
    q = pd.qcut(pd.Series(pooled_y_proba).rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    loio_cal_df = (pd.DataFrame({"y_true": pooled_y_true, "y_proba": pooled_y_proba, "bin": q})
                   .groupby("bin", observed=False).agg(mean_pred=("y_proba", "mean"), obs_rate=("y_true", "mean")).reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(loio_cal_df["mean_pred"], loio_cal_df["obs_rate"], marker="o", lw=2, color=STYLE_PALETTE["primary"], label="Pooled LOIO")
    ax.plot([0, 1], [0, 1], "--", color=STYLE_PALETTE["neutral"], lw=1.2, label="Ideal")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean Predicted Probability", ylabel="Observed Event Rate",
           title=f"{model_name}: LOIO Pooled Calibration")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "loio_pooled_calibration.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    valid_icu_df = loio_results_df.dropna(subset=["ROC_AUC"]).sort_values("ROC_AUC").reset_index(drop=True)
    macro_auc_row = macro_summary_df.loc[macro_summary_df["Metric"] == "ROC_AUC"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, max(5.0, len(valid_icu_df) * 0.45 + 2.0)))
    y_pos = np.arange(len(valid_icu_df))
    ax.scatter(valid_icu_df["ROC_AUC"], y_pos, color=STYLE_PALETTE["primary"], s=65, zorder=3, label="Left-out ICU AUC")
    for idx, row in valid_icu_df.iterrows():
        ax.text(0.02, idx, f"{row['Left_Out_ICU']} (N={int(row['Test_N'])}, Events={int(row['Test_Events'])})", va="center", fontsize=8.5)
    summary_y = len(valid_icu_df) + 0.6
    ax.errorbar(macro_auc_row["Macro_Mean"], summary_y,
                xerr=[[macro_auc_row["Macro_Mean"] - macro_auc_row["95%_CI_Lower"]], [macro_auc_row["95%_CI_Upper"] - macro_auc_row["Macro_Mean"]]],
                fmt="D", color=STYLE_PALETTE["secondary"], markersize=7, capsize=5, zorder=4, label="Macro Average (95% CI)")
    ax.axvline(0.5, linestyle=":", color="#777777", linewidth=1.0)
    ax.set(yticks=[], xlabel="AUROC", title=f"{model_name}: LOIO Discrimination Across ICUs", xlim=(0.0, 1.05))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "loio_icu_auc_forest_plot.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"{model_name} pooled LOIO AUROC: {pooled_auc:.3f} | macro AUROC: {macro_auc_row['Macro_Mean']:.3f}")

    return {
        "model_name": model_name, "loio_results_df": loio_results_df, "loio_predictions_df": loio_predictions_df,
        "macro_summary_df": macro_summary_df, "pooled_y_true": pooled_y_true, "pooled_y_proba": pooled_y_proba,
        "pooled_auroc": pooled_auc, "pooled_auprc": pooled_prauc, "pooled_brier": pooled_brier,
    }

def run_all_loio():
    return {name: run_loio(name) for name in MODEL_NAMES}

# ==============================================================================
# SECTION 16 —  LOIO comparison between models
# ==============================================================================

def build_loio_comparison(loio_results):
    rows = []
    for name, res in loio_results.items():
        macro_auc = res["macro_summary_df"].loc[res["macro_summary_df"]["Metric"] == "ROC_AUC"].iloc[0]
        rows.append({
            "Model": name, "Pooled_AUROC": res["pooled_auroc"], "Pooled_AUPRC": res["pooled_auprc"],
            "Pooled_Brier": res["pooled_brier"], "Macro_AUROC_Mean": macro_auc["Macro_Mean"],
            "Macro_AUROC_CI_Lower": macro_auc["95%_CI_Lower"], "Macro_AUROC_CI_Upper": macro_auc["95%_CI_Upper"],
        })
    loio_comparison_df = pd.DataFrame(rows)
    loio_comparison_df.to_csv(COMPARISON_TABLE_DIR / "loio_pooled_macro_auroc_all_models.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(loio_comparison_df))
    ax.errorbar(
        loio_comparison_df["Macro_AUROC_Mean"], y_pos,
        xerr=[loio_comparison_df["Macro_AUROC_Mean"] - loio_comparison_df["Macro_AUROC_CI_Lower"],
              loio_comparison_df["Macro_AUROC_CI_Upper"] - loio_comparison_df["Macro_AUROC_Mean"]],
        fmt="o", capsize=5, color=STYLE_PALETTE["primary"],
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(loio_comparison_df["Model"])
    ax.axvline(0.5, linestyle=":", color="#777777")
    ax.set(xlabel="Macro-average LOIO AUROC (95% CI)", title="LOIO Discrimination — All Models")
    fig.tight_layout()
    fig.savefig(COMPARISON_FIG_DIR / "loio_macro_auroc_forest_all_models.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    ref = loio_results[REFERENCE_MODEL]
    delong_rows = []
    for name, res in loio_results.items():
        if name == REFERENCE_MODEL:
            continue
        auc_ref, auc_other, z, p = delong_roc_test(ref["pooled_y_true"], ref["pooled_y_proba"], res["pooled_y_proba"])
        delong_rows.append({
            "Comparison": f"{REFERENCE_MODEL} vs {name}", "Pooled_AUROC_SVM": auc_ref,
            f"Pooled_AUROC_{name}": auc_other, "Z_statistic": z, "P_value": p,
            "Significant_at_0.05": (p < 0.05) if pd.notna(p) else False,
        })
    delong_loio_df = pd.DataFrame(delong_rows)
    delong_loio_df.to_csv(COMPARISON_TABLE_DIR / "delong_test_svm_vs_others_loio.csv", index=False)

    print("\n=== Pooled LOIO: DeLong AUROC comparisons vs SVM ===")
    print(delong_loio_df.to_string(index=False))

    return loio_comparison_df, delong_loio_df

# ==============================================================================
# SECTION 17 — Master execution
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70 + "\n# PHASE 1: MAIN MODEL (shared 60/20/20 split) FOR ALL MODELS\n" + "#" * 70)
    main_results = run_all_main_models()
    main_comparison_df, main_delong_df = build_main_comparison(main_results)

    print("\n" + "#" * 70 + "\n# PHASE 2: LEAVE-ONE-ICU-OUT (LOIO) FOR ALL MODELS\n" + "#" * 70)
    loio_results = run_all_loio()
    loio_comparison_df, loio_delong_df = build_loio_comparison(loio_results)

    print("\n" + "#" * 70 + "\n# BENCHMARK COMPLETE\n" + "#" * 70)
    print(f"All outputs written under: {OUTPUT_DIR.resolve()}")
    print("\nHeld-out test set metrics (all models):")
    print(main_comparison_df.to_string(index=False))
    print("\nLOIO pooled/macro AUROC (all models):")
    print(loio_comparison_df.round(4).to_string(index=False))