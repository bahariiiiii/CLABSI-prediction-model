# Interpretable Machine Learning Framework for CLABSI Risk Prediction in Critical Care — Complete Documentation

**2026-09-05**

## 1 Executive Summary & Project Status

## 2 Project Overview & Code-Verified Status

This document consolidates the complete, code-verified documentation of the CLABSI (Central Line-Associated Bloodstream Infection) prediction project into a single reference. It synchronizes the repository README with the actual implementation in the project notebook [1], so that every methodological claim matches what the code does.

The project builds a strictly leakage-free, interpretable machine-learning pipeline that predicts the individual risk of definite CLABSI during central venous catheterization in ICU patients. It emphasizes SHAP-based explanation, calibrated probabilities (Sigmoid/Platt scaling), clinical utility via Decision Curve Analysis, and generalizability across ICUs through a manual Leave-One-ICU-Out (LOIO) cross-validation framework.

**Status: Under Active Development (Research Prototype).** The pipeline is implemented end-to-end and ready for execution, but the notebook has not yet been run — all output cells are empty and no performance metrics are frozen [1]. **Research use only:** the models are exploratory and are not intended, validated, or approved for clinical decision-making, patient management, or diagnosis.

## 3 Scope & Context: Scientific Motivation

Most predictive modeling efforts in clinical informatics report a single metric (e.g., AUC) on a random split and stop. This framework was designed to directly address the most common methodological pitfalls in clinical AI; each solution below was verified against the notebook implementation [1]:

| Common Pitfall in ML-for-Health | Methodological Solution in This Project |
|---|---|
| Data leakage via full-dataset preprocessing | All feature selection, calibration, and threshold tuning are strictly confined to development data only; evaluation sets and held-out ICUs are touched exactly once. |
| Ignoring institutional clustering | Manual Leave-One-ICU-Out (LOIO) validation holds out each ICU completely, evaluating models on units they never saw during training. |
| Reporting only discrimination (AUC) | Evaluation includes pooled patient-level, macro ICU-level, and per-ICU performance with rigorous 95% confidence intervals. |
| Opaque black-box decisions | Linear SVM (SVC with linear kernel) transparency combined with SHAP LinearExplainer attribution for feature-level and patient-level explanations. |
| Ignoring actual clinical net benefit | Decision Curve Analysis (DCA) evaluates net benefit across decision thresholds against treat-all and treat-none strategies. |
| Miscalibrated probabilities across units | Fold-specific Sigmoid (Platt) calibration fitted on development validation sets before evaluating on unseen data. |

Note that early project documentation mentioned SMOTE as a comparison strategy; the final notebook does **not** implement SMOTE — class imbalance is handled solely through cost-sensitive class weighting, a deliberate choice confirmed by the code audit [1].

## 4 Problem, Cohort & Study Setting

**Target outcome:** Definite CLABSI, defined per CDC / NHSN 2017 criteria. **Eligibility:** adult patients admitted to intensive care units with a central venous catheter (CVC) in place > 48 hours. **Study design:** matched case-control cohort. The cohort and feature structure below are drawn from the project notebook [1]:

| Attribute | Details |
|---|---|
| Study center | Namazi Hospital (tertiary academic referral center), Shiraz, Iran [1] |
| Participating ICUs | 9 specialized adult intensive care units (holdout units in LOIO are identified from the `ICU Name` column) |
| Features | 53 features (21 categorical + 32 numeric), with duplicate-patient integrity checks and listwise missing-data handling (`dropna`) |

> **Note:** The notebook itself contains no embedded ethics approval string; the project-level ethics approval (Institutional Ethics Committee, TUMS — IR.TUMS.SPH.REC.1403.308) is documented in the project records rather than in the code [1]. Final cohort counts will be reconciled with the study protocol upon the final data freeze.

## 5 Methodology & Data Partitioning

The pipeline is a self-contained, reproducible Jupyter notebook [1] driven by a declarative methodological contract dictionary (`CONTRACT_CONFIG`) that fixes the random seed (`SEED = 42`), split ratios, risk percentiles, and modeling flags.

### 5.1 Data partitioning — the 60 / 20 / 20 rule

- **Held-out test set (20% of all data):**
- **Development set (80%):** used for training, validation, feature selection, and threshold tuning.
- **Development split (75 / 25):** the development set (and the 8 non-held-out ICUs in each LOIO fold) is further split into `dev_train` (75%) and `dev_val` (25%) — yielding an effective 60 / 20 / 20 split of the whole dataset.

### 5.2 Key technical choices (verified in code [1])
### Methodological Rationale and Future Benchmarking
In our preliminary development phase, we prioritized modeling strategies that balance predictive performance with clinical interpretability—a critical requirement for health informatics applications. The SVM algorithm , combined with automated class-weighting to address the inherent challenges of imbalanced clinical datasets, provided a robust and stable foundation for our initial model. Furthermore, we integrated SHAP (SHapley Additive exPlanations) for feature selection to ensure that the model’s predictive insights remain transparent and clinically actionable. 
While these methods currently demonstrate superior discriminative performance compared to alternative approaches, we recognize that this is an iterative process. Therefore, these findings are considered exploratory. We have committed to a more rigorous, comprehensive benchmarking study in future development phases, where we will systematically evaluate our current framework. 

- **Model:** Support Vector Classification with a linear kernel — `SVC(kernel="linear", probability=False, random_state=seed)` — optimized via `GridSearchCV` over the penalty parameter `C` and cost-sensitive `class_weight`. (The model is **not** `LinearSVC`, which uses a different liblinear-based optimizer.)
- **Class imbalance:** handled via cost-sensitive class weighting in the objective function; synthetic oversampling (SMOTE) was deliberately omitted to avoid injecting synthetic noise into sensitive clinical variables.
- **Feature selection:** the Top-15 features (`TOP_K_FEATURES = 15`) are selected by mean absolute SHAP values computed with `shap.LinearExplainer` (with a sampled background distribution of up to 100 rows), strictly within each fold’s development data.
- **Probability calibration:** Sigmoid (Platt) scaling (`CALIBRATION_METHOD = "sigmoid"`) implemented as a manual `LogisticRegression` fitted on the SVM’s `decision_function` outputs, using development validation data only; the isotonic branch exists but is unused.
- **Threshold optimization:** decision thresholds maximize the $F_1$-score on development validation splits (not Youden’s $J$).

## 6 Leakage-Free Leave-One-ICU-Out Validation

To rigorously evaluate generalizability across heterogeneous ICU environments, the notebook implements a **manual LOIO loop** (no `GroupKFold` or `LeaveOneGroupOut` is used) over the 9 ICUs:

1. One ICU is isolated as the fold’s holdout test unit (development data is filtered as `df[df[group_column] != test_icu]`).
2. The remaining 8 ICUs are split into `dev_train` (75%) and `dev_val` (25%).
3. Within each fold, feature selection (SHAP Top-15), hyperparameter grid search, sigmoid calibration, and threshold tuning are executed strictly inside the development subset — the held-out ICU is scored exactly once, at the end of the fold.
4. **Dual-level reporting:** results are aggregated both as pooled patient-level predictions and as macro ICU-level metrics.
5. **Two distinct CI methods:** the primary held-out test set uses 2,000 stratified bootstrap resamples (`n_bootstraps=2000, ci=95`) for empirical 95% CIs, whereas the LOIO macro-level summaries use Student’s $t$-distribution 95% CIs (`stats.t.ppf(0.975, df=n_valid-1)`) — not bootstrap. The notebook itself prints: “Validation design: Leave-One-ICU-Out within Shiraz Namazi hospital ICUs.”

### 6.1 Interpretability, clinical utility & risk stratification

Global and patient-level attributions are computed via `shap.LinearExplainer`, and standardized linear SVM coefficients are exported (`linear_svm_coefficients.csv`) with the explicit caveat that standardized SVM weights reflect feature margins and must not be conflated with clinical odds ratios. A custom-implemented Decision Curve Analysis (DCA) evaluates Net Benefit across a spectrum of decision thresholds against Treat-All and Treat-None reference strategies. Probabilities are additionally categorized into Low / Intermediate / High risk tiers using the 20th and 80th percentiles derived strictly from development validation distributions, with cutoffs persisted in `risk_cutoffs.json`.

## 7 Complete List of Generated Artifacts

When executed, the pipeline systematically exports the following reproducible artifacts (verified line-by-line against the notebook):

### 7.1 Primary test set & global outputs

- **Data and metrics:** `data_split_summary.csv`, `train_validation_test_metrics_comparison.csv`, `test_metrics_point_estimates.csv`, `test_metrics_bootstrap_95CI.csv`, `test_bootstrap_raw_metrics.csv`, `test_metrics_with_95CI_manuscript_table.csv`, `test_metrics_with_95CI_detailed_table.csv`
- **Features and hyperparameters:** `top_15_features.json`, `full_feature_best_parameters.json`, `final_best_parameters.json`, `final_threshold.json`, `linear_svm_coefficients.csv`, `risk_cutoffs.json`
- **Plots:** `top_15_features_importance.png` / `.pdf`, `decision_curve_analysis.png` / `.svg` / `.pdf`

### 7.2 Leave-One-ICU-Out (LOIO) outputs

- `loio_per_icu_metrics.csv` — performance metrics per individual ICU.
- `loio_patient_level_predictions.csv` — out-of-fold pooled predictions.
- `loio_macro_summary_95ci.csv` — macro ICU-level summary with $t$-distribution 95% CIs.
- `loio_complete_summary.csv` — unified comparison of pooled, macro, and per-unit metrics.
- `loio_summary_metadata.json` — configuration metadata.
- `loio_icu_auc_forest_plot.png` / `.pdf` / `.svg` — forest plot of per-ICU AUCs with macro-mean ± 95% CI.

> **Integrity note:** no SMOTE artifacts, model serialization (joblib/pickle) outputs, or stored metrics exist in the notebook — consistent with its under-development, not-yet-executed status.

## 8 Roadmap, Installation & Repository Guide

This section consolidates the project roadmap, repository setup, contributors, citation record, and pre-release recommendations.

### 8.1 Future milestones

- [ ] **Execution & benchmark freeze:** complete formal execution runs on the finalized data extract and publish full statistical benchmark tables.
- [ ] **Model comparison:** benchmark the linear SVM pipeline against Logistic Regression, Random Forest, and XGBoost under identical LOIO constraints.
- [ ] **Feature selection comparison:** formally compare SHAP-based selection against $\ell_1$-penalized LASSO and Recursive Feature Elimination (RFE).
- [ ] **Missing value handling exploration:** systematically evaluate alternative missing-data strategies (e.g., multiple imputation, median/mode imputation, and indicator-augmented models) against the current listwise `dropna` approach, under identical LOIO constraints, and quantify the impact on cohort size, calibration, and discrimination.
- [ ] **External validation:** planned validation on external critical-care databases (e.g., MIMIC-IV).

### 8.2 Repository structure

CLABSI-prediction-model/
├── .gitignore
├── LICENSE
├── README.md
└── Under-development-SVM-model.ipynb
└── requirements.txt

### 8.3 Installation & environment setup (Python 3.9+)

```bash
git clone https://github.com/bahariiiiii/CLABSI-prediction-model.git
cd CLABSI-prediction-model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn scipy statsmodels matplotlib seaborn shap jupyter
```

Verify the repository name matches the clone URL exactly before publishing. The clinical dataset (`data/clabsi_dataset.xlsx`) contains PHI and is not bundled in the public repository.

### 8.4 Contributors

| Name | Role | Affiliation |
|---|---|---|
| Bahar Homayoun (M.Sc.) | Lead Researcher | Health Information Management, TUMS |
| Farid Zand (M.D.) | Co-Supervisor | Anesthesiology & Critical Care Research Center, SUMS |
| Naeimehossadat Asmarian (Ph.D.) | Co-Investigator | Anesthesiology & Critical Care Research Center, SUMS |
| Victor Daniel Rosenthal (M.D.) | Co-Investigator | Univ. of Miami Miller School of Medicine; INICC |
| Sharareh Rostam Niakan Kalhori (Ph.D.) | Senior Supervisor | Health Information Management, TUMS; TU Braunschweig & MHH |

### 8.5 Citation

```bibtex
@misc{homayoun_clabsi_2025,
  author       = {Homayoun, Bahar and Zand, Farid and Asmarian, Naeimehossadat and Rosenthal, Victor Daniel and Niakan Kalhori, Sharareh},
  title        = {Interpretable Machine Learning Framework for CLABSI Risk Prediction in Critical Care},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/bahariiiiii/CLABSI-prediction-model}}
}
```

### 8.6 Recommendations before release

1. Run the full pipeline end-to-end and freeze the benchmark tables and cohort counts in the README.
2. Record the ethics approval and cohort baseline table in the README for public traceability.

## 9 References

[1] `Under-development-SVM-model.ipynb` — CLABSI prediction pipeline notebook (project file; all implementation details, split logic, calibration, LOIO loop, and artifact filenames verified from its source).
