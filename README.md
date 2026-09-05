Interpretable Machine Learning Framework for CLABSI Risk Prediction in Critical Care
2026-09-05
Table of Contents
1 ⚠️ Research Prototype — Important Notice	
2 Overview	
3 🏥 Clinical Cohort & Setting	
3.1 Data Availability Statement	
4 🛠️ Methodology & Pipeline Architecture	
4.1 Model & hyperparameter tuning	
4.2 Feature selection & explainability	
4.3 Probability calibration & threshold tuning	
4.4 Class imbalance handling	
4.5 Three-tier risk stratification	
5 🔒 Leakage-Free Validation: Leave-One-ICU-Out (LOIO)	
5.1 Pooled evaluation & heterogeneity	
6 📊 Uncertainty Quantification & Clinical Utility	
6.1 Decision Curve Analysis (DCA)	
6.2 Reproducible artifact trail	
7 🚀 Roadmap & Future Directions	
7.1 1. Algorithmic benchmarking	
7.2 2. Advanced pipeline optimization	
7.3 3. External validation — international collaboration	
8 🧬 Contributors & Collaborators	
9 📂 Repository Structure, Installation & Usage	
9.1 Installation & usage	
9.2 Key output artifacts	
9.3 License	
9.4 Acknowledgment	
1 ⚠️ Research Prototype — Important Notice
[!IMPORTANT] RESEARCH PROTOTYPE ONLY. This repository is under active development. The models and pipelines presented here are for research purposes only and are not intended for clinical decision-making, diagnosis, or patient care. Final performance metrics and benchmark comparisons are undergoing evaluation and will be published upon study completion.
Status: Under active development · Target cohort: Adult ICUs, Namazi Hospital, Shiraz
Current focus
•	Refining the end-to-end pipeline to guarantee absolute data integrity (zero-leakage).
•	Optimizing preprocessing: missing-value handling and class-imbalance management.
•	Fine-tuning model components: feature selection, uncertainty estimation, decision-threshold tuning, and probability calibration.
No final numeric performance figures are reported in this README by design: the persisted notebook ships without executed outputs, and we prefer verified results over provisional claims.
2 Overview
This repository hosts an interpretable machine-learning framework for predicting Central Line-Associated Bloodstream Infection (CLABSI) in critically ill adults. The framework is built around three commitments that are often missing in clinical prediction tools:
1.	Strict leakage-free evaluation. Every data-dependent step — feature selection, hyperparameter tuning, probability calibration, and threshold tuning — is executed exclusively within development folds, so that held-out units are never touched until final evaluation.
2.	Unit-level transportability. Validation follows a Leave-One-ICU-Out (LOIO) design, explicitly testing whether the model generalizes across specialized intensive-care units rather than across random patient splits.
3.	Clinical interpretability and utility. Model explanations (SHAP, linear coefficients) are paired with decision-curve analysis and a three-tier risk stratification scheme, so that outputs translate into actionable risk levels rather than opaque scores.
The implementation is fully reproducible: a CONTRACT_CONFIG block pins all hyperparameters, seeds, and methodological choices, and every analytical artifact (tables, figures, and their plain-text/JSON interpretations) is exported automatically.
3 🏥 Clinical Cohort & Setting
•	Study location: Namazi Hospital, a tertiary academic medical center in Shiraz, Iran.
•	Participating units: 9 specialized adult intensive care units.
•	Cohort: N = 5,200 matched patients — 295 CLABSI cases matched 1:2 with controls.
•	Inclusion criteria: Adult patients with central venous catheter (CVC) duration > 48 hours.
•	Outcome definition: Definite CLABSI according to the CDC/NHSN 2017 criteria.
•	Ethics: Approved by the Institutional Ethics Committee of Tehran University of Medical Sciences (IR.TUMS.SPH.REC.1403.308).
3.1 Data Availability Statement
The clinical datasets contain sensitive Protected Health Information (PHI) and are governed by institutional data-use agreements; raw patient data cannot be shared publicly. Researchers interested in collaboration or reproducibility validation are invited to contact the lead author to discuss data-access protocols and the required institutional approvals.
4 🛠️ Methodology & Pipeline Architecture
The pipeline is implemented as a deterministic, configuration-driven workflow in Under-development-SVM-model.ipynb.
4.1 Model & hyperparameter tuning
The primary estimator is a linear-kernel Support Vector Machine (SVC(kernel="linear"), scored via the decision_function). Hyperparameters are optimized with GridSearchCV under stratified cross-validation, sweeping a regularization grid of C ∈ {0.01, 0.1, 1, 5} against five class-weighting configurations (including None, "balanced", and event-weight variants up to {0:1, 1:3}), with AUROC as the scoring criterion.
4.2 Feature selection & explainability
•	SHAP-based selection: shap.LinearExplainer is applied over a background sample of the processed data; mean absolute SHAP values are aggregated back to original clinical variables, and the top-15 features (TOP_K_FEATURES = 15) are retained. The model is then re-tuned on the selected subset.
•	Dual explainability: Beyond global/local SHAP analyses, the linear coefficients are extracted and mapped to original variables (linear_svm_coefficients.csv), with an explicit caveat that coefficients of scaled features are not directly interpretable as clinical odds ratios.
4.3 Probability calibration & threshold tuning
•	Platt scaling (sigmoid calibration): A logistic calibrator is fit on each fold’s development-validation decision scores and applied unchanged to held-out data. Calibration is therefore fold-specific, never global — one of the strongest guarantees against optimistic drift.
•	F1-maximizing threshold: The operating threshold is chosen on the development-validation set by maximizing the F1 score over the precision–recall curve; Youden’s J is deliberately not used.
4.4 Class imbalance handling
Class imbalance is addressed through cost-sensitive learning (class weighting) rather than synthetic oversampling. SMOTE is not used in the final pipeline: injecting synthetic points into sensitive clinical variables risks introducing artifacts, whereas class weighting preserves the empirical data distribution. In preliminary evaluations, class weighting yielded superior discriminative performance compared with resampling; a formal comparative benchmark is planned (see Roadmap).
4.5 Three-tier risk stratification
For clinical translation, predicted probabilities are binned into Low / Intermediate / High risk using cutoffs at the 20th and 80th percentiles of the validation probability distribution. Cutoffs are derived exclusively from validation data (risk_cutoffs.json, source: “Validation set only (leakage-free)”).
5 🔒 Leakage-Free Validation: Leave-One-ICU-Out (LOIO)
The central validation strategy is Leave-One-ICU-Out (LOIO): each of the 9 ICUs is iteratively held out as a complete test unit, and the model is retrained from scratch on the remaining units. This directly evaluates transportability across different specialized units within the same institution — a far more demanding test than random patient-level splits.
Inside every fold, the following order is strictly enforced:
1.	One ICU is removed entirely.
2.	Remaining data are split (stratified) into development-train and development-validation.
3.	SHAP feature selection, GridSearchCV tuning, sigmoid calibration, and F1-threshold tuning are performed only on development data.
4.	The held-out ICU is scored exactly once, using the fold-specific calibrator and threshold.
5.	Per-ICU predictions and metrics are saved; the held-out ICU contributes nothing to any upstream decision.
5.1 Pooled evaluation & heterogeneity
Fold-level predictions are pooled into patient-level outputs (loio_patient_level_predictions.csv) and macro-averaged across ICUs (loio_macro_summary_95ci.csv).
A dedicated forest plot (loio_icu_auc_forest_plot.{png,pdf,svg}, 600 dpi) displays each ICU’s AUC alongside its patient and event counts, the macro-average with 95% CI, and the pooled AUC against the 0.5 chance line — making between-ICU heterogeneity (case mix, event frequency, local care practices) transparent rather than hidden behind a single summary number.
Transportability of probabilities is also assessed: pooled calibration curves, the Brier score, and the mean absolute calibration gap quantify whether calibrated predictions remain reliable when the model moves to an ICU it never saw during training.
6 📊 Uncertainty Quantification & Clinical Utility
The framework reports uncertainty at two statistically distinct levels, matching the two evaluation designs.
•	Primary held-out evaluation: 95% confidence intervals for all metrics are estimated with 2,000 stratified bootstrap resamples (positive and negative classes resampled independently; percentile CIs).
•	LOIO macro-averages: ICU-level confidence intervals use t-based intervals (mean ± t₀.₉₇₅ · sd/√n) across the held-out ICUs, respecting the small number of folds rather than resampling patient rows.
6.1 Decision Curve Analysis (DCA)
A custom DCA implementation computes the net benefit of acting on the model across threshold probabilities (0.01–0.99), benchmarked against the Treat-All and Treat-None default strategies. Net benefit is calculated as TP/n − (FP/n)·pt/(1−pt); curves are exported as decision_curve_analysis.{png,svg,pdf} with accompanying machine-readable interpretations. DCA answers the question that AUROC cannot: at which probability thresholds does using this model yield more clinical benefit than treating everyone or no one?
6.2 Reproducible artifact trail
Every analytical step exports its outputs: per-ICU and pooled metric tables, patient-level predictions, bootstrap CI tables (loio_test_bootstrap_95ci.csv), calibration-deviation tables, selected-feature and best-parameter JSON files, and figures (ROC, precision–recall, calibration, confusion matrix, SHAP summary, learning curve) — each accompanied by *_interpretation.txt / .json files documenting what the figure shows.
7 🚀 Roadmap & Future Directions
The roadmap is organized around three priorities that build on the current leakage-free foundation.
7.1 1. Algorithmic benchmarking
In preliminary comparative experiments, the Linear SVM demonstrated superior overall discriminative capacity with minimal overfitting risk compared with more complex architectures on this cohort. Upcoming updates will benchmark against standard algorithms:
•	Logistic Regression (LR)
•	Random Forest (RF)
•	XGBoost
A controlled class-weighting vs. SMOTE comparison will also be documented as a standalone experiment, distinct from the LOIO evaluation.
7.2 2. Advanced pipeline optimization
•	Missing data strategy: evaluating principled probabilistic imputation frameworks (e.g., multiple imputation / MICE) versus current deterministic approaches.
•	Imbalance handling: exploring hybrid and more sophisticated strategies beyond pure class weighting.
•	Methodological transparency: alignment with the TRIPOD reporting standard for clinical prediction models.
7.3 3. External validation — international collaboration
To test generalizability beyond a single institution, we are developing protocols for external validation using the international MIMIC dataset, with preliminary discussions underway regarding a research collaboration with Dr. Leo Anthony Celi’s group at MIT.
8 🧬 Contributors & Collaborators
Contributor	Role	Affiliation
Bahar Homayoun (M.Sc.)	Lead Researcher	Department of Health Information Management and Medical Informatics, Tehran University of Medical Sciences (TUMS) — bahariiii2515@gmail.com
Farid Zand (M.D.)	Collaborator	Anesthesiology and Critical Care Research Center, Shiraz University of Medical Sciences (SUMS)
Naeimehossadat Asmarian (Ph.D.)	Collaborator	Anesthesiology and Critical Care Research Center, SUMS
Victor Daniel Rosenthal (M.D.)	Collaborator	University of Miami Miller School of Medicine, USA; International Nosocomial Infection Control Consortium (INICC)
Sharareh Rostam Niakan Kalhori (Ph.D.)	Senior Supervisor	Department of Health Information Management and Medical Informatics, TUMS; Research Fellow, TU Braunschweig & Hannover Medical School (MHH)
9 📂 Repository Structure, Installation & Usage
CLABSI-prediction-model/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── Under-development-SVM-model.ipynb
9.1 Installation & usage
Requirements: Python 3.9+
1.	Clone the repository:
git clone https://github.com/bahariiiiii/CLABSI-prediction-model.git
cd CLABSI-prediction-model
2.	Install dependencies:
pip install -r requirements.txt
3.	Run the pipeline: open Under-development-SVM-model.ipynb in Jupyter and execute the cells in order to reproduce the analytical workflow. All exported artifacts (tables, figures, interpretation files) are written to the configured output directory.
9.2 Key output artifacts
Artifact	Content
loio_per_icu_metrics.csv	Metrics for each held-out ICU
loio_patient_level_predictions.csv	Pooled patient-level LOIO predictions
loio_macro_summary_95ci.csv	Macro-averaged LOIO metrics with t-based 95% CIs
loio_complete_summary.csv / loio_summary_metadata.json	Consolidated run summary and metadata
risk_cutoffs.json	Three-tier risk-stratification cutoffs (validation-derived)
linear_svm_coefficients.csv	Interpretable model coefficients mapped to original variables
loio_icu_auc_forest_plot.* / decision_curve_analysis.*	Forest plot and DCA figures (PNG/SVG/PDF)
9.3 License
This project is licensed under the MIT License. See the LICENSE file in the repository root.
9.4 Acknowledgment
If you find this code, methodology, or pipeline useful for your own research, we kindly ask that you acknowledge this repository. We value transparency and open collaboration in critical-care informatics.

