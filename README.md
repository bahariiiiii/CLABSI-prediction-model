Interpretable Machine Learning Framework for CLABSI Risk Prediction in Critical Care
[!IMPORTANT]
RESEARCH PROTOTYPE ONLY: This repository is currently under active development. The models and pipelines presented here are for research purposes and are not intended for clinical decision-making or medical diagnosis.
⚠️ Project Status: Under Active Development
This project is an evolving research framework. Our primary goal is to build a robust, high-performance, and strictly leakage-free pipeline for predicting Central Line-Associated Bloodstream Infections (CLABSI). Exact performance metrics and benchmark comparisons are undergoing final evaluation and will be published upon study completion.
Current Focus:
•	Refining the end-to-end pipeline to ensure absolute data integrity (Zero-Leakage).
•	Optimizing preprocessing strategies (Missing value handling, Class imbalance management).
•	Fine-tuning model components: Feature selection, Bootstrap-based uncertainty estimation, Threshold tuning, and Probability calibration.
________________________________________
🚀 Roadmap & Future Directions
1. Algorithmic Benchmarking
Initial Benchmarking Insights: In preliminary comparative experiments, the Linear Support Vector Machine (Linear SVM) demonstrated superior overall performance—achieving optimal discriminative capacity (discrimination ability across evaluation metrics) while exhibiting minimal risk of overfitting compared to more complex architectures on this cohort. Furthermore, upcoming updates will include benchmarking against other standard algorithms, such as:
•	Logistic Regression (LR)
•	Random Forest (RF)
•	XGBoost
2. Advanced Pipeline Optimization
We are actively testing various methodologies to maximize performance and clinical utility, including:
•	Advanced feature selection techniques.
•	More sophisticated handling of missing data and class imbalance.
•	Enhanced probability calibration for better reliability.
3. External Validation (International Collaboration)
To test the model’s generalizability, we are developing protocols for External Validation using:
•	The international MIMIC dataset.
•	Preliminary discussions regarding a potential research collaboration with Dr. Leo Anthony Celi’s research group at MIT
________________________________________
🧬 Contributors & Collaborators
•	Bahar Homayoun (M.Sc.) (Lead Researcher)
Department of Health Information Management and Medical Informatics, Tehran University of Medical Sciences (TUMS).
Email: bahariiii2515@gmail.com
•	Farid Zand (M.D.)
Anesthesiology and Critical Care Research Center, Shiraz University of Medical Sciences (SUMS).
•	Naeimehossadat Asmarian (Ph.D.)
Anesthesiology and Critical Care Research Center, SUMS.
•	Victor Daniel Rosenthal (M.D.)
University of Miami Miller School of Medicine, USA; International Nosocomial Infection Control Consortium (INICC).
•	Sharareh Rostam Niakan Kalhori (Ph.D.) (Senior Supervisor)
Department of Health Information Management and Medical Informatics, TUMS& Research Fellow at Technical University of Braunschweig (TU Braunschweig) & the Hannover Medical School (MHH).
________________________________________
🏥 Clinical Cohort & Setting
•	Study Location: Namazi Hospital (Tertiary academic center), Shiraz, Iran.
•	Participating ICUs: 9 specialized adult intensive care units.
•	Cohort Details: 
o	N = 5200 matched patients (295 CLABSI cases matched 1:2 with controls).
o	Inclusion: Adult patients with CVC duration > 48h.
o	Outcome: Definite CLABSI (based on CDC/NHSN 2017 criteria).
•	Ethics: Approved by the Institutional Ethics Committee of TUMS (IR.TUMS.SPH.REC.1403.308).
Data Availability Statement
The clinical datasets utilized in this research contain sensitive Protected Health Information (PHI) and are governed by institutional data use agreements. As such, raw patient data cannot be shared publicly. Researchers interested in collaboration or reproducibility validation should contact the lead author for potential data access protocols and institutional approvals.
________________________________________
🛠️ Key Methodological Highlights
Leakage-Free Validation (LOIO)
We use a Leave-One-ICU-Out (LOIO) validation framework. This evaluates the model’s ability to perform across different specialized units within the same institution. To prevent data leakage:
•	Feature Selection and Calibration are performed exclusively on the development set.
•	The held-out ICU remains completely isolated until final evaluation.
Interpretability & Reliability
•	SHAP (SHapley Additive exPlanations): Transparency in feature contribution.
•	Decision Curve Analysis (DCA): Quantifying clinical net benefit.
•	Bootstrap Uncertainty: Performance metrics include 95% Confidence Intervals.
________________________________________
📂 Repository Structure
                                            content_copy                        CLABSI-Prediction/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└──Under-development-SVM-model.ipynb
________________________________________
💻 Installation & Usage
Requirements: Python 3.9+
1.	Clone the repository:
bashnote_addویرایش با Canvas
   git clone https://github.com/bahariiiiii/ CLABSI-prediction-model.git
   cd CLABSI-prediction-model
2.	Install dependencies:
                                            content_copy                        bashnote_addویرایش با Canvas
   pip install -r requirements.txt
3.	Run the pipeline:Open the Jupyter Notebook to reproduce the analytical workflow.
________________________________________
📜 License
This project is licensed under the MIT License. For more details, please see the LICENSE file in the repository root.
🤝 Acknowledgment
If you find this code, methodology, or pipeline useful for your own research or development projects, we appreciate it if you acknowledge this repository. We value transparency and open collaboration in critical care informatics.

