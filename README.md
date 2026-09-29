Loan Default Prediction
Binary classification project predicting whether a loan applicant will default. Built as a class project and Kaggle competition entry ("Loan Approval Prediction"), comparing Logistic Regression, Random Forest, XGBoost, and a Deep Neural Network on an imbalanced dataset where only 14.2% of loans default. XGBoost won with ROC-AUC 0.955 on the validation set.
Results (validation set, after leakage fix)
Model
Accuracy
F1-Score
ROC-AUC
XGBoost ✅
0.952
0.809
0.955
Random Forest
0.951
0.807
0.948
Deep Neural Network
0.947
0.788
0.933
Logistic Regression
0.907
0.590
0.903
Dataset
58,645 training rows × 20 variables; 39,098 test rows (data/submission.csv holds the predicted default probabilities for the test set).
Target: binary default flag (0 = repaid, 1 = default).
Feature groups:
Personal: age, home ownership, employment length
Loan details: loan amount, interest rate, loan intent, loan grade (A–G), loan-to-income ratio
Credit history: prior default on file, length of credit history
Methodology
Cleaning & encoding — loan grade ordinal-mapped (A→G); prior-default flag binarized; home ownership and loan intent one-hot encoded.
Feature engineering — loan_percent_income ratio; log transforms on income and loan amount to tame skew.
Train/test alignment — columns aligned between train and test before modeling.
Modeling — Logistic Regression (interpretable baseline), Random Forest and XGBoost (tree ensembles), plus a Keras/TensorFlow Deep Neural Network.
Evaluation — with a 14.2% minority class, raw accuracy is misleading, so models are compared on F1-score and ROC-AUC, with attention to precision/recall trade-offs.
Leakage detection & fix — the first modeling pass scored a suspicious perfect 1.0. Feature-importance analysis revealed the target column (loan_status) had leaked into the feature set during one-hot encoding. It was removed, and the results above are the honest re-run — a realistic assessment of credit risk, which is what matters for actual deployment.
Repository structure
loan-default-prediction/
├── README.md
├── loan_default_prediction.ipynb     # full pipeline: EDA → feature engineering → modeling → submission
├── data/
│   └── submission.csv                  # test-set predicted default probabilities (id, loan_status)
└── presentation/
    └── loan_default_presentation.pdf   # class presentation: problem, EDA, modeling, takeaways
How to run
Requires Python 3 with:
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
Open loan_default_prediction.ipynb in Jupyter and run top to bottom. The notebook reads the Kaggle competition files (train.csv, test.csv) — download them from the competition page into the working directory first.
Skills demonstrated
Python · scikit-learn · XGBoost · deep neural networks (Keras/TensorFlow) · binary classification · imbalanced data (F1, ROC-AUC over accuracy) · feature engineering (ratios, log transforms, ordinal/one-hot encoding) · Kaggle competition workflow
