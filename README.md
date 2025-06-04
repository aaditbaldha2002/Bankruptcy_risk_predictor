# ML_Project_Spring_2025

This machine learning project is designed to assess bankruptcy risk in corporate entities using structured financial indicators. The objective is to support internal compliance teams, auditors, and investors with predictive insights, while maintaining strict data governance and operational security.

---

## Tech Stack Used

- **Jupyter Notebook** – Exploratory development and rapid prototyping  
- **Python (v3.10)** – Primary programming language  
- **scikit-learn** – Core ML models and pipelines  
- **XGBoost / LightGBM** – High-performance gradient boosting models  
- **pandas / NumPy** – Data manipulation and preprocessing  
- **matplotlib / seaborn / Plotly** – Visual analysis and correlation mapping  
- **MLflow** – Experiment tracking and model registry  
- **Git** – Source control and team collaboration

> ⚠️ *All sensitive configurations and deployments are handled through secure secrets management and CI/CD pipelines.*

## Models Used

- **Logistic Regression** – Baseline classifier for benchmarking  
- **Random Forest** – Interpretability and feature importance  
- **XGBoost** – Optimal performance in structured tabular data  
- **Voting Ensemble** – Combining predictions for improved generalization

Each model is trained with cross-validation and hyperparameter tuning. Production-ready models are logged in MLflow and encrypted prior to deployment.

---

## Data Analysis Process

- Descriptive statistics and financial signal profiling  
- Outlier detection using IQR and Mahalanobis distance  
- Multicollinearity analysis via VIF scores  
- Dimensionality reduction (PCA, t-SNE) for clustering insights  
- Trend and seasonality detection on time-indexed financial features

All analysis steps are documented and conducted in isolated compute environments.

---

## Project Workflow

1. **Data Ingestion**: CSVs/APIs with secure credential handling  
2. **Preprocessing**: Null imputation, encoding, normalization  
3. **Model Training**: Stratified splits, grid/random search for tuning  
4. **Evaluation**: ROC-AUC, Precision-Recall, F1, and business cost function  
5. **Model Packaging**: `joblib` or `pickle` with encrypted serialization  
6. **Version Control**: Tracked using Git and MLflow  
7. **Deployment Prep**: Ready for batch or API inference pipelines

---

## Inference Process

- Accepts structured financial input via API or secure batch process  
- Returns binary output: `1` (High Risk), `0` (Low Risk)  
- Provides associated probability/confidence score  
- Includes SHAP-based feature contribution for explainability  
- Logs inference results securely for compliance and monitoring

---

## Future Improvements

- Integration with real-time financial data feeds  
- Migration to transformer-based tabular models (e.g., TabNet, FT-Transformer)  
- Live dashboard with drill-down visualizations for decision-makers  
- CI-triggered model retraining pipelines with data drift detection  
- Inclusion of adversarial robustness and penetration testing for critical predictions

---

> 📁 For internal use only. All data access, model deployment, and usage policies are subject to corporate IT and legal compliance reviews.

