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

---

## Data-Feature Description

The complete semantic description of each of the feature columns is in the file data/feature_dictionary.csv

---

## Models Used

- **Logistic Regression** – Baseline classifier for benchmarking  
- **Random Forest** – Interpretability and feature importance  
- **XGBoost** – Optimal performance in structured tabular data  
- **Voting Ensemble** – Combining predictions for improved generalization

Each model is trained with cross-validation and hyperparameter tuning. Production-ready models are logged in MLflow and encrypted prior to deployment.

---

## Data Analysis Process

- Descriptive statistics and financial signal profiling
- colinearity detection using Pearson Correlation Matrix, VIF
- Manually dropping feature columns and dimensionality reduction with the help of PCA
- Checking presence of higher level collinearity using condition number metric   
- Outlier detection using visualizations 
- Clustering Process for Cluster prediction models as well as pattern revelations made by the clustering process
- Respective visualizations for each of the clusters for explaining causes of the cluster formation

All analysis steps are documented and conducted in isolated compute environments.

---

## Data Segregation Pipeline

- KMeans Clustering implemented for the formation of the clusters
- XGB Classifier model used for cluster label prediction
- Separating dataset with respect to the predicted cluster labels

---

## Model Training and Prediction Pipeline

- Separate models for each of the clusters
- models trained on the separate cluster dataset
- Final testing dataset segregated using trained cluster label prediction model based on the cluster labels made using inference
- models trained on the cluster dataset predicts the target on the respective cluster dataset
- Final testing dataset prediction joined together for final results

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

