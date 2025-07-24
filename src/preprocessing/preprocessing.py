import os
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

from src.preprocessing.adaptive_transform import adaptive_transform
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction

def preprocess_data(data_path: str) -> str:
    try:
        logging.info(f"[Preprocessing] Loading dataset from: {data_path}")
        dataset = pd.read_csv(data_path)
        dataset.columns=dataset.columns.str.strip()
        # Separate key columns
        indexes = dataset['Index']
        bankrupt = dataset['Bankrupt?']
        dataset = dataset.drop(columns=['Index', 'Bankrupt?'])
        ARTIFACTS_DIR = "artifacts/preprocessing"

        # Scaling
        logging.info("[Preprocessing] Applying StandardScaler.")
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(dataset)
        joblib.dump(scaler,os.path.join(ARTIFACTS_DIR, "first_scaler.pkl"))
        scaled_dataset = pd.DataFrame(scaled_array, columns=dataset.columns)

        # Drop low-importance or redundant features
        columns_to_drop = [
            'Research and development expense rate', 
            'Interest-bearing debt interest rate',
            'Allocation rate per person', 
            'Net Value Per Share (B)',
            'Net Value Per Share (A)', 
            'Net Value Per Share (C)',
            'Per Share Net profit before tax (Yuan ¥)',
            'Non-industry income and expenditure/revenue', 
            'Revenue per person',
            'Operating profit per person', 
            'Net Income Flag', 
            'Cash Flow Per Share',
            'Operating Expense Rate', 
            'Tax rate (A)', 
            'Revenue Per Share (Yuan ¥)',
            'Fixed Assets Turnover Frequency', 
            'Inventory Turnover Rate (times)',
            'Net Worth Turnover Rate (times)', 
            'Total Asset Turnover',
            'Accounts Receivable Turnover', 
            'Average Collection Days',
            'Current Asset Turnover Rate', 
            'Quick Asset Turnover Rate',
            'Cash Turnover Rate', 
            'Total assets to GNP price',
            'Inventory and accounts receivable/Net value',
            'Inventory/Working Capital', 
            'Inventory/Current Liability',
        ]

        joblib.dump(columns_to_drop, os.path.join(ARTIFACTS_DIR, "columns_to_drop_before_pca.pkl"))
        scaled_dataset.drop(columns=columns_to_drop,axis=1, inplace=True)
        logging.info(f"[Preprocessing] Dropped {len(columns_to_drop)} columns.")

        # Apply hybrid PCA
        logging.info("[Preprocessing] Performing PCA reduction.")
        dataset_pca, pca_features, dropped_cols, pca_pairs_df, pca_models = hybrid_iterative_reduction(
            scaled_dataset, thresh_low=0.8, thresh_high=0.95, verbose=True
        )
        logging.info("PCA reduction completed")
        # Drop additional column (if still exists)
        dataset_pca.drop(columns=['Working Capital to Total Assets'], inplace=True, errors='ignore')
        dataset_pca['Bankrupt?']=bankrupt

        # Output saving
        logging.info("Storing the artifacts...")
        PCA_DIR = os.path.join(ARTIFACTS_DIR, "pca")
        INTERMEDIATE_DIR = os.path.join(ARTIFACTS_DIR, "intermediate")

        os.makedirs(PCA_DIR, exist_ok=True)
        joblib.dump(['Working Capital to Total Assets'],os.path.join(PCA_DIR,"columns_to_drop_after_pca.pkl"))
        joblib.dump(dropped_cols, os.path.join(PCA_DIR, "columns_to_drop.pkl"))
        joblib.dump(pca_pairs_df, os.path.join(PCA_DIR, "pca_pairs_used.pkl"))
        joblib.dump(pca_models, os.path.join(PCA_DIR, "fitted_pca_models.pkl"))

        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        intermediate_path = os.path.join(INTERMEDIATE_DIR, "dataset_pca.csv")
        dataset_pca.to_csv(intermediate_path, index=False)
        logging.info('Artifacts stored successfully.')
        # Gaussian transformation
        logging.info("[Preprocessing] Applying Gaussian transformation.")
        transformed_data_path = adaptive_transform(intermediate_path)

        logging.info(f"[Preprocessing] Transformation complete. Output: {transformed_data_path}")
        return transformed_data_path

    except Exception as e:
        logging.error(f"[Preprocessing] Error occurred: {e}", exc_info=True)
        raise
