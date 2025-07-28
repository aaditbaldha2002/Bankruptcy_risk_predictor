import logging
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
import stat

logger = logging.getLogger(__name__)

@register_preprocessor(0)
def cluster_0_preprocessing(data_path: str) -> str:
    # Create preprocessing artifacts directory
    output_dir = os.path.join('artifacts', 'cluster_0', 'preprocessing')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Reading dataset from {data_path}")
        dataset = pd.read_csv(data_path)

        # Preserve target column
        target_col = 'Bankrupt?'
        if target_col not in dataset.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset")

        bankrupt_ = dataset[target_col]

        # Standardize features (excluding last 2 cols and target)
        sc = StandardScaler()
        dataset.drop(columns=['Bankrupt?'],inplace=True)
        features = dataset
        scaled_features = sc.fit_transform(features)
        joblib.dump(sc,os.path.join(output_dir,'standard_scaler.pkl'))
        dataset = pd.DataFrame(scaled_features, columns=features.columns)

        # Drop known redundant/irrelevant features
        cols_to_drop = [
            'Cash/Total Assets', 
            'Total debt/Total net worth', 
            'Equity to Long-term Liability',
            'Cash/Current Liability', 
            'Long-term Liability to Current Assets', 
            'Quick Ratio',
            'Working capitcal Turnover Rate', 
            'Current Ratio', 
            'Quick Assets/Current Liability',
        ]

        dataset.drop(columns=[col for col in cols_to_drop if col in dataset.columns], inplace=True)
        joblib.dump(cols_to_drop,os.path.join(output_dir,'cols_to_drop_before_pca.pkl'))
        pca_dir=os.path.join(output_dir,'pca')
        os.makedirs(pca_dir,exist_ok=True)

        # Dimensionality reduction
        final_df, pca_features, dropped_cols, all_pca_pairs, pca_models = hybrid_iterative_reduction(
            dataset,
            thresh_low=0.9,
            thresh_high=0.95,
            verbose=True
        )

        # Handle PCA pairs output
        if not all_pca_pairs.empty:
            pca_pairs_df = all_pca_pairs
        else:
            pca_pairs_df = pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])

        if os.path.isfile(output_dir):
            raise RuntimeError(f"Expected {output_dir} to be a directory, but it's a file. Please delete or rename it.")
        

        # Persist artifacts
        joblib.dump(dropped_cols, os.path.join(pca_dir, 'columns_to_drop.pkl'))
        joblib.dump(pca_pairs_df, os.path.join(pca_dir, 'pca_pairs_used.pkl'))
        joblib.dump(pca_models, os.path.join(pca_dir, 'fitted_pca_models.pkl'))

        # Append target back
        final_df[target_col] = bankrupt_
        final_df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)

        for root, dirs, files in os.walk(output_dir):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, 0o666)

        logger.info(f"Preprocessing completed and saved to: {output_dir}")
        return os.path.join(output_dir,'processed_data.csv')

    except Exception as e:
        logger.error(f"Error during preprocessing for cluster 0: {e}", exc_info=True)
        raise
