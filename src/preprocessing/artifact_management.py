import os
import joblib
import logging

logger = logging.getLogger(__name__)

def save_artifacts(output_dir, dropped_cols, pca_pairs_df, pca_models):
    try:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(dropped_cols, os.path.join(output_dir, 'columns_to_drop.pkl'))
        joblib.dump(pca_pairs_df, os.path.join(output_dir, 'pca_pairs_used.pkl'))
        joblib.dump(pca_models, os.path.join(output_dir, 'fitted_pca_models.pkl'))
        logger.info(f"Artifacts saved successfully to {output_dir}")
    except Exception as e:
        logger.exception(f"Failed to save artifacts to {output_dir}: {e}")
        raise

def load_artifacts(output_dir):
    try:
        dropped_cols = joblib.load(os.path.join(output_dir, 'columns_to_drop.pkl'))
        pca_pairs_df = joblib.load(os.path.join(output_dir, 'pca_pairs_used.pkl'))
        pca_models = joblib.load(os.path.join(output_dir, 'fitted_pca_models.pkl'))
        logger.info(f"Artifacts loaded successfully from {output_dir}")
        return dropped_cols, pca_pairs_df, pca_models
    except FileNotFoundError as fnf_error:
        logger.error(f"Artifact file not found in {output_dir}: {fnf_error}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load artifacts from {output_dir}: {e}")
        raise
