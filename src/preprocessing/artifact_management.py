import os
import joblib

def save_artifacts(output_dir, dropped_cols, pca_pairs_df, pca_models):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(dropped_cols, os.path.join(output_dir, 'columns_to_drop.pkl'))
    joblib.dump(pca_pairs_df, os.path.join(output_dir, 'pca_pairs_used.pkl'))
    joblib.dump(pca_models, os.path.join(output_dir, 'fitted_pca_models.pkl'))

def load_artifacts(output_dir):
    dropped_cols = joblib.load(os.path.join(output_dir, 'columns_to_drop.pkl'))
    pca_pairs_df = joblib.load(os.path.join(output_dir, 'pca_pairs_used.pkl'))
    pca_models = joblib.load(os.path.join(output_dir, 'fitted_pca_models.pkl'))
    return dropped_cols, pca_pairs_df, pca_models
