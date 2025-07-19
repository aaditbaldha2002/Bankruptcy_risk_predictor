import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from collections import defaultdict

logger = logging.getLogger(__name__)

def get_corr_pairs(df, threshold_low=0.85, threshold_high=0.95):
    try:
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_pairs, pca_pairs = [], []

        for row in upper_tri.index:
            for col in upper_tri.columns:
                corr = upper_tri.loc[row, col] 
                if pd.notna(corr):
                    if corr > threshold_high:
                        drop_pairs.append((row, col, corr))
                    elif corr > threshold_low:
                        pca_pairs.append((row, col, corr))

        drop_df = pd.DataFrame(drop_pairs, columns=["Feature_1", "Feature_2", "Correlation"])
        pca_df = pd.DataFrame(pca_pairs, columns=["Feature_1", "Feature_2", "Correlation"])
        return drop_df.sort_values(by='Correlation', ascending=False), pca_df.sort_values(by='Correlation', ascending=False)
    except Exception as e:
        logger.exception("Error while computing correlation pairs.")
        raise

def filter_unique_pairs(df_corr):
    try:
        used, final = set(), []
        for _, row in df_corr.iterrows():
            f1, f2 = row['Feature_1'], row['Feature_2']
            if f1 not in used and f2 not in used:
                final.append(row)
                used.update([f1, f2])
        return pd.DataFrame(final)
    except Exception as e:
        logger.exception("Error while filtering unique pairs.")
        raise

def apply_drops(df, drop_df, tracked_drops):
    try:
        df_out = df.copy()
        for _, row in drop_df.iterrows():
            f1, f2 = row['Feature_1'], row['Feature_2']
            if f1 in df_out.columns and f2 in df_out.columns:
                tracked_drops.append(f2)
                df_out.drop(columns=[f2], inplace=True)
        return df_out
    except Exception as e:
        logger.exception("Error while dropping features.")
        raise

def apply_pca(df, pairs_df, pca_model_dict):
    try:
        df_out = df.copy()
        new_cols = []

        for _, row in pairs_df.iterrows():
            f1, f2 = row['Feature_1'], row['Feature_2']
            if f1 not in df_out.columns or f2 not in df_out.columns:
                continue

            subset = df_out[[f1, f2]].dropna()
            if subset.empty:
                continue

            data = subset.values - subset.values.mean(axis=0)

            # Check for constant variance
            if np.all(np.std(data, axis=0) == 0):
                continue

            pca = PCA(n_components=1)
            try:
                pca.fit(data)
            except Exception as e:
                logger.warning(f"Skipping PCA for {f1} and {f2} due to fitting error: {e}")
                continue

            key = f"{sorted([f1, f2])[0]}__{sorted([f1, f2])[1]}"
            pca_model_dict[key] = pca

            new_col = f"PCA_{f1}_{f2}"
            df_out[new_col] = np.nan
            df_out.loc[subset.index, new_col] = pca.transform(data).flatten()
            df_out.drop(columns=[f1, f2], inplace=True)
            new_cols.append(new_col)

        return df_out, new_cols
    except Exception as e:
        logger.exception("Error while applying PCA.")
        raise

def hybrid_iterative_reduction(df, thresh_low=0.8, thresh_high=0.95, verbose=False):
    try:
        df_iter = df.copy()
        iteration = 1
        all_pca_cols = []
        drop_track = []
        pca_model_dict = {}
        all_pca_pairs = []

        while True:
            drop_df, pca_df = get_corr_pairs(df_iter, threshold_low=thresh_low, threshold_high=thresh_high)
            if drop_df.empty and pca_df.empty:
                if verbose:
                    logger.info("No more correlated features to process.")
                break

            if verbose:
                logger.info(f"\n--- Iteration {iteration} ---")
                if not drop_df.empty:
                    logger.info("Drop Pairs:\n%s", drop_df)
                if not pca_df.empty:
                    logger.info("PCA Pairs:\n%s", pca_df)

            if not drop_df.empty:
                df_iter = apply_drops(df_iter, drop_df, drop_track)

            if not pca_df.empty:
                unique_pca_df = filter_unique_pairs(pca_df)
                df_iter, new_pca_cols = apply_pca(df_iter, unique_pca_df, pca_model_dict)
                all_pca_pairs.append(unique_pca_df)
                all_pca_cols.extend(new_pca_cols)

            iteration += 1
        
        if all_pca_pairs:
            all_pca_pairs_df = pd.concat(all_pca_pairs, ignore_index=True)
        else:
            all_pca_pairs_df = pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])

        return df_iter.reset_index(drop=True), all_pca_cols, drop_track, all_pca_pairs_df, pca_model_dict

    except Exception as e:
        logger.exception("Unexpected error in hybrid iterative PCA reduction.")
        raise
