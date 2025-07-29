import logging
import os

import joblib
import pandas as pd
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
from sklearn.preprocessing import StandardScaler

@register_preprocessor(4)
def cluster_4_preprocessing(data_path:str)->str:
    output_dir=os.path.join('artifacts','cluster_4','preprocessing')
    os.makedirs(output_dir,exist_ok=True)

    dataset=pd.read_csv(data_path)
    sc=StandardScaler()
    bankrupt_=dataset['Bankrupt?']
    dataset.drop(columns=['Bankrupt?'],inplace=True)
    dataset=pd.DataFrame(sc.fit_transform(dataset),columns=dataset.columns)
    
    joblib.dump(sc,os.path.join(output_dir,'cluster_4_standard_scaler.pkl'))
    
    final_df, pca_features, dropped_cols, all_pca_pairs, pca_models = hybrid_iterative_reduction(
        dataset,
        thresh_low=0.9,
        thresh_high=0.95,
        verbose=True
    )
    
    if not all_pca_pairs.empty:
        pca_pairs_df = all_pca_pairs
    else:
        pca_pairs_df = pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])

    pca_dir=os.path.join(output_dir,'pca')
    os.makedirs(pca_dir,exist_ok=True)
    
    joblib.dump(dropped_cols, os.path.join(pca_dir,'cluster_4_columns_to_drop.pkl'))
    joblib.dump(pca_pairs_df, os.path.join(pca_dir,'cluster_4_pca_pairs_used.pkl'))
    joblib.dump(pca_models, os.path.join(pca_dir,'cluster_4_fitted_pca_models.pkl'))

    final_df['Bankrupt?']=bankrupt_

    final_df.to_csv(os.path.join(output_dir,'preprocessed_data.csv'),index=False)
    os.chmod(output_dir,0o777)
    for root, dirs, files in os.walk(output_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            file_path = os.path.join(root, f)
            os.chmod(file_path, 0o666)

    return os.path.join(output_dir,'preprocessed_data.csv')