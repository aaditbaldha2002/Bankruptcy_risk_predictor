import logging
import os

import joblib
import pandas as pd
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
from sklearn.preprocessing import StandardScaler

@register_preprocessor(4)
def cluster_4_preprocessing(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    sc=StandardScaler()
    bankrupt_=dataset['Bankrupt?']
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])

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


    output_dir=os.path.join('artifacts','cluster_4','preprocessing')
    os.makedirs(output_dir,exist_ok=True)

    joblib.dump(dropped_cols, f'{output_dir}/columns_to_drop.pkl')
    joblib.dump(pca_pairs_df, f'{output_dir}/pca_pairs_used.pkl')
    joblib.dump(pca_models, f'{output_dir}/fitted_pca_models.pkl')


    final_df['Bankrupt?']=bankrupt_


    final_df.to_csv(os.path.join(output_dir,'preprocessed_data.csv'))
    return os.path.join(output_dir,'preprocessed_data.csv')