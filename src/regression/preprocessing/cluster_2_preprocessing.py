import logging
import os

import joblib
import pandas as pd
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
from sklearn.preprocessing import StandardScaler

@register_preprocessor(2)
def cluster_2_preprocessing(data_path:str)->str:
    output_dir=os.path.join('artifacts','cluster_2','preprocessing')
    os.makedirs(output_dir,exist_ok=True)

    dataset=pd.read_csv(data_path)

    sc=StandardScaler()
    bankrupt_=dataset['Bankrupt?']
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])
    joblib.dump(sc,os.path.join(output_dir,'scaler.pkl'))

    dataset['Bankrupt?']=bankrupt_

    columns_to_drop=['Total debt/Total net worth',
                                    'Cash/Current Liability',
                                    'Long-term Liability to Current Assets',
                                    'Quick Ratio',
                                    'Current Ratio',
                                    'Quick Assets/Current Liability',
                                    'Bankrupt?'
                                    ]

    dataset=dataset.drop(columns=columns_to_drop)

    final_df, pca_features, dropped_cols, all_pca_pairs, pca_models = hybrid_iterative_reduction(
        dataset,
        thresh_low=0.9,
        thresh_high=0.95,
        verbose=True
    )

    joblib.dump(columns_to_drop,f'{output_dir}/columns_to_drop.pkl')
    final_df['Bankrupt?']=bankrupt_

    if not all_pca_pairs.empty:
        pca_pairs_df = all_pca_pairs
    else:
        pca_pairs_df = pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])

    final_df.to_csv(os.path.join(output_dir,'processed_data.csv'),index=False)
    pca_dir=os.path.join(output_dir,'pca')
    os.makedirs(pca_dir,exist_ok=True)
    joblib.dump(dropped_cols, f'{pca_dir}/columns_to_drop.pkl')
    joblib.dump(pca_pairs_df, f'{pca_dir}/pca_pairs_used.pkl')
    joblib.dump(pca_models, f'{pca_dir}/fitted_pca_models.pkl')

    return os.path.join(output_dir,'processed_data.csv')