import logging
import os

import joblib
import pandas as pd
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction
from src.regression.preprocess_cluster_data import register_preprocessor
from sklearn.preprocessing import StandardScaler

@register_preprocessor(3)
def cluster_3_preprocessing(data_path:str)->str:
    output_dir=os.path.join('artifacts','cluster_3','preprocessing')
    os.makedirs(output_dir,exist_ok=True)

    dataset=pd.read_csv(data_path)

    sc=StandardScaler()
    bankrupt_=dataset['Bankrupt?']
    columns_to_drop=['Quick Assets/Current Liability',
                                  "PCA_Net Income to Stockholder's Equity_PCA_Borrowing dependency_Current Liabilities/Equity",
                                  'Fixed Assets to Assets',
                                  'Working capitcal Turnover Rate',
                                  'Quick Ratio',
                                  'Long-term Liability to Current Assets',
                                  'Cash/Current Liability',
                                  'Bankrupt?'
                                  ]
    
    dataset=dataset.drop(columns=columns_to_drop)
    joblib.dump(columns_to_drop,f'{output_dir}/columns_to_drop.pkl')
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])
    joblib.dump(sc,f'{output_dir}/scaler.pkl')

    final_df, pca_features, dropped_cols, all_pca_pairs, pca_models = hybrid_iterative_reduction(
        dataset,
        thresh_low=0.8,
        thresh_high=0.95,
        verbose=True
    )
    final_df['Bankrupt?']=bankrupt_

    if not all_pca_pairs.empty:
        pca_pairs_df = all_pca_pairs
    else:
        pca_pairs_df = pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])

    pca_dir=os.path.join(output_dir,'pca')
    os.makedirs(pca_dir,exist_ok=True)
    joblib.dump(dropped_cols, f'{pca_dir}/columns_to_drop.pkl')
    joblib.dump(pca_pairs_df, f'{pca_dir}/pca_pairs_used.pkl')
    joblib.dump(pca_models, f'{pca_dir}/fitted_pca_models.pkl')

    final_df.to_csv(os.path.join(output_dir,'processed_data.csv'),index=False)

    for root, dirs, files in os.walk(output_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            file_path = os.path.join(root, f)
            os.chmod(file_path, 0o666)

    return os.path.join(output_dir,'processed_data.csv')