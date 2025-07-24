import logging
import os

import joblib

from src.regression.preprocess_cluster_data import register_preprocessor
import pandas as pd
from sklearn.preprocessing import StandardScaler

@register_preprocessor(1)
def cluster_1_preprocessing(data_path:str)->str:
    output_dir = os.path.join('artifacts', 'cluster_1', 'preprocessing')
    os.makedirs(output_dir,exist_ok=True)
    
    dataset=pd.read_csv(data_path)
    sc=StandardScaler()
    bankrupt_=dataset['Bankrupt?']
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])
    joblib.dump(sc,f'{output_dir}/scaler.pkl')
    dataset['Bankrupt?']=bankrupt_

    columns_to_drop=[
    'Total debt/Total net worth',
    'Equity to Long-term Liability',
    'Long-term Liability to Current Assets',
    'Interest Expense Ratio'
    ]
    joblib.dump(columns_to_drop,f'{output_dir}/columns_to_drop.pkl')
    dataset=dataset.drop(columns=columns_to_drop)

    dataset.to_csv(os.path.join(output_dir,'processed_data.csv'),index=False)

    return os.path.join(output_dir,'processed_data.csv')