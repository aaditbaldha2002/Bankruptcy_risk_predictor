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
    dataset.drop(columns=['Bankrupt?'],inplace=True)
    dataset=pd.DataFrame(sc.fit_transform(dataset),columns=dataset.columns)
    joblib.dump(sc,os.path.join(output_dir,'scaler.pkl'))
    dataset['Bankrupt?']=bankrupt_

    columns_to_drop=[
    'Total debt/Total net worth',
    'Equity to Long-term Liability',
    'Long-term Liability to Current Assets',
    'Interest Expense Ratio'
    ]
    dataset=dataset.drop(columns=columns_to_drop)
    joblib.dump(columns_to_drop,os.path.join(output_dir,'columns_to_drop.pkl'))

    dataset.to_csv(os.path.join(output_dir,'processed_data.csv'),index=False)

    for root, dirs, files in os.walk(output_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            file_path = os.path.join(root, f)
            os.chmod(file_path, 0o666)

    return os.path.join(output_dir,'processed_data.csv')