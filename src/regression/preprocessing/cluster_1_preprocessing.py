import logging
import os

from src.regression.preprocess_cluster_data import register_preprocessor
import pandas as pd
from sklearn.preprocessing import StandardScaler

@register_preprocessor(1)
def cluster_1_preprocessing(data_path:str)->str:
    dataset=pd.read_csv(data_path)
    sc=StandardScaler()
    indexes=dataset['Index']
    bankrupt_=dataset['Bankrupt?']
    dataset=pd.DataFrame(sc.fit_transform(dataset.iloc[:,:-2]),columns=dataset.columns[:-2])
    dataset['Bankrupt?']=bankrupt_

    columns_to_drop=[
    'Total debt/Total net worth',
    'Equity to Long-term Liability',
    'Long-term Liability to Current Assets',
    'Interest Expense Ratio'
    ]
    dataset=dataset.drop(columns=columns_to_drop)

    DATA_PATH='artifacts'
    DATA_PATH=os.path.join('cluster_1')
    DATA_PATH=os.path.join('preprocessing')

    dataset.to_csv(DATA_PATH,index=False)

    return DATA_PATH