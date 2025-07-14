
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing import adaptive_transform
from src.preprocessing.pca_feature_reduction import hybrid_iterative_reduction

def preprocess_data(data_path:str)->str:
    dataset= pd.read_csv(data_path)
    indexes=dataset['Index']
    bankrupt=dataset['Bankrupt?']
    dataset=dataset.drop(columns=['Index','Bankrupt?'])
    scaler=StandardScaler()
    scaled_dataset=scaler.fit_transform(dataset)

    columns_to_drop=[
        'Research and development expense rate',
        'Interest-bearing debt interest rate',
        'Allocation rate per person',
        'Net Value Per Share (B)',
        'Net Value Per Share (A)',
        'Net Value Per Share (C)',
        'Per Share Net profit before tax (Yuan ¥)',
        'Non-industry income and expenditure/revenue',
        'Revenue per person',
        'Operating profit per person',
        'Net Income Flag',
        'Cash Flow Per Share',
        'Operating Expense Rate',
        'Tax rate (A)',
        'Revenue Per Share (Yuan ¥)',
        'Fixed Assets Turnover Frequency',
        'Inventory Turnover Rate (times)',
        'Net Worth Turnover Rate (times)',
        'Total Asset Turnover',
        'Accounts Receivable Turnover',
        'Average Collection Days',
        'Current Asset Turnover Rate',
        'Quick Asset Turnover Rate',
        'Cash Turnover Rate',
        'Total assets to GNP price',
        'Inventory and accounts receivable/Net value',
        'Inventory/Working Capital',
        'Inventory/Current Liability',
    ]

    scaled_dataset=pd.DataFrame(scaled_dataset,columns=dataset.columns)
    scaled_dataset=dataset.drop(columns=columns_to_drop)

    dataset_pca,pca_features,dropped_cols,pca_pairs_df,pca_models=hybrid_iterative_reduction(scaled_dataset,thresh_low=0.8,thresh_high=0.95,verbose=True)

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/pca',exist_ok=True)

    joblib.dump(dropped_cols, 'output/pca/columns_to_drop.pkl')
    joblib.dump(pca_pairs_df, 'output/pca/pca_pairs_used.pkl')
    joblib.dump(pca_models, 'output/pca/fitted_pca_models.pkl')
    
    dataset_pca=dataset.drop(columns=['Working Capital to Total Assets'])
    os.makedirs('output/intermediate',exist_ok=True)
    file_path='output/intermediate/dataset_pca.csv'
    dataset_pca.to_csv(file_path,index=False)
    # PCA step concludes

    #Gaussian transformation step starts
    transformed_data_path=adaptive_transform(file_path)

    return transformed_data_path
    