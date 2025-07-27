import os
import pandas as pd

dataset=pd.read_csv('data/raw/train_data.csv')
input_data=dataset.iloc[[1]]

print(type(input_data))
print(input_data.head())

input_data_dir=os.path.join('data','raw')
input_data.to_csv(os.path.join(input_data_dir,'inference_test_data.csv'),index=False)