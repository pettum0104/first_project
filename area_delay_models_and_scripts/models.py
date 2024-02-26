import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from train import *

def result(done_folder_path, targets_folder_path):
    data = []
    for folder in os.listdir(done_folder_path):
        folder = '_'.join(folder.split('_')[:-1])
        if '$' not in folder:
            df1 = pd.read_csv(f'{done_folder_path}/{folder}_my.csv')
            df2 = pd.read_csv(f'{targets_folder_path}/ad_{folder}.csv')
            m = df1.merge(df2, how='inner', on='sid')
            data.append(m)
            
    data = pd.concat(data, axis=0, ignore_index=True)
    data = data.drop(["sid", 'LEN', 'transitivity'], axis=1)

    X, area, delay = data.drop(["area", "delay"], axis=1), data[["area"]].values.flatten(), data[["delay"]].values.flatten()
    
    scaler = load('scaler.joblib')
    X = pd.DataFrame(scaler.transform(X), columns = X.columns)
    
    area_model = load('area_model.joblib')
    delay_model = load('delay_model.joblib')
    
    print(f'area MAPE: {mape(area, area_model.predict(X))*100:.2f}%, delay MAPE: {mape(delay, delay_model.predict(X))*100:.2f}%')


area_xgboost_model = XGBRegressor(objective='reg:squarederror', tree_method="hist", eval_metric=mape, n_jobs=-1, 
                     n_estimators=100, reg_lambda=10, max_depth=4, device="cuda")
delay_dtr_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=400)
scaler = load('scaler_filename.joblib')
    