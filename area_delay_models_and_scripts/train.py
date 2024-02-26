import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape


def train(model, scaler, done_folder_path, targets_folder_path):
    data = []

    for i, folder in enumerate(os.listdir(done_folder_path)):
        folder = '_'.join(folder.split('_')[:-1])
        if '$' not in folder:
            df1 = pd.read_csv(f'{done_folder_path}/{folder}_my.csv')
            df2 = pd.read_csv(f'{targets_folder_path}/ad_{folder}.csv')
            m = df1.merge(df2, how='inner', on='sid')
            m = m.drop(['sid', 'LEN', 'transitivity'], axis=1).values
            data.append(m)
            
    val_idx = np.random.randint(len(os.listdir(done_folder_path)), size=3)
    train = np.vstack([data[i] for i in list(set(range(len(data))) - set(val_idx))]) 
    val = np.vstack([data[i] for i in val_idx]) 
    X_train, area_train, delay_train = scaler.transform(train[:, :-2]), train[:, -2], train[:, -1]
    X_val, area_val, delay_val = scaler.transform(val[:, :-2]), val[:, -2], val[:, -1]
    
    model.fit(X_train, delay_train)

    val_predictions = model.predict(X_val)
    train_predictions = model.predict(X_train)

    print(f'Val MAPE: {mape(delay_val, val_predictions)*100:.2f}% | Train MAPE: {mape(delay_train, train_predictions)*100:.2f}%')
