import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_absolute_percentage_error as mape
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

EMBEDINGS_PATH = 'two_data.pt'
FEATURES_PATH = "done_all"
TARGETS_PATH = 'targets'

def train(MODEL, EMBEDINGS_PATH, FEATURES_PATH, TARGETS_PATH, NUM_FOLDS):
    X_emb = torch.load(EMBEDINGS_PATH)[:,:-1].numpy()
    al = []

    for i, folder in enumerate(os.listdir(FEATURES_PATH)):
        if '$' not in folder:
            df1 = pd.read_csv(f'{FEATURES_PATH}/{folder}')
            df2 = pd.read_csv(f'{TARGETS_PATH}/ad_{folder}')
            m = df1.merge(df2, how='inner', on='sid')
            al.append(m)

    df_al = pd.concat(al, axis=0, ignore_index=True)
    df_al = df_al.drop("sid", axis=1)
    al = df_al.values
    X_f, y = al[:,:29], al[:,-2:]
    data = np.hstack((X_f, X_emb))
    indexes = np.arange(data.shape[0])
    np.random.seed(27)
    np.random.shuffle(indexes)


    num_folds = NUM_FOLDS
    folds_indexes = np.array_split(indexes, num_folds)
    val_area_mapes, train_area_mapes = [], []
    val_delay_mapes, train_delay_mapes = [], []

    for i, fold in enumerate(folds_indexes):
        val_indexes = np.unique(fold)
        train_indexes = np.setdiff1d(np.unique(indexes), val_indexes)
        X_val, y_val = data[val_indexes], y[val_indexes]
        X_train, y_train = data[train_indexes], y[train_indexes]

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

        model = MODEL.set_params(nthread=1)
        
        model.fit(X_train, y_train)

        model.set_params(device="cpu")
        val_predictions = model.predict(X_val)
        train_predictions = model.predict(X_train)

        val_area_mape, train_area_mape = mape(y_val[:,0], val_predictions[:,0]), mape(y_train[:,0], train_predictions[:,0])
        val_area_mapes.append(val_area_mape)
        train_area_mapes.append(train_area_mape)

        val_delay_mape, train_delay_mape = mape(y_val[:,1], val_predictions[:,1]), mape(y_train[:,1], train_predictions[:,1])
        val_delay_mapes.append(val_delay_mape)
        train_delay_mapes.append(train_delay_mape)

        print(f'(Fold {i}) Train Area MAPE: {train_area_mape*100:.2f}%\t| Val Area MAPE: {val_area_mape*100:.2f}%\t\t| Train Delay MAPE: {train_delay_mape*100:.2f}%\t| Val Delay MAPE: {val_delay_mape*100:.2f}%')

    print(f'\nMean Train Area MAPE: {np.mean(train_area_mapes)*100:.2f}%\t| Mean Val Area MAPE: {np.mean(val_area_mapes)*100:.2f}%\t| Mean Train Delay MAPE: {np.mean(train_delay_mapes)*100:.2f}%\t| Mean Val Delay MAPE: {np.mean(val_delay_mapes)*100:.2f}%')
