from models import *
import pandas as pd
import numpy as np
import os
import networkx as nx
import scipy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import MeanAbsolutePercentageError
from sklearn.metrics import mean_absolute_percentage_error as cpu_mape
import gc
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

BATCHES_PATH = 'D:/100_batches' # путь к папке с батчами

def train_model(model, dataset, batch_train_sampler, val_loader, criterion, optimizer, num_epochs, device):    
    loss_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train() 
        
        loss_accum = 0
        c = 0
        for i_step in batch_train_sampler:
            data = dataset[i_step]
            x_gpu = data.X.to(device)
            y_gpu = data.delay[:,None].to(device) # delay или area
            a_gpu = data.A.to(device)
            batch_gpu = data.batch.type(torch.int64).to(device)
            prediction = model(x_gpu, a_gpu, batch_gpu)    
            loss_value = criterion(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            print("Epoch: %d Step %d Train mape: %f" % (epoch, c, loss_value)) 
            c+=1
            loss_accum += loss_value
            
            torch.cuda.empty_cache()
            del x_gpu, y_gpu, a_gpu, batch_gpu, prediction
            gc.collect()

        ave_loss = loss_accum / c
        torch.cuda.empty_cache()
        val_mape = compute_mape(model, dataset, batch_val_sampler, device)

        loss_history.append(float(ave_loss))
        val_history.append(val_mape)

        print("\nAverage mape: %f, Val mape: %f\n" % (ave_loss, val_mape))
        
    return loss_history, val_history
        
def compute_mape(model, dataset, batch_val_sampler, device):
    model.eval()
    val_mape = 0
    for i_step in batch_val_sampler:
        data = dataset[i_step]
        x_gpu = data.X.to(device)
        y_gpu = data.delay[:,None].to(device) # delay или area
        a_gpu = data.A.type(torch.int64).to(device)
        batch_gpu = data.batch.type(torch.int64).to(device)
        prediction = model(x_gpu, a_gpu, batch_gpu)  
        mape_value = cpu_mape(y_gpu.cpu().detach().numpy(), prediction.cpu().detach().numpy())

        val_mape += mape_value
        
        torch.cuda.empty_cache()
        del x_gpu, y_gpu, a_gpu, batch_gpu, prediction
        gc.collect()
        
    mape = val_mape/len(batch_val_sampler)
    return mape


class BatchDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path #100_batches small

    def __getitem__(self, i):
        A = torch.load(f'{self.file_path}/A_{i}.pt')
        X = torch.load(f'{self.file_path}/X_{i}.pt')
        batch = torch.load(f'{self.file_path}/batch_{i}.pt')
        area = torch.load(f'{self.file_path}/area_{i}.pt')
        delay = torch.load(f'{self.file_path}/delay_{i}.pt')
        
        if X.shape[1] == 2:
            X_enc = F.one_hot(X[:,0].type(torch.int64))
            X_enc = torch.cat((X_enc, X[:,1][:,None]), dim=-1)
            X = X_enc
            
        return Data(A=A, X=X, area=area, delay=delay, batch=batch)

    def __len__(self):
        return len(os.listdir(self.file_path))//5


def main():
#     train_indices, val_indices = list(range(750)), list(range(750, 871)) # 50 batch
    train_indices, val_indices = list(range(375)), list(range(375, 436)) # 100 batch

    batch_train_sampler = SubsetRandomSampler(train_indices)
    batch_val_sampler = SubsetRandomSampler(val_indices)

    file_path = BATCHES_PATH
    dataset = BatchDataset(file_path)
    
    model = GNN(input_size = 4, hidden_channels=30, n_predict=1, conv = GCNConv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    model.to(device)
    criterion = MeanAbsolutePercentageError().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # , weight_decay=1e-4
    loss_history, val_history = train_model(model, dataset, batch_train_sampler, batch_val_sampler, criterion, optimizer, 1, device)
