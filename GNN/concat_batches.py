import pandas as pd
import numpy as np
import os
import networkx as nx
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix, from_networkx
import scipy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys


class LargeDataset(Dataset):
    def __init__(self, folder_path, target_path):
        self.folder_path = folder_path
        self.target_path = target_path

    def __getitem__(self, index):
        l = os.listdir(self.folder_path)
        A_name = l[2*index]
        X_name = l[2*index + 1]
        name = '_'.join(A_name.split('_')[:-3])
        sid = int(A_name.split('_')[-3][3:])
        df = pd.read_csv(f'{self.target_path}/ad_{name}.csv') 
        trs = df[['sid', 'area', 'delay']]
        sparse_matrix = scipy.sparse.load_npz(self.folder_path+'/'+A_name)
        row, col = sparse_matrix.nonzero()
        A = torch.tensor(np.column_stack((row, col)).T)

        X = np.load(self.folder_path+'/'+X_name)
        X = torch.tensor(X, dtype=torch.float32)
        
        area = trs[trs['sid']==sid]['area'].iloc[0]
        delay = trs[trs['sid']==sid]['delay'].iloc[0]
            
        return Data(A=A, X=X, area=area, delay=delay, A_name=A_name)

    def __len__(self):
        return len(os.listdir(self.folder_path))//2

    
def main(folder_path, target_path, batch_path, n_batch):
    train_indices, val_indices = [], []

    names = np.array(os.listdir(target_path))
    files = os.listdir(folder_path)
    n_val = 4
    val_names = list(map(lambda x: x[:-7], names[np.random.choice(len(names), n_val, replace=False)]))

    for name in val_names:          # из имен схем получаем имена и индексы всех файлов, относящихся к ним
        A_name = f'{name}_syn0_step20_A.npz'
        if name != 'mem_ctrl':
            ind = files.index(A_name)
            val_indices += list(range(ind//2, ind//2 + 1500))
        else:
            ind = files.index(A_name)
            val_indices += list(range(ind//2, ind//2 + 1498)) # в дыух файлах тогда возникли ошибки

    train_indices = list(set(range(len(files)//2)) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    batch_size = 1
    dataset = LargeDataset(folder_path, target_path)

    train_loader = DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                                             sampler=val_sampler)


    batch_train_indices, batch_val_indices = [], []

    Ab, Xb = None, None # A и Х батча
    batch, areab, delayb = torch.tensor([]), [], []
    c, ind, indf = 0, 0, 0
    for data in train_loader:          # создаем батчи
        A, X, area, delay = data.A, data.X, data.area, data.delay

#         G = nx.DiGraph()
#         G.add_nodes_from(range(len(X)))
#         G.add_edges_from(A.T.numpy())                                            # добавление центральносстей в качестве признаков
#         dc = torch.tensor(list(nx.degree_centrality(G).items()))[:,1][:,None]
#         pr = torch.tensor(list(nx.pagerank(G).items()))[:,1][:,None]

        X_enc = F.one_hot(X[:,0].type(torch.int64))
        X_enc = torch.cat((X_enc, X[:,1][:,None]), dim=-1)
        X = X_enc
        batch = torch.cat((batch, torch.tensor([ind]*X.shape[0]).type(torch.int32)))
        areab.append(area)
        delayb.append(delay)
        if Ab != None:
            Ab = torch.cat((Ab, A+Xb.shape[0]), dim=1) # конкатенируем список ребер одного графа, прибавляя к нему кол-во вершин
            Xb = torch.cat((Xb, X), dim=0)
        else:
            Ab, Xb = A, X
        c += 1
        ind += 1

        if c%100 == 0:
            print(f'{c}/{len(train_loader) + len(val_loader)}')
            print(Ab.shape, Xb.shape, Ab.max())

        if c%n_batch == 0:
            batch_train_indices.append(indf)
            torch.save(Ab, f'{batch_path}/A_{indf}.pt')
            torch.save(Xb, f'{batch_path}/X_{indf}.pt')
            torch.save(batch, f'{batch_path}/batch_{indf}.pt')
            torch.save(torch.tensor(areab), f'{batch_path}/area_{indf}.pt')
            torch.save(torch.tensor(delayb), f'{batch_path}/delay_{indf}.pt')
            indf += 1
            ind = 0
            Ab, Xb = None, None
            batch, areab, delayb = torch.tensor([]), [], []

    if Ab != None:  # если после цикла остались необработанные графы (их кол-во не делится на n_batch)
        batch_train_indices.append(indf)
        torch.save(Ab, f'{batch_path}/A_{indf}.pt')
        torch.save(Xb, f'{batch_path}/X_{indf}.pt')
        torch.save(batch, f'{batch_path}/batch_{indf}.pt')
        torch.save(torch.tensor(areab), f'{batch_path}/area_{indf}.pt')
        torch.save(torch.tensor(delayb), f'{batch_path}/delay_{indf}.pt')
        indf += 1

    # То же самое для валидации
    
    Ab, Xb = None, None
    batch, areab, delayb = torch.tensor([]), [], []
    ind = 0
    for data in val_loader:
        A, X, area, delay = data.A, data.X, data.area, data.delay

#         G = nx.DiGraph()
#         G.add_nodes_from(range(len(X)))
#         G.add_edges_from(A.T.numpy()) 
#         dc = torch.tensor(list(nx.degree_centrality(G).items()))[:,1][:,None]
#         pr = torch.tensor(list(nx.pagerank(G).items()))[:,1][:,None]

        X_enc = F.one_hot(X[:,0].type(torch.int64))
        X_enc = torch.cat((X_enc, X[:,1][:,None]), dim=-1)
        X = X_enc
        batch = torch.cat((batch, torch.tensor([ind]*X.shape[0]).type(torch.int32)))
        areab.append(area)
        delayb.append(delay)
        if Ab != None:
            Ab = torch.cat((Ab, A+Xb.shape[0]), dim=1)
            Xb = torch.cat((Xb, X), dim=0)
        else:
            Ab, Xb = A, X
        c += 1
        ind += 1

        if c%100 == 0:
            print(f'{c}/{len(train_loader) + len(val_loader)}')
            print(Ab.shape, Xb.shape, Ab.max())

        if c%n_batch == 0:
            batch_val_indices.append(indf)
            torch.save(Ab, f'{batch_path}/A_{indf}.pt')
            torch.save(Xb, f'{batch_path}/X_{indf}.pt')
            torch.save(batch, f'{batch_path}/batch_{indf}.pt')
            torch.save(torch.tensor(areab), f'{batch_path}/area_{indf}.pt')
            torch.save(torch.tensor(delayb), f'{batch_path}/delay_{indf}.pt')
            indf += 1
            ind = 0
            Ab, Xb = None, None
            batch, areab, delayb = torch.tensor([]), [], []

    if Ab != None:      
        batch_val_indices.append(indf)
        torch.save(Ab, f'{batch_path}/A_{indf}.pt')
        torch.save(Xb, f'{batch_path}/X_{indf}.pt')
        torch.save(batch, f'{batch_path}/batch_{indf}.pt')
        torch.save(torch.tensor(areab), f'{batch_path}/area_{indf}.pt')
        torch.save(torch.tensor(delayb), f'{batch_path}/delay_{indf}.pt')
        indf += 1

    print(batch_train_indices, batch_val_indices)

    
if __name__ == "__main__":
    folder_path, target_path, batch_path, n_batch = sys.argv[1:]
    main(folder_path, target_path, batch_path, int(n_batch))
    