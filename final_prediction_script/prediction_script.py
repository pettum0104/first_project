import sys
sys.path.append('models')

import pathlib
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from igraph import read
from pyintergraph import igraph2nx
import torch.nn.functional as F
import torch
import scipy
from encoder import *
from joblib import load
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import gc


def predict(file):
    graph_graphml = igraph2nx(read(file))  # получили граф NetworkX

    # инициализация списка параметров графа
    G_related_row = [graph_graphml.number_of_nodes(), graph_graphml.number_of_edges(), nx.density(graph_graphml),
                    nx.number_connected_components(nx.to_undirected(graph_graphml)),
                    nx.degree_assortativity_coefficient(graph_graphml)]

    # корреляция Пирсона определяется двумя параметрами
    # для расчёта используется готовая функция pearsonr из scipy.stats
    left_degree = [graph_graphml.degree[P] for (P, Q) in graph_graphml.edges()]
    right_degree = [graph_graphml.degree[Q] for (P, Q) in graph_graphml.edges()]
    corr_coefficient = pearsonr(left_degree, right_degree)
    G_related_row.append(corr_coefficient[0])
    G_related_row.append(corr_coefficient[1])

    # статистика по степеням вершин
    degrees = [d for n, d in graph_graphml.degree()]

    G_related_row.append(np.mean(degrees))  # средняя степень вершины
    G_related_row.append(np.std(degrees))  # стандартное отклонение
    G_related_row.append(np.max(degrees))  # максимальная степень вершины
    G_related_row.append(np.min(degrees))  # минимальная степень вершины

    # статистика по мерам близости вершин
    closeness = list(nx.closeness_centrality(graph_graphml).values())

    G_related_row.append(np.mean(closeness))  # средняя степень вершины
    G_related_row.append(np.std(closeness))  # стандартное отклонение
    G_related_row.append(np.max(closeness))  # максимальная степень вершины
    G_related_row.append(np.min(closeness))  # минимальная степень вершины

    # статистика по гармонической центральности вершин
    harmonic = list(nx.harmonic_centrality(graph_graphml).values())

    G_related_row.append(np.mean(harmonic))  # средняя степень вершины
    G_related_row.append(np.std(harmonic))  # стандартное отклонение
    G_related_row.append(np.max(harmonic))  # максимальная степень вершины
    G_related_row.append(np.min(harmonic))  # минимальная степень вершины

    # статистика по степенной центральности вершин
    degree_centrality = list(nx.degree_centrality(graph_graphml).values())

    G_related_row.append(np.mean(degree_centrality))  # средняя степень вершины
    G_related_row.append(np.std(degree_centrality))  # стандартное отклонение
    G_related_row.append(np.max(degree_centrality))  # максимальная степень вершины
    G_related_row.append(np.min(degree_centrality))  # минимальная степень вершины

    edges = graph_graphml.edges(data=True)
    nodes = graph_graphml.nodes(data=True)
    EDGES = len(edges)
    NOT = sum([i[2]['edge_type'] == 1 for i in edges]) # кол-во вершин НЕ
    BUFF = EDGES - NOT # кол-во вершин буфферов
    AND, PO, PI = 0, 0, 0
    for i in nodes:
        if i[1]['node_type'] == 0:
            PI += 1 # количество входов
        elif i[1]['node_type'] == 1:
            PO += 1 # кол-во выходов
        else:
            AND += 1 # кол-во вершин И
    LP = nx.dag_longest_path_length(graph_graphml) # глубина графа

    G_related_row += [BUFF, NOT, AND, PI, PO, LP]

    X_f = np.array(G_related_row)
    
    # получение матрицы смежности и призкаон вершин
    A = nx.adjacency_matrix(graph_graphml)
    sparse_matrix = scipy.sparse.coo_matrix(A)
    row, col = sparse_matrix.nonzero()
    A = torch.tensor(np.column_stack((row, col)).T)
    X = torch.tensor([[i['node_type'], i['num_inverted_predecessors']] for i in np.array(nodes)[:,1]])
    X_enc = F.one_hot(X[:,0].type(torch.int64))
    X = torch.cat((X_enc, X[:,1][:,None]), dim=-1)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
    else:
        device = torch.device("cpu")

    # получение эмбеддинга
    encoder = Encoder(input_size = 4, hidden_channels=128, conv=GCNConv, n_output=256)
    encoder.load_state_dict(torch.load('models/encoder.pt'))
    encoder.type(torch.cuda.FloatTensor).to(device)

    X_gpu, A_gpu = X.to(device), A.to(device)
    X_emb = encoder(X_gpu, A_gpu).detach().cpu().numpy()
    del X_gpu, A_gpu, encoder
    torch.cuda.empty_cache()
    gc.collect()
    
    x = np.hstack((X_f[None,:], X_emb))
    scaler = load('models/final_scaler.joblib')
    x = scaler.transform(x)
    
    model = load('models/final_model.joblib')
    area, delay = model.predict(x)[0]
    
    return area, delay
