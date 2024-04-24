import networkx as nx
from joblib import load
from igraph import read
from pyintergraph import igraph2nx
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def compute_area_delay(file):
    G = igraph2nx(read(file))
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)
    EDGES = len(edges)
    NOT = sum([i[2]['edge_type'] == 1 for i in edges]) 
    BUFF = EDGES - NOT
    AND, PO, PI = 0, 0, 0
    for i in nodes:
        if i[1]['node_type'] == 0:
            PI += 1
        elif i[1]['node_type'] == 1:
            PO += 1
        else:
            AND += 1
    LP = nx.dag_longest_path_length(G) 
    degree_assortativity_coefficient = nx.degree_assortativity_coefficient(G) 
    number_weakly_connected_components = nx.number_weakly_connected_components(G)
    s_metric = nx.algorithms.s_metric(G, normalized=False) 
    density = nx.density(G)

    X = [[BUFF, NOT, AND, PI, PO, LP, degree_assortativity_coefficient, \
    number_weakly_connected_components, s_metric, density, EDGES]]
    
    area_model = load('area_model.joblib')
    delay_model = load('delay_model.joblib')
    scaler = load('scaler.joblib')
    
    X = scaler.transform(X)
    
    return int(area_model.predict(X)), int(delay_model.predict(X))
