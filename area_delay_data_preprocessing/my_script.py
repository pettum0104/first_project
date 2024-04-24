import sys
import os
import csv
import networkx as nx
from networkx.algorithms import approximation as apxa


def features(file):
    G = nx.read_graphml(file)
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)

    sid = int(file.split('_')[-2][3:])

    NOT = sum([i[2]['edge_type'] == 1 for i in edges]) # кол-во вершин НЕ
    BUFF = sum([i[2]['edge_type'] == 0 for i in edges]) # кол-во вершин буфферов
    AND = sum([i[1]['node_type'] == 2 for i in nodes]) # кол-во вершин И
    PO = sum([i[1]['node_type'] == 1 for i in nodes]) # кол-во выходов
    PI = sum([i[1]['node_type'] == 0 for i in nodes]) # количество входов
    LP = nx.dag_longest_path_length(G) # глубина графа
    LEN = len(G) # кол-во узлов в графе
    degree_assortativity_coefficient = nx.degree_assortativity_coefficient(G) # измеряет тенденцию вершин с высокой степенью 
    # соединяться с другими вершинами с высокой степенью (положительное значение коэффициента) или с вершинами с низкой степенью 
    # (отрицательное значение коэффициента)

    transitivity = nx.transitivity(G) # измеряет вероятность того, что две вершины, связанные с общей третьей вершиной, будут связаны друг с другом

    number_weakly_connected_components = nx.number_weakly_connected_components(G)

    s_metric = nx.algorithms.s_metric(G, normalized=False) # сумма произведений степеней вершин для каждого ребра в графе

    density = nx.density(G) # отношение числа ребер в графе к максимально возможному числу ребер. Он показывает, насколько граф плотно связан

    EDGES = nx.number_of_edges(G) # кол-во ребер

    return [sid, BUFF, NOT, AND, PI, PO, LP, LEN, degree_assortativity_coefficient, \
    transitivity, number_weakly_connected_components, \
    s_metric, density, EDGES]

def create_csv(folder_path):
    w_file = open(os.path.split(folder_path)[-1]+"_my.csv", 'w', newline='')
    file_writer = csv.writer(w_file)
    file_writer.writerow(["sid", "BUFF", "NOT", "AND", "PI", "PO", "LP", "LEN", \
        "degree_assortativity_coefficient", \
        "transitivity", "number_weakly_connected_components", \
        "s_metric", "density", "EDGES"])

    for file in os.listdir(folder_path):
        if 'graphml' in file:
            try:
                print(file)
                file_writer.writerow(features(os.path.join(folder_path, file)))
            except:
                print(file+'\n', file=open('output.txt', 'a'))


if __name__ == "__main__":
    folder_path = sys.argv[1]
    create_csv(folder_path)
    sys.exit(0)
    