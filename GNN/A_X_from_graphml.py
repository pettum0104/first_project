import threading
import numpy as np
import os
import networkx as nx
import scipy

FOLDER_PATH = 'D:/data' # путь к папке с файлами graphml
TARGET_PATH = 'D:/matrix' # путь к папке, куда складывать списки ребер и признаков

def thread_function(l):
    for folder in l:
        folder = os.path.join(folder_path, folder) 
        for file in os.listdir(folder):
            if 'graphml' in file:
                try:
                    G = nx.read_graphml(os.path.join(folder, file))
                    file = file.split('.')[0]
                    A = nx.adjacency_matrix(G)
                    scipy.sparse.save_npz(f'{target_path}/{file}_A.npz', A)
                    X = np.array([[i['node_type'], i['num_inverted_predecessors']] for i in np.array(G.nodes(data=True))[:,1]])
                    np.save(f'{target_path}/{file}_X', X)
                    print(file)
                except:
                    print(file+'\n', file=open('output.txt', 'a'))

def main():
    threads = []
    n_threads = 12
    folder_path = FOLDER_PATH
    target_path = TARGET_PATH
    lists = np.array_split(os.listdir(folder_path), n)

    for i in range(n_threads):
        thread = threading.Thread(target=thread_function, args=(lists[i],))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All threads have finished")

if __name__ == "__main__":
    main()
    