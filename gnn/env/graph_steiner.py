import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import networkx as nx

class BaseGraph:
    def __init__(self, ver_num, ver_coor, graph_type, edge_list):
        self.ver_num = ver_num
        self.ver_coor = np.array(ver_coor)
        self.graph_type = graph_type
        self.edge_list = edge_list
        self.G = nx.Graph()
        for u, v, w in edge_list:
          self.G.add_edge(u, v, weight=w)

    def cal_dis_mat(self):
        if self.graph_type=='GE':
          distA = pdist(self.ver_coor, metric='euclidean')
          return squareform(distA)
        else:
          distA = [[] for _ in range(self.ver_num)]
          #for u in range(self.ver_num):
          for u in self.G.nodes():
            sp_arr = nx.single_source_dijkstra(self.G, u)
            #for i in range(self.ver_num):
            for i in self.G.nodes():
              distA[u].append(sp_arr[0][i])
          return np.array(distA)

    def init_graph(self, k_num):
        self.dis_mat = self.cal_dis_mat()
        self.edge_index, self.knn_mat = self.cal_k_neighbor(k_num)

    def cal_k_neighbor(self, k_n):
        source_ver = []
        target_ver = []
        knn_mat = np.zeros((self.ver_num, k_n))

        v_idx = torch.topk(torch.tensor(self.dis_mat, dtype=torch.float), k_n + 1, largest=False)
        values = v_idx[0].detach().numpy()
        indices = v_idx[1].detach().numpy()

        for i in range(self.ver_num):
            source_ver.extend([i for _ in range(k_n)])
            target_ver.extend(indices[i][1:])
            knn_mat[i] = values[i][1:]

        return [source_ver, target_ver], knn_mat
