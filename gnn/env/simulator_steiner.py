import torch
import numpy as np
from env.graph import BaseGraph
from torch_geometric.data import Data
import itertools
import random


class TspSimulator:
    def __init__(self, config, graph: BaseGraph, steiner_points, terminals):
        self.config = config
        self.graph = graph
        self.steiner_points = steiner_points
        self.terminals = terminals
        self.terminal_indicator = np.zeros(self.graph.ver_num)
        for v in terminals:
          self.terminal_indicator[v] = 1

    def init(self):
        #self.obj_path = []
        self.steiner_permutations = list(itertools.permutations(self.steiner_points))
        if len(self.steiner_permutations)>10:
          # randomly select some index
          ind_list = []
          for i in range(10):
            ind_list.append(random.choice(list(range(len(self.steiner_permutations)))))
          self.steiner_permutations = [self.steiner_permutations[i] for i in ind_list]
        self.obj_path = []
        for i in range(len(self.steiner_permutations)):
          self.obj_path.append([])
        self.visited = np.zeros((len(self.steiner_permutations), self.graph.ver_num))
        #print(self.visited.shape)

        self.data_list = []

    def move(self, i, vertex):
        self.visited[i][vertex] = 1
        self.obj_path[i].append(vertex)

    def play(self):
        self.init()

        #for idx, vertex in enumerate(self.opt_path):
        for id, perm in enumerate(self.steiner_permutations):
          #print("id, permutation:", id, perm)
          for vertex in self.terminals:
            #print("id, vertex:", id, vertex)
            self.move(id, vertex)
          for idx, vertex in enumerate(perm):
            #print("idx, vertex:", idx, vertex)
            #if idx == 0:
            #else:
            self.save_state(id, vertex)
            self.move(id, vertex)

        return self.data_list

    def save_state(self, i, move):
        #print(self.obj_path[i][-1])
        node_tag = np.zeros((self.graph.ver_num, self.config['arch']['args']['node_dim']), dtype=np.float)
        node_tag[:, 0] = self.visited[i]
        if self.graph.graph_type=='GE':
          node_tag[:, 1:3] = self.graph.ver_coor
        node_tag[:, 3:4] = 1  # [s_node:tag]
        if self.graph.graph_type=='GE':
          node_tag[:, 4:6] = \
            self.graph.ver_coor[self.obj_path[i][0]]  # [s_node:x,y]
        node_tag[:, 6] = self.terminal_indicator

        edge_tag = np.zeros(
            (self.graph.ver_num, self.config['data_loader']['data']['knn'], self.config['arch']['args']['edge_dim']),
            dtype=np.float)
        #print(edge_tag)
        #print(self.graph.knn_mat)
        edge_tag[:, :, 0] = self.graph.knn_mat
        #print(edge_tag)
        edge_tag[:, :, 1:2] = 1
        #print(edge_tag)
        if self.graph.graph_type=='GE':
          edge_tag[:, :, 2:4] = self.graph.ver_coor[self.obj_path[i][-1]]
        #print(edge_tag)

        node_tag = torch.tensor(node_tag, dtype=torch.float)
        edge_tag = torch.tensor(edge_tag, dtype=torch.float).view(-1, self.config['arch']['args']['edge_dim'])
        edge_index = torch.tensor(self.graph.edge_index, dtype=torch.long)
        y = torch.tensor([move], dtype=torch.long)
        data = Data(x=node_tag, edge_index=edge_index, edge_attr=edge_tag, y=y)
        self.data_list.append(data)
