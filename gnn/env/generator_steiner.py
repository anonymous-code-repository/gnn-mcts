import os
import os.path as osp
import torch
import numpy as np
#from env.graph import BaseGraph
from env.graph_steiner import BaseGraph
from env.simulator_steiner import TspSimulator
from tqdm import tqdm


class DataGenerator:
    def __init__(self, config):
        super(DataGenerator, self).__init__()
        self.config = config

        # Data file path
        self.points_dir = os.path.join('data', self.config['data_loader']['data']['graph_type'],
                                       'st{}'.format(self.config['arch']['args']['graph_size']),
                                       'st{}_train.txt'.format(self.config['arch']['args']['graph_size']))
        self.steiner_dir = os.path.join('data', self.config['data_loader']['data']['graph_type'],
                                      'st{}'.format(self.config['arch']['args']['graph_size']),
                                      'st{}_train_steiner_nodes.txt'.format(self.config['arch']['args']['graph_size']))
        self.terminal_dir = os.path.join('data', self.config['data_loader']['data']['graph_type'],
                                      'st{}'.format(self.config['arch']['args']['graph_size']),
                                      'st{}_train_terminals.txt'.format(self.config['arch']['args']['graph_size']))
        self.edge_dir = os.path.join('data', self.config['data_loader']['data']['graph_type'], 
                                      'st{}'.format(self.config['arch']['args']['graph_size']),
                                      'st{}_train_edges.txt'.format(self.config['arch']['args']['graph_size']))
        self.graph_type = self.config['arch']['args']['graph_generator'];

    def generate_data(self, graph, steiner_points, terminals):

        simulator = TspSimulator(self.config, graph, steiner_points, terminals)
        data = simulator.play()

        return data

    def run(self, dir):
        print("inside run")
        graphs, steiner_points, terminals = self.load_data()
        i = 0
        for (graph, steiner_points, terminals) in tqdm(zip(graphs, steiner_points, terminals), total=len(graphs), desc='Generate Data'):
            print(graph, steiner_points, terminals)
            data = self.generate_data(graph, steiner_points, terminals)
            for d in data:
                torch.save(d, osp.join(dir, 'data_{}.pt'.format(i)))
                i += 1

    def load_data(self):
        print("Inside load data...")
        graphs = []
        steiner_arr = []
        terminal_arr = []
        vertex_number = self.config['arch']['args']['graph_size']
        steiner_content = None
        if os.path.exists(self.steiner_dir):
            #steiner_content = np.genfromtxt(self.steiner_dir)
            f = open(self.steiner_dir, 'r')
            steiner_content = f.read().split("\n")
            for i, l in enumerate(steiner_content):
              arr = l.split()
              steiner_content[i] = arr
            f.close()
            steiner_content = steiner_content[:-1]
        terminal_content = None
        if os.path.exists(self.terminal_dir):
            f = open(self.terminal_dir, 'r')
            terminal_content = f.read().split("\n")
            for i, l in enumerate(terminal_content):
              arr = l.split()
              terminal_content[i] = arr
            f.close()
            terminal_content = terminal_content[:-1]

        if os.path.exists(self.edge_dir):
            f = open(self.edge_dir, 'r')
            file_content_edges = f.read().split("\n")
            for i, l in enumerate(file_content_edges):
              arr = l.split()
              file_content_edges[i] = arr
            f.close()
            file_content_edges = file_content_edges[:-1]


        if os.path.exists(self.points_dir):
            #file_content = np.genfromtxt(self.points_dir)
            f = open(self.points_dir, 'r')
            file_content = f.read().split("\n")
            for i, l in enumerate(file_content):
              arr = l.split()
              file_content[i] = arr
            f.close()
            file_content = file_content[:-1]
            #points_content = np.delete(file_content, [vertex_number * 2, vertex_number * 3 + 1], axis=1)
            #if steiner_content is None:
            #    path_content = points_content[:, vertex_number * 2:]
            # path_content = np.genfromtxt(self.paths_dir)

            for idx, c in enumerate(tqdm(file_content, desc='Load Graph...')):
                vertex_coordinate = c[0:vertex_number * 2]
                vertex_coordinate = [float(i) for i in vertex_coordinate]
                vertex_coordinate = np.array(vertex_coordinate).reshape(vertex_number, 2)
                steiner_points = [int(i) for i in steiner_content[idx]]
                steiner_points = [int(i) - 1 for i in steiner_points]
                terminal_points = [int(i) for i in terminal_content[idx]]
                terminal_points = [int(i) - 1 for i in terminal_points]

                print("vertex_coordinate:", vertex_coordinate)
                print("steiner_points:", steiner_points)
                print("terminal_points:", terminal_points)

                edge_list = file_content_edges[idx]
                i = 0
                edges = []
                while(i<len(edge_list)):
                  e = [int(edge_list[i]), int(edge_list[i+1]), float(edge_list[i+2])]
                  edges.append(e)
                  i = i + 3

                g = BaseGraph(vertex_number, vertex_coordinate, self.graph_type, edges)
                g.init_graph(self.config['data_loader']['data']['knn'])
                graphs.append(g)
                steiner_arr.append(steiner_points)
                terminal_arr.append(terminal_points)

                if len(graphs) == self.config['data_loader']['data']['graph_num']:
                    break

        return graphs, steiner_arr, terminal_arr
