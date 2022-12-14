import os
import json
import numpy as np
from env.graph_Steiner import Graph
from collections import OrderedDict
from datetime import datetime


'''
def load_data(dir, dir_terminals, graph_size):
    graphs = []
    if os.path.exists(dir):
        file_content = np.genfromtxt(dir)
        coordinates = np.delete(file_content, [graph_size * 2, graph_size * 3 + 1], axis=1)

    for idx, c in enumerate(coordinates):
        vertex_coordinate = c[0:graph_size * 2].reshape(graph_size, 2)
        g = Graph(graph_size, vertex_coordinate)
        g.init()
        graphs.append(g)

    return graphs
'''

#def load_data(dir, dir_edges, dir_terminals, graph_size):
def load_data(dir, dir_edges, dir_terminals, graph_size, graph_generator):
    print("Inside load data...")
    graphs = []
    edges = []

    if os.path.exists(dir):
        f = open(dir, 'r')
        file_content = f.read().split("\n")
        for i, l in enumerate(file_content):
          arr = l.split()
          file_content[i] = arr
        f.close()
        file_content = file_content[:-1]

    if os.path.exists(dir_edges):
        f = open(dir_edges, 'r')
        file_content_edges = f.read().split("\n")
        for i, l in enumerate(file_content_edges):
          arr = l.split()
          file_content_edges[i] = arr
        f.close()
        file_content_edges = file_content_edges[:-1]

    terminals = np.genfromtxt(dir_terminals, dtype=np.int32)

    vertex_number = graph_size
    for idx, c in enumerate(file_content):
        vertex_coordinate = c[0:vertex_number * 2]
        steiner_nodes = c[vertex_number * 2:]
        vertex_coordinate = [float(i) for i in vertex_coordinate]
        vertex_coordinate = np.array(vertex_coordinate).reshape(vertex_number, 2)
        edge_list = file_content_edges[idx]
        i = 0
        edges = []
        while(i<len(edge_list)):
          e = [int(edge_list[i]), int(edge_list[i+1]), float(edge_list[i+2])]
          edges.append(e)
          i = i + 3
        g = Graph(graph_size, vertex_coordinate, edges, terminals[idx], steiner_nodes, graph_generator)
        g.init()
        graphs.append(g)

    return graphs


def save_path(file_path, tour, path_len, episode):
    file = open(file_path, 'a')
    file.write(str(episode))
    file.write(' ')

    for vertex in tour:
        file.write(str(vertex))
        file.write(' ')

    file.write(str(path_len))
    file.write('\n')
    file.close()


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)



