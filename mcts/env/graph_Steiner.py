import torch
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import networkx as nx
import math
import copy

class Graph:
    #def __init__(self, ver_num, ver_coo, k_num=10):
    #def __init__(self, ver_num, ver_coo, k_num=4):
    #def __init__(self, ver_num, ver_coo, edge_list, terminals, k_num=14):
    #def __init__(self, ver_num, ver_coo, edge_list, terminals, steiner_nodes, k_num=19):
    def __init__(self, ver_num, ver_coo, edge_list, terminals, steiner_nodes, graph_generator, k_num=19):
        self.ver_num = ver_num
        self.ver_coo = ver_coo
        self.edge_list = edge_list
        self.G = nx.Graph()
        for u, v, w in edge_list:
          self.G.add_edge(u, v, weight=w)
        self.k_num = k_num
        self.T = terminals
        self.steiner_nodes = [int(i) for i in steiner_nodes]
        self.terminal_indicator = np.zeros(self.ver_num)
        for v in self.T:
          self.terminal_indicator[v-1] = 1
        self.graph_generator = graph_generator

    def init(self):
        self.edge_index, self.knn_mat = self.cal_k_neighbor(self.k_num)

    def cal_dis_mat(self, vertex_i, vertex_j):
        return np.linalg.norm(self.ver_coo[vertex_i] - self.ver_coo[vertex_j])

    def cal_k_neighbor(self, k_n):
        source_ver = []
        target_ver = []
        knn_mat = np.zeros((self.ver_num, k_n))
        if self.graph_generator == 'GE':
          self.dis_mat = squareform(pdist(self.ver_coo, metric='euclidean'))
        else:
          distA = [[] for _ in range(self.ver_num)]
          for u in range(self.ver_num):
            sp_arr = nx.single_source_dijkstra(self.G, u)
            for i in range(self.ver_num):
              distA[u].append(sp_arr[0][i])
          self.dis_mat = np.array(distA)

        value_idx = torch.topk(torch.tensor(self.dis_mat, dtype=torch.float), k_n + 1, largest=False)
        values, indices = value_idx[0].numpy(), value_idx[1].numpy()

        for i in range(self.ver_num):
            source_ver.extend([i for _ in range(k_n)])
            target_ver.extend(indices[i][1:])
            knn_mat[i] = values[i][1:]

        return [source_ver, target_ver], knn_mat

    def contains_terminals(self, C, T):
      for t in T:
        if (t-1) not in C.nodes():
          return False
      return True

    def non_terminal_leaf_nodes(self, tree):
      leaf_nodes = [x for x in tree.nodes() if tree.degree(x)==1]
      non_terminals = []
      for l in leaf_nodes:
          if (l+1) not in self.T:
            non_terminals.append(l)
      return non_terminals

    def prune_non_terminals(self, tree):
      non_terminals = self.non_terminal_leaf_nodes(tree)
      while len(non_terminals)>0:
        tree.remove_nodes_from(non_terminals)
        non_terminals = self.non_terminal_leaf_nodes(tree)
      return tree

    def InclusiveTerminals(self,TT):
        M=len(TT)
        Tm = list(TT)
        for m in range(1,M):
            Tm[m] = Tm[m-1]+TT[m]
        return(tuple(Tm))

    def Kruskal(self,G):

        MST=nx.create_empty_copy(G); # MST(G)
        N=nx.number_of_nodes(G)
        E=nx.number_of_edges(G)
        i=0; # counter for edges of G
        k=0; # counter for MST(G)

        edge_list = sorted(G.edges(data=True), key=lambda x:x[2]['weight'])

        while k<(N-1) and i<(E):
            e=edge_list[i];
            i+=1
            if not nx.has_path(MST,e[0],e[1]):
                MST.add_edge(e[0],e[1],weight=e[2]['weight'])
                k+=1

        return(MST)

    def SteinerTree(self,G,T):

        HG=nx.Graph()
        HG.add_nodes_from(T)  # Hyper graph with nodes T and edges with weight equal to distance
        n=len(T)

        for i in range(n):
            for j in range(i+1,n):
                HG.add_edge(T[i], T[j], weight=nx.shortest_path_length(G,T[i], T[j],'weight'))

        HG_MST = self.Kruskal(HG)

        G_ST=nx.Graph()
        for e in HG_MST.edges(data=False):
            P=nx.shortest_path(G,e[0],e[1],'weight')
            #print(P)
            #G_ST.add_path(P)
            for i in range(1, len(P)):
              u = P[i-1]
              v = P[i]
              G_ST.add_edge(u, v, weight=G[u][v]["weight"])

        # find the minimum spanning tree of the resultant graph

        return(G_ST)

    def PruneBranches(self,G,T):
        has_one=False
        for v in G.nodes(data=False):
            if (v not in T) and (G.degree(v)==1):
                has_one=True
                G.remove_edge(v,*G.neighbors(v))
        if has_one:
            self.PruneBranches(G,T)

    def MLST_TOP(self,G,TT):

        M=len(TT);
        Tm= self.InclusiveTerminals(TT)

        G_ST_TOP=[None]*M
        G_ST_TOP[0] = self.SteinerTree(G,Tm[0])
        #G_ST_TOP[0] = SteinerTree2Approx(G,Tm[0])
        #print(G_ST_TOP[0].edges())

        for m in range(1,M):
            G2=copy.deepcopy(G)
            for e in G_ST_TOP[m-1].edges(data=True):
                #nx.set_edge_attributes(G2, 'weight', {(e[0],e[1]):0})
                nx.set_edge_attributes(G2, {(e[0],e[1]):{'weight':0}})
            #G_ST_TOP[m] = SteinerTree2Approx(G2,Tm[m])
            G_ST_TOP[m] = self.SteinerTree(G2,Tm[m])

        return(tuple(G_ST_TOP))

    def MLST_QoS(self,G,TT):
        M= len(TT)
        q=math.ceil(math.log2(M))
        TR=[]; TTR=[]
        Tm= self.InclusiveTerminals(TT)
        for l in range(M):
            qp=math.ceil(math.log2(M-l))
            if qp<q:
                TTR.append(TR)
                q=qp
                TR=[]
            TR=TR+TT[l]

        TTR.append(TR)
        #print(TTR)

        STq=self.MLST_TOP(G,TTR)

        G_ST_QoS = [None]*M
        q=math.ceil(math.log2(M))
        Tdummy=TTR[0]; k=0;
        for l in range(M):
            qp=math.ceil(math.log2(M-l))
            if qp<q:
                k+=1
                q=qp

            if l==0:
                STdummy=copy.deepcopy(STq[0])
                self.PruneBranches(STdummy,Tm[0])
                G_ST_QoS[0]=copy.deepcopy(STdummy)
            else:
                STdummy=nx.create_empty_copy(G)
                STdummy.add_edges_from(STq[k].edges(),weight=2)
                STdummy.add_edges_from(G_ST_QoS[l-1].edges(),weight=1)
                STdummy=self.Kruskal(STdummy)
                self.PruneBranches(STdummy,Tm[l])
                G_ST_QoS[l]=copy.deepcopy(STdummy);


        return(tuple(G_ST_QoS))

    def MLST_Costs(self,G,G_MLST):
        M=len(G_MLST)
        C=[0]*M
        for m in range(M):
            #print(G_MLST[m].edges(data=False))
            #print_edges(G_MLST[m])
            for e in G_MLST[m].edges(data=False):
                C[m]+=G.get_edge_data(e[0],e[1])['weight']
                #print(str(e[0])+","+str(e[1])+":"+str(G.get_edge_data(e[0],e[1])['weight']))
        #print(C)
        #print(sum(C))
        return(C,sum(C))

    def compute_path_len(self, path):
        Ts = [[(t-1) for t in self.T]]
        TT = [Ts[0]]
        for i in range(1, len(Ts)):
          TT.append(list(set(Ts[i])-set(Ts[i-1])))
        min_cost = 1000000000000.0
        for i in range(1, len(path)+1):
          H = self.G.subgraph(path[:i])
          cost = 0.0
          for S in nx.connected_components(H):
            C = self.G.subgraph(S)
            if self.contains_terminals(C, self.T):
              for u, v in C.edges():
                C[u][v]['weight'] = self.G[u][v]['weight']
              T=nx.minimum_spanning_tree(C)
              T = self.prune_non_terminals(T)
              for u, v in T.edges():
                cost += self.G[u][v]['weight']
          if cost>0:
            min_cost = min(min_cost, cost)
          QOS = self.MLST_QoS(self.G,TT)
          min_cost = min(min_cost, self.MLST_Costs(self.G,QOS)[1])
        return min_cost


