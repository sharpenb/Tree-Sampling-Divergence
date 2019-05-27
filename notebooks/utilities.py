import numpy as np
import networkx as nx
import random as rnd

### Rank score (reconstruction score)

def rank_score(graph, score):
    rank_score = []
    for u, v in graph.edges():
        
        rank_u = np.argsort(-score[u,:])
        for r, w in enumerate(rank_u):
            if w in graph.neighbors(u):
                rank_score.append(r)
        
        rank_v = np.argsort(-score[v,:])
        for r, w in enumerate(rank_v):
            if w in graph.neighbors(v):
                rank_score.append(r)
                
    return np.mean(rank_score)

### Shuffle graph edges

def shuffle_edges(graph, p):
    graph_copy = graph.copy()
    for u, v in list(graph_copy.edges()):
        if np.random.choice([False, True], p=[p, 1 - p]):
            u_, v_ = np.random.choice(graph_copy.nodes(), 2, replace=False)
            graph_copy.add_edge(u_, v_, weight=graph_copy[u][v]['weight'])
            graph_copy.remove_edge(u, v)
    return graph_copy

### Hierarchical SBM

class HSBM:
    def __init__(self, nodes, probability=1.):
        self._nodes = nodes
        self.next_level = []
        self._p_matrix = probability * np.ones((1, 1))

    @staticmethod
    def balanced(n_levels, decay_factor, division_factor, core_community_size, p_in):
        nodes = range(division_factor**n_levels * core_community_size)
        hsbm = HSBM(nodes, probability=p_in)
        hsbm._balanced_recursive(n_levels, decay_factor, division_factor, core_community_size, p_in)
        return hsbm

    def _balanced_recursive(self, n_levels, decay_factor, division_factor, core_community_size, p_in):
        if n_levels > 0:
            community_size = len(self._nodes) / division_factor
            partition = division_factor * [community_size]
            p_matrix = p_in * decay_factor**(n_levels) * np.ones((division_factor, division_factor))
            np.fill_diagonal(p_matrix, p_in * np.ones(division_factor))
            self.divide_cluster(partition=partition, p_matrix=p_matrix)
            for cluster in self.next_level:
                cluster._balanced_recursive(n_levels - 1, decay_factor, division_factor, core_community_size, p_in)

    def divide_cluster(self, partition, p_matrix):
        partition.insert(0, 0)
        repartition = np.cumsum(np.array(partition))
        try:
            assert sum(partition) == len(self._nodes)
            self._p_matrix = p_matrix
            for i in range(len(partition) - 1):
                sub_cluster = HSBM(self._nodes[int(repartition[i]):int(repartition[i+1])], p_matrix[i][i])
                self.next_level.append(sub_cluster)
        except AssertionError:
            print("\nSum partition (", sum(partition), ") != Number of nodes (", len(self._nodes), ")")

    def create_graph(self, distribution='Binomial'):
        G = nx.Graph()
        G.add_nodes_from(self._nodes)
        if distribution == 'Binomial':
            edges = self._create_edges_binomial()
        elif distribution == 'Poisson':
            edges = self._create_edges_poisson()
        G.add_weighted_edges_from(edges)
        return G

    def _create_edges_binomial(self):
        edges = []
        if len(self.next_level) == 0:
            for u in self._nodes:
                for v in range(self._nodes[0], u):
                    if rnd.random() < self._p_matrix[0][0]:
                        edges.append((u, v, 1.))
        else:
            for i in range(0, len(self.next_level)):
                new_edges = self.next_level[i]._create_edges_binomial()

                # Add new edges
                edges = edges + new_edges

                # Add intra-edges
                C_i = self.next_level[i]._nodes
                for j in range(0,i):
                    C_j = self.next_level[j]._nodes
                    for u in C_i:
                        for v in C_j:
                            if rnd.random() < self._p_matrix[i][j]:
                                edges.append((u, v, 1.))
        return edges

    def _create_edges_poisson(self):
        edges = []
        if len(self.next_level) == 0:
            for u in self._nodes:
                for v in range(self._nodes[0], u):
                    weight = np.random.poisson(self._p_matrix[0][0])
                    if weight > 0:
                        edges.append((u, v, weight))
        else:
            for i in range(0, len(self.next_level)):
                new_edges = self.next_level[i]._create_edges_poisson()

                # Add new edges
                edges = edges + new_edges

                # Add intra-edges
                C_i = self.next_level[i]._nodes
                for j in range(0,i):
                    C_j = self.next_level[j]._nodes
                    for u in C_i:
                        for v in C_j:
                            weight = np.random.poisson(self._p_matrix[i][j])
                            if weight > 0:
                                edges.append((u, v, weight))
        return edges

    def clusters_at_level(self, level):
        clusters = []
        if level == 0:
            clusters.append(self._nodes)
        else:
            if len(self.next_level) > 0:
                for i in range(0, len(self.next_level)):
                    clusters = clusters + self.next_level[i].clusters_at_level(level - 1)
            else:
                clusters.append(self._nodes)
        return clusters

    def info(self):
        print('\nNodes: ', self._nodes)
        print('Next level: ', self.next_level)
        print('Probability matrix: ', self._p_matrix)

### Obtain clustering from dendrogram

def clustering_from_dendrogram(D, n_clusters):
    n_nodes = np.shape(D)[0] + 1
    cluster = {i: [i] for i in range(n_nodes)}
    for t in range(n_clusters):
        i = int(D[t][0])
        j = int(D[t][1])
        cluster[n_nodes + t] = cluster.pop(i) + cluster.pop(j)
    clusters = [cluster[c] for c in cluster]
    return clusters        
        
### Load dataset

def load_dataset(directory, dataset_name):
    try:
        file = open(directory + dataset_name + "/type.txt", "r")
        graph_type = file.readline()[0:-1]
        file.close()
        if graph_type == "DW":
            G = nx.read_weighted_edgelist(directory + dataset_name + "/edge.txt", nodetype=int, create_using=nx.DiGraph())
        elif graph_type == "UW":
            G = nx.read_weighted_edgelist(directory + dataset_name + "/edge.txt", nodetype=int)
        elif graph_type == "DU":
            G = nx.read_edgelist(directory + dataset_name + "/edge.txt", nodetype=int, create_using=nx.DiGraph())
            for (u, v) in G.edges():
                G.add_edge(u, v, weight=1.)
        else:
            G = nx.read_edgelist(directory + dataset_name + "/edge.txt", nodetype=int)
        G.name = dataset_name
        G = G.to_undirected()
    except:
        print("\nCannot load graph")
        G = nx.Graph()

    pos = {}
    try:
        file = open(directory + dataset_name + "/position.txt", "r")
        u = 0
        for line in file:
            s = line.split()
            pos[u] = (float(s[0]), float(s[1]))
            G.add_node(u)
            u += 1
        file.close()
        G.add_nodes_from(pos.keys())
    except:
        print("\nCannot load positions")

    label = {}
    try:
        file = open(directory + dataset_name + "/label.txt", "r")
        u = 0
        for line in file:
            label[u] = line[0:-1]
            G.add_node(u)
            u += 1
        file.close()
    except:
        print("\nCannot load labels")

    return G, pos, label