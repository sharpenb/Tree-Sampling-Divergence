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

### Random dendrogram

def random_dendrogram(number_nodes = 100):
    nodes = list(range(number_nodes))
    dendrogram = []
    t = 0
    size = {u: 1 for u in nodes}
    while (len(nodes)) > 1:
        u = nodes.pop(np.random.randint(len(nodes)))
        v = nodes.pop(np.random.randint(len(nodes)))
        new_node = number_nodes + t
        t += 1
        size[new_node] = size.pop(u) + size.pop(v)
        dendrogram.append([u,v,size[new_node],size[new_node]])
        nodes.append(new_node)
    return np.array(dendrogram, float)

def get_similarity(dendrogram):
    n = np.shape(dendrogram)[0] + 1
    sim = np.zeros((n,n),float)
    cluster = {u:[u] for u in range(n)}
    for t in range(n - 1):
        u = int(dendrogram[t][0])
        v = int(dendrogram[t][1])
        for i in cluster[u]:
            for j in cluster[v]:
                sim[i][j] = 1 / dendrogram[t][2]
        cluster[n + t] = cluster.pop(u) + cluster.pop(v)
    return sim

def generate_hierarchical_graph(dendrogram, average_degree):
    n = np.shape(dendrogram)[0] + 1
    similarity = get_similarity(dendrogram)
    is_connected = False
    while not is_connected:
        adjacency = np.random.rand(n,n) < similarity / np.sum(similarity) * n * average_degree / 2
        adjacency = np.array(adjacency + adjacency.T,int)
        graph = nx.from_numpy_matrix(adjacency)
        is_connected = nx.is_connected(graph)
    return graph

def add_noise(graph, prob):
    is_connected = False
    while not is_connected:
        new_graph = graph.copy()
        edges = list(graph.edges())
        indices = np.random.choice(list(range(len(edges))),replace = False, size = int(np.floor(prob * len(edges))))
        for i in indices:
            u,v = edges[i]
            new_graph.remove_edge(u,v)
            new_edge = np.random.choice(list(new_graph.nodes()), replace = False, size = 2)
            new_graph.add_edge(new_edge[0],new_edge[1],weight = 1.)
        is_connected = nx.is_connected(new_graph)
    return new_graph

