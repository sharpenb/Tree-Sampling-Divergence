import networkx as nx
import numpy as np
import random as rnd
from sklearn.cluster import SpectralClustering

def clusters_dict2clusters_list(cluster_dict):
    i = 0
    cluster_index = {}
    cluster_list = []
    for u, c in cluster_dict.items():
        if c not in cluster_index:
            cluster_list.append([u])
            cluster_index[c] = i
            i += 1
        else:
            cluster_list[cluster_index[c]].append(u)
    return cluster_list


def clusters_list2clusters_dict(cluster_list):
    cluster_dict = {}
    for i, c in enumerate(cluster_list):
        for u in c:
            cluster_dict[u] = i
    return cluster_dict

### Clustering algorithms

def spectral_clustering(G, n_clusters=2):
    adj_mat = nx.to_numpy_matrix(G)
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    clusters = {}
    for i in range(len(sc.labels_)):
        if sc.labels_[i] not in clusters:
            clusters[sc.labels_[i]] = []
        clusters[sc.labels_[i]].append(i)
    return clusters.values()

def louvain(G, resolution=1, eps=0.001):
    clusters_dict = maximize(G, resolution, eps)
    n = len(clusters_dict)
    k = len(set(clusters_dict.values()))
    while k < n:
        H = aggregate(G, clusters_dict)
        new_cluster = maximize(H, resolution, eps)
        clusters_dict = {u: new_cluster[clusters_dict[u]] for u in G.nodes()}
        n = k
        k = len(set(clusters_dict.values()))
    return clusters_dict2clusters_list(clusters_dict)


def maximize(G, resolution, eps):
    # node weights
    node_weight = {u: 0. for u in G.nodes()}
    for (u, v) in G.edges():
        node_weight[u] += G[u][v]['weight']
        node_weight[v] += G[u][v]['weight']
    # total weight
    wtot = sum(list(node_weight.values()))
    # clusters
    cluster = {u: u for u in G.nodes()}
    # total weight of each cluster
    cluster_weight = {u: node_weight[u] for u in G.nodes()}
    # weights in each community to which the nodes are linked
    w = {u: {v: G[u][v]['weight'] for v in G.neighbors(u) if v != u} for u in G.nodes()}
    increase = True
    while increase:
        increase = False
        for u in G.nodes():
            # Compute delta for every neighbor
            delta = {}
            for k in w[u].keys():
                delta[k] = w[u][k] - resolution * node_weight[u] * cluster_weight[k] / wtot
            # Compute delta for u itself (if not already done)
            k = cluster[u]
            if k not in w[u].keys():
                delta[k] = - resolution * node_weight[u] * cluster_weight[k] / wtot
            # Compare the greatest delta to epsilon
            l = max(delta, key=delta.get)
            if delta[l] - delta[k] > resolution * (node_weight[u] * node_weight[u] / wtot) + eps / wtot:
                increase = True
                cluster[u] = l
                # Update information about neighbors and the community change of u
                cluster_weight[k] -= node_weight[u]
                cluster_weight[l] += node_weight[u]
                for v in G.neighbors(u):
                    if v != u:
                        w[v][k] -= G[u][v]['weight']
                        if w[v][k] == 0:
                            w[v].pop(k)
                        if l not in w[v].keys():
                            w[v][l] = 0
                        w[v][l] += G[u][v]['weight']
    return cluster


def aggregate(G, clusters_dict):
    H = nx.Graph()
    H.add_nodes_from(list(clusters_dict.values()))
    for (u,v) in G.edges():
        if H.has_edge(clusters_dict[u], clusters_dict[v]):
            H[clusters_dict[u]][clusters_dict[v]]['weight'] += G[u][v]['weight']
        else:
            H.add_edge(clusters_dict[u], clusters_dict[v])
            H[clusters_dict[u]][clusters_dict[v]]['weight'] = G[u][v]['weight']
    return H

### Agglomerative algorithms

import numpy as np
import networkx as nx


_AFFINITY = {'unitary', 'weighted'}
_LINKAGE = {'single', 'average', 'complete', 'modular'}


def agglomerative_clustering(graph, affinity='weighted', linkage='modular', f=lambda l: - np.log(l), check=True):

    if affinity not in _AFFINITY:
        raise ValueError("Unknown affinity type %s."
                         "Valid options are %s" % (affinity, _AFFINITY.keys()))

    if linkage not in _LINKAGE:
        raise ValueError("Unknown linkage type %s."
                         "Valid options are %s" % (linkage, _LINKAGE.keys()))

    graph_copy = graph.copy()

    if check:

        graph_copy = nx.convert_node_labels_to_integers(graph_copy)

        if affinity == 'unitary':
            for e in graph_copy.edges():
                graph_copy.add_edge(e[0], e[1], weight=1)

        n_edges = len(list(graph_copy.edges()))
        n_weighted_edges = len(nx.get_edge_attributes(graph_copy, 'weight'))
        if affinity == 'weighted' and not n_weighted_edges == n_edges:
            raise KeyError("%s edges among %s do not have the attribute/key \'weight\'."
                           % (n_edges - n_weighted_edges, n_edges))

    if linkage == 'single':
        dendrogram = single_linkage_hierarchy(graph_copy, f)
    elif linkage == 'average':
        dendrogram = average_linkage_hierarchy(graph_copy, f)
    elif linkage == 'complete':
        dendrogram = complete_linkage_hierarchy(graph_copy, f)
    elif linkage == 'modular':
        dendrogram = modular_linkage_hierarchy(graph_copy, f)

    return reorder_dendrogram(dendrogram)



def single_linkage_hierarchy(graph, f):
    remaining_nodes = set(graph.nodes())
    n_nodes = len(remaining_nodes)

    cluster_size = {u: 1 for u in range(n_nodes)}
    connected_components = []
    dendrogram = []
    u = n_nodes

    while n_nodes > 0:
        for new_node in remaining_nodes:
            chain = [new_node]
            break
        while chain:
            a = chain.pop()
            linkage_max = - float("inf")
            b = -1
            neighbors_a = list(graph.neighbors(a))
            for v in neighbors_a:
                if v != a:
                    linkage = float(graph[a][v]['weight'])
                    if linkage > linkage_max:
                        b = v
                        linkage_max = linkage
                    elif linkage == linkage_max:
                        b = min(b, v)
            linkage = linkage_max
            if chain:
                c = chain.pop()
                if b == c:
                    dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
                    graph.add_node(u)
                    remaining_nodes.add(u)
                    neighbors_a = list(graph.neighbors(a))
                    neighbors_b = list(graph.neighbors(b))
                    for v in neighbors_a:
                        graph.add_edge(u, v, weight=graph[a][v]['weight'])
                    for v in neighbors_b:
                        if graph.has_edge(u, v):
                            graph[u][v]['weight'] = max(graph[b][v]['weight'], graph[u][v]['weight'])
                        else:
                            graph.add_edge(u, v, weight=graph[b][v]['weight'])
                    graph.remove_node(a)
                    remaining_nodes.remove(a)
                    graph.remove_node(b)
                    remaining_nodes.remove(b)
                    n_nodes -= 1
                    cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
                    u += 1
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                chain.append(a)
                chain.append(b)
            else:
                connected_components.append((a, cluster_size[a]))
                graph.remove_node(a)
                cluster_size.pop(a)
                n_nodes -= 1

    a, cluster_size = connected_components.pop()
    for b, t in connected_components:
        cluster_size += t
        dendrogram.append([a, b, float("inf"), cluster_size])
        a = u
        u += 1

    return np.array(dendrogram)


def average_linkage_hierarchy(graph, f):
    remaining_nodes = set(graph.nodes())
    n_nodes = len(remaining_nodes)

    cluster_size = {u: 1 for u in range(n_nodes)}
    connected_components = []
    dendrogram = []
    u = n_nodes

    while n_nodes > 0:
        for new_node in remaining_nodes:
            chain = [new_node]
            break
        while chain:
            a = chain.pop()
            linkage_max = - float("inf")
            b = -1
            neighbors_a = list(graph.neighbors(a))
            for v in neighbors_a:
                if v != a:
                    linkage = float(graph[a][v]['weight'])/(cluster_size[a]*cluster_size[v])
                    if linkage > linkage_max:
                        b = v
                        linkage_max = linkage
                    elif linkage == linkage_max:
                        b = min(b, v)
            linkage = linkage_max
            if chain:
                c = chain.pop()
                if b == c:
                    dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
                    graph.add_node(u)
                    remaining_nodes.add(u)
                    neighbors_a = list(graph.neighbors(a))
                    neighbors_b = list(graph.neighbors(b))
                    for v in neighbors_a:
                        graph.add_edge(u, v, weight=graph[a][v]['weight'])
                    for v in neighbors_b:
                        if graph.has_edge(u, v):
                            graph[u][v]['weight'] += graph[b][v]['weight']
                        else:
                            graph.add_edge(u, v, weight=graph[b][v]['weight'])
                    graph.remove_node(a)
                    remaining_nodes.remove(a)
                    graph.remove_node(b)
                    remaining_nodes.remove(b)
                    n_nodes -= 1
                    cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
                    u += 1
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                chain.append(a)
                chain.append(b)
            else:
                connected_components.append((a, cluster_size[a]))
                graph.remove_node(a)
                cluster_size.pop(a)
                n_nodes -= 1

    a, cluster_size = connected_components.pop()
    for b, t in connected_components:
        cluster_size += t
        dendrogram.append([a, b, float("inf"), cluster_size])
        a = u
        u += 1

    return np.array(dendrogram)


def complete_linkage_hierarchy(graph, f):
    remaining_nodes = set(graph.nodes())
    n_nodes = len(remaining_nodes)

    cluster_size = {u: 1 for u in range(n_nodes)}
    connected_components = []
    dendrogram = []
    u = n_nodes

    while n_nodes > 0:
        for new_node in remaining_nodes:
            chain = [new_node]
            break
        while chain:
            a = chain.pop()
            linkage_max = - float("inf")
            b = -1
            neighbors_a = list(graph.neighbors(a))
            for v in neighbors_a:
                if v != a:
                    linkage = float(graph[a][v]['weight'])
                    if linkage > linkage_max:
                        b = v
                        linkage_max = linkage
                    elif linkage == linkage_max:
                        b = min(b, v)
            linkage = linkage_max
            if chain:
                c = chain.pop()
                if b == c:
                    dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
                    graph.add_node(u)
                    remaining_nodes.add(u)
                    neighbors_a = list(graph.neighbors(a))
                    neighbors_b = list(graph.neighbors(b))
                    for v in neighbors_a:
                        graph.add_edge(u, v, weight=graph[a][v]['weight'])
                    for v in neighbors_b:
                        if graph.has_edge(u, v):
                            graph[u][v]['weight'] = min(graph[b][v]['weight'], graph[u][v]['weight'])
                        else:
                            graph.add_edge(u, v, weight=graph[b][v]['weight'])
                    graph.remove_node(a)
                    remaining_nodes.remove(a)
                    graph.remove_node(b)
                    remaining_nodes.remove(b)
                    n_nodes -= 1
                    cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
                    u += 1
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                chain.append(a)
                chain.append(b)
            else:
                connected_components.append((a, cluster_size[a]))
                graph.remove_node(a)
                cluster_size.pop(a)
                n_nodes -= 1

    a, cluster_size = connected_components.pop()
    for b, t in connected_components:
        cluster_size += t
        dendrogram.append([a, b, float("inf"), cluster_size])
        a = u
        u += 1

    return np.array(dendrogram)


def modular_linkage_hierarchy(graph, f):
    remaining_nodes = set(graph.nodes())
    n_nodes = len(remaining_nodes)

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph.edges():
        weight = graph[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    cluster_size = {u: 1 for u in range(n_nodes)}
    connected_components = []
    dendrogram = []
    u = n_nodes

    while n_nodes > 0:
        for new_node in remaining_nodes:
            chain = [new_node]
            break
        while chain:
            a = chain.pop()
            linkage_max = - float("inf")
            b = -1
            neighbors_a = list(graph.neighbors(a))
            for v in neighbors_a:
                if v != a:
                    linkage = wtot * float(graph[a][v]['weight'])/(w[a]*w[v])
                    if linkage > linkage_max:
                        b = v
                        linkage_max = linkage
                    elif linkage == linkage_max:
                        b = min(b, v)
            linkage = linkage_max
            if chain:
                c = chain.pop()
                if b == c:
                    dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
                    graph.add_node(u)
                    remaining_nodes.add(u)
                    neighbors_a = list(graph.neighbors(a))
                    neighbors_b = list(graph.neighbors(b))
                    for v in neighbors_a:
                        graph.add_edge(u, v, weight=graph[a][v]['weight'])
                    for v in neighbors_b:
                        if graph.has_edge(u, v):
                            graph[u][v]['weight'] += graph[b][v]['weight']
                        else:
                            graph.add_edge(u, v, weight=graph[b][v]['weight'])
                    graph.remove_node(a)
                    remaining_nodes.remove(a)
                    graph.remove_node(b)
                    remaining_nodes.remove(b)
                    n_nodes -= 1
                    w[u] = w.pop(a) + w.pop(b)
                    cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
                    u += 1
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                chain.append(a)
                chain.append(b)
            else:
                connected_components.append((a, cluster_size[a]))
                graph.remove_node(a)
                w.pop(a)
                cluster_size.pop(a)
                n_nodes -= 1

    a, cluster_size = connected_components.pop()
    for b, t in connected_components:
        cluster_size += t
        dendrogram.append([a, b, float("inf"), cluster_size])
        a = u
        u += 1

    return np.array(dendrogram)

def reorder_dendrogram(D):
    n = np.shape(D)[0] + 1
    order = np.zeros((2, n - 1), float)
    order[0] = range(n - 1)
    order[1] = np.array(D)[:, 2]
    index = np.lexsort(order)
    nindex = {i: i for i in range(n)}
    nindex.update({n + index[t]: n + t for t in range(n - 1)})
    return np.array([[nindex[int(D[t][0])], nindex[int(D[t][1])], D[t][2], D[t][3]] for t in range(n - 1)])[index, :]


### Graph Models

class PPM:
    def __init__(self, partition, p_in, p_out):
        partition = np.array(partition)
        k = len(partition)
        n = sum(partition)
        index = np.cumsum(partition)
        self._clusters = [list((index[i - 1] % n) + range(partition[i])) for i in range(k)]
        self._p_in = p_in
        self._p_out = p_out

    @staticmethod
    def from_probability(partition, p_in, p_out):
        return PPM(partition, p_in, p_out)

    @staticmethod
    def from_degree(n_clusters, size_cluster, d_in, d_out):
        return PPM(n_clusters * [size_cluster], min(d_in / float(size_cluster), 1), min(d_out / float((n_clusters-1) * size_cluster), 1))

    def create_graph(self, distribution='Binomial'):
        G = nx.Graph()
        G.add_nodes_from(range(self._clusters[-1][-1]))
        if distribution == 'Binomial':
            edges = self._create_edges_binomial()
        elif distribution == 'Poisson':
            edges = self._create_edges_poisson()
        G.add_weighted_edges_from(edges)
        return G

    def _create_edges_binomial(self):
        edges = []
        for i, C_i in enumerate(self._clusters):
            for j, C_j in enumerate(self._clusters):
                if i == j:
                    for u in C_i:
                        for v in C_j:
                            if u != v and rnd.random() < self._p_in:
                                edges.append((u, v, 1.))
                else:
                    for u in C_i:
                        for v in C_j:
                            if u != v and rnd.random() < self._p_out:
                                edges.append((u, v, 1.))
        return edges

    def _create_edges_poisson(self):
        edges = []
        for i, C_i in enumerate(self._clusters):
            for j, C_j in enumerate(self._clusters):
                if i == j:
                    for u in C_i:
                        for v in C_j:
                            weight = np.random.poisson(self._p_in)
                            if u != v and weight > 0:
                                edges.append((u, v, weight))

                else:
                    for u in C_i:
                        for v in C_j:
                            weight = np.random.poisson(self._p_out)
                            if u != v and weight > 0:
                                edges.append((u, v, weight))

        return edges

    def clusters(self):
        return self._clusters

    def info(self):
        print("\nClusters: ", self._clusters)
        print("Probability in: ", self._p_in)
        print("Probability out: ", self._p_out)

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
        
### Graph sampling

def block_size_ratio_samples(range_ratio, big_block_size=100, graph_size=600, p_in=.5, p_out=.01, n_samples=10):
    samples = []
    true_clusters = []
    random_clusters = []
    n_big_bocks = int(graph_size/float(2 * big_block_size))
    for ratio in range_ratio:
        samples.append([])
        true_clusters.append([])
        random_clusters.append([])
        small_block_size = int(big_block_size / float(ratio))
        n_small_blocks = int(graph_size / float(2 * small_block_size))
        partition = n_small_blocks * [small_block_size] + n_big_bocks * [big_block_size]
        partition_ = n_small_blocks * [big_block_size / float(ratio)] + n_big_bocks * [big_block_size]
        model = PPM(partition, p_in, p_out)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph()
            samples[-1].append(graph)
            true_clusters[-1].append(model.clusters())
            rdm_cluster = [[] for i in range(n_small_blocks + n_big_bocks)]
            #for u in range(graph_size):
            #    i = np.random.choice(n_small_blocks + n_big_bocks, p=[partition_[i] / float(graph_size) for i in range(len(partition))])
            #    rdm_cluster[i].append(u)
            random_clusters[-1].append(rdm_cluster)
    return samples, true_clusters, random_clusters

def samples_evaluation(samples, clusters, score):
    results_score = []
    for i, graph_list in enumerate(samples):
        results_score.append([])
        for j, graph in enumerate(graph_list):
            results_score[-1].append(score(graph, clusters[i][j]))
    return results_score



