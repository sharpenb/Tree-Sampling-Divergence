import numpy as np

### Euclidean Divergence

def euclidean(graph, dendrogram):
    graph_copy = graph.copy()
    n_nodes = np.shape(dendrogram)[0] + 1

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph_copy.edges():
        weight = graph_copy[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    
    u = n_nodes
    similarity = 0
    sum_qsig = 0
    pi = {t: w[t]/float(wtot) for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])
        d = dendrogram[t][2]

        w[u] = w.pop(a) + w.pop(b)
        pi[u] = w[u] / float(wtot)

        pi_a = pi[a]
        pi_b = pi[b]
        if graph_copy.has_edge(a, b):
            p_ab = 2 * graph_copy[a][b]['weight'] / float(wtot)
            similarity += (p_ab - (pi_a * pi_b))**2

        # Update graph
        graph_copy.add_node(u)
        neighbors_a = list(graph_copy.neighbors(a))
        neighbors_b = list(graph_copy.neighbors(b))
        for v in neighbors_a:
            graph_copy.add_edge(u, v, weight=graph_copy[a][v]['weight'])
        for v in neighbors_b:
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]['weight'] += graph_copy[b][v]['weight']
            else:
                graph_copy.add_edge(u, v, weight=graph_copy[b][v]['weight'])
        graph_copy.remove_node(a)
        graph_copy.remove_node(b)

        u += 1

    return similarity

### Dasgupta's cost

def dasgupta(graph, dendrogram):
    graph_copy = graph.copy()
    n_nodes = np.shape(dendrogram)[0] + 1

    pi = {u: 1/n_nodes for u in range(n_nodes)}
    wtot = 0
    for u,v in graph_copy.edges():
        wtot += 2 * graph_copy[u][v]['weight']
    
    u = n_nodes
    similarity = 0
    sum_qsig = 0

    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])
        d = dendrogram[t][2]

        pi_a = pi.pop(a)
        pi_b = pi.pop(b)
        pi[u] = pi_a + pi_b

        if graph_copy.has_edge(a, b):
            p_ab = 2 * graph_copy[a][b]['weight'] / float(wtot)
            similarity += p_ab * (pi_a + pi_b)

        # Update graph
        graph_copy.add_node(u)
        neighbors_a = list(graph_copy.neighbors(a))
        neighbors_b = list(graph_copy.neighbors(b))
        for v in neighbors_a:
            graph_copy.add_edge(u, v, weight=graph_copy[a][v]['weight'])
        for v in neighbors_b:
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]['weight'] += graph_copy[b][v]['weight']
            else:
                graph_copy.add_edge(u, v, weight=graph_copy[b][v]['weight'])
        graph_copy.remove_node(a)
        graph_copy.remove_node(b)

        u += 1

    return similarity

### Tree Sampling Divergence

def tsd(graph, dendrogram, normalized=True):
    graph_copy = graph.copy()
    n_nodes = np.shape(dendrogram)[0] + 1

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph_copy.edges():
        weight = graph_copy[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    
    u = n_nodes
    similarity = 0
    sum_qsig = 0
    pi = {t: w[t]/float(wtot) for t in range(n_nodes)}
    p = {t: w[t]/float(wtot) for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])
        d = dendrogram[t][2]

        w[u] = w.pop(a) + w.pop(b)
        pi[u] = w[u] / float(wtot)

        pi_a = pi[a]
        pi_b = pi[b]
        if graph_copy.has_edge(a, b):
            p_ab = 2 * graph_copy[a][b]['weight'] / float(wtot)
            similarity += p_ab * np.log(p_ab / (pi_a * pi_b))
            sum_qsig += p_ab

        # Update graph
        graph_copy.add_node(u)
        neighbors_a = list(graph_copy.neighbors(a))
        neighbors_b = list(graph_copy.neighbors(b))
        for v in neighbors_a:
            graph_copy.add_edge(u, v, weight=graph_copy[a][v]['weight'])
        for v in neighbors_b:
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]['weight'] += graph_copy[b][v]['weight']
            else:
                graph_copy.add_edge(u, v, weight=graph_copy[b][v]['weight'])
        graph_copy.remove_node(a)
        graph_copy.remove_node(b)

        u += 1
    
    similarity -= np.log(sum_qsig)
    
    if normalized:
        norm = 0
        for u, v in graph.edges():
            p_uv = 2 * graph[u][v]['weight'] / float(wtot)
            norm += p_uv * np.log(p_uv / (p[u] * p[v]))
        similarity /= norm
    return similarity
