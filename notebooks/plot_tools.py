import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from scipy.cluster.hierarchy import dendrogram

### Plot graph

def plot_graph(G, pos, figsize=(10, 5), node_size=50, alpha =.2, nodes_numbering=False, edges_numbering=False, file_name=""):
    plt.figure(figsize=figsize)
    plt.axis('off')

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='w')
    nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, pos, alpha=alpha)

    if edges_numbering:
        w = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=w)

    if nodes_numbering:
        nx.draw_networkx_labels(G, pos)

    if file_name != "":
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.show()
plt.close()

### plot graph clustering

def plot_graph_clustering(G, clusters, pos, figsize=(15, 8), node_size=50, alpha=.2, title=True, nodes_numbering=False, edges_numbering=False, file_name=""):
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] + sorted(list(mcd.XKCD_COLORS))
    k = min(len(colors), len(clusters))
    clusters = sorted(clusters, key=len, reverse=True)

    plt.figure(figsize=figsize)
    plt.axis('off')

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='w')
    nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, pos, alpha=alpha)
    for l in range(k):
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, nodelist=clusters[l], node_color=colors[l])
        nodes.set_edgecolor('k')

    if edges_numbering:
        w = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=w)

    if nodes_numbering:
        nx.draw_networkx_labels(G, pos)

    if file_name != "":
        plt.savefig(file_name, bbox_inches ='tight')
    else:
        plt.show()
    plt.close()

### Plot dendrogram

def plot_dendrogram(D, scaling=lambda x:np.log(x), figsize=(10, 5), file_name=""):
    plt.figure(figsize=figsize)
    D_scaled = D.copy()
    D_scaled[:, 2] = scaling((D[:, 2])) - scaling((D[0, 2]))
    dendrogram(D_scaled, leaf_rotation=90.)
    plt.axis('off')

    if file_name != "":
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.show()
plt.close()

### Plot dendrogram clustering

def plot_dendrogram_clustering(D, clusters, scaling=lambda x: np.log(x), figsize=(10, 5), file_name=""):
    n_nodes = np.shape(D)[0] + 1
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] + sorted(list(mcd.XKCD_COLORS))
    clusters = sorted(clusters, key=len, reverse=True)
    default_color = "grey"
    node_colors={u: "grey" for u in range(n_nodes)}
    for i, c in enumerate(clusters):
        for u in c:
            if i < len(colors):
                node_colors[u] = colors[i]
            else:
                node_colors[u] = "grey"

    cluster_colors = {}
    for t, i12 in enumerate(D[:, :2].astype(int)):
        c1, c2 = (cluster_colors[x] if x > n_nodes - 1 else node_colors[x]
                  for x in i12)
        cluster_colors[t + n_nodes] = c1 if c1 == c2 else default_color

    plt.figure(figsize=figsize)
    D_scaled = D.copy()
    D_scaled[:, 2] = scaling((D[:, 2])) - scaling((D[0, 2]))
    dendrogram(D_scaled, leaf_rotation=90., link_color_func=lambda x: cluster_colors[x])
    plt.axis('off')

    if file_name != "":
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
