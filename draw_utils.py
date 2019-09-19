import numpy as np
import torch
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt
# Select the color map named rainbow
import matplotlib.cm as cm


plt.figure(figsize=(100,100))

color_map = ['red', 'green', 'blue', 'black', 'purple', 'pink', 'brown', 'navy', 'olive', 'yellowgreen', 'white', 'brown', 'palegreen', 'c', 'violet', 'crimson', 'pink', 'hotpink', 'midnightblue', 'teal', 'lime', 'cornsilk', 'lightyellow', 'dimgray', 'indianred', 'peru', 'wheat']


def get_knn(G, nodeidx, K):
    edges = G.edges()
    ret_edges = []
    nodeset = {nodeidx}
    for k in range(K):
        tmp = set()
        for e in edges:
            if tuple(e) in ret_edges:
                continue
            
            if e[0] in nodeset or e[1] in nodeset:
                ret_edges.append(tuple(e))
                tmp.add(e[0])
                tmp.add(e[1])
        nodeset |= set(tmp)
    return ret_edges, nodeset

def draw_with_edge_color(G):
    plt.close()
    try:
        colors = [color_map[G[u][v]['color']] for u,v in G.edges()]
        pos = nx.drawing.nx_agraph.graphviz_layout(G)
        nx.draw(G, pos, edge_color=colors, node_color='black', node_size=8, font_size=8)
    except:
        colors = [G[u][v]['color'] for u,v in G.edges()]
        pos = nx.drawing.nx_agraph.graphviz_layout(G)
        nx.draw(G, pos, edge_color=colors, node_color='black', node_size=8, font_size=8)
    plt.show()

def draw_with_node_color(G, node_color_idx):
    plt.close()
    pos = nx.drawing.nx_agraph.graphviz_layout(G)
    nx.draw(G, pos, node_color=node_color_idx, edge_color='black', node_size=8, font_size=8)
    plt.show()
    
def draw_around_node(G, nodeidx, K):
    edges, nodeset = get_knn(G, nodeidx, K)
    print('edges', len(edges))
    print('nodes', len(nodeset))
    newG = nx.Graph()
    for e in edges:
        newG.add_edge(e[0], e[1], color=G[e[0]][e[1]]['color'])

    draw_with_edge_color(newG)

def draw_model_edge_pred(G, model):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)
    color_idx = q.argmax(dim=-1)
    G = G.copy()

    # cmap = cm.get_cmap(name='rainbow')

    for idx, e in enumerate(edges):
        G[e[0]][e[1]]['color'] = color_map[color_idx[idx]]
        #G[e[0]][e[1]]['color'] = cmap(color_idx[idx])
    draw_with_edge_color(G)

def draw_uncertainty_graph(G, mod):
    if type(G) is list:
        edges = G
    else:
        edges = [(u,v) for u,v in G.edges()]

    model.eval()
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)
    q = F.softmax(q, dim=-1)
    color_idx =  (-torch.log(q) * q).sum(dim=-1).cpu().data.numpy()
    
    plt.close()
    G = G.copy()
    for idx, e in enumerate(edges):
        G[e[0]][e[1]]['color'] = color_idx[idx]
    
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    pos = nx.drawing.nx_agraph.graphviz_layout(G)
    nx.draw(G, pos, edge_color=colors, edge_vmin=np.min(colors), edge_vmax=np.max(colors), edge_cmap=plt.cm.Blues, with_labels=False, node_color='black', node_size=8, font_size=8)
    #nx.draw(G, pos, edge_color=colors, edge_cmap=plt.cm.Blues, with_labels=False, node_color='black', node_size=8, font_size=8)
    plt.show()

def draw_model_prior(G, model):
    if type(G) is list:
        edges = G
        G = nx.Graph()
        for e in edges:
            G.add_edge(e[0], e[1])
    else:
        edges = [(u,v) for u,v in G.edges()]

    model.eval()
    batch = torch.LongTensor(edges)
    _, _, prior = model(batch[:, 0], batch[:, 1], 1.)
    node_color_idx = prior.argmax(dim=-1).cpu().data.numpy()
    draw_with_node_color(G, [color_map[idx] for idx in node_color_idx])

def draw_class_prob(G, model, tpe=0):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    n_nodes = G.number_of_nodes()
    for c in range(num_classes):
        plt.close()
        colors = [0. for i in range(n_nodes)]
        for idx, e in enumerate(edges):
            if tpe == 0:
                colors[e[0]] += q[idx][c]
                colors[e[1]] += q[idx][c]
            else:
                if q_argmax[idx] == c:
                    colors[e[0]] += 1
                    colors[e[1]] += 1
                    
                
        for i in range(n_nodes):
            colors[i] /= G.degree[i]

        pos = nx.drawing.nx_agraph.graphviz_layout(G)
        nx.draw(G, pos, node_color=colors,
                vmin=np.min(colors), vmax=np.max(colors),
                node_cmap=plt.cm.Blues, with_labels=False,
                edge_color='black', node_size=8, font_size=8)

        plt.show()

def draw_class_gt(G, num_classes, tpe=0):
    n_nodes = G.number_of_nodes()
    for c in range(num_classes):
        plt.close()
        colors = [0. for i in range(n_nodes)]
        for idx, multi_e in enumerate(G.edges()):
            u, v = multi_e[0], multi_e[1]
            for _, attr in G[u][v].items():
                if tpe == 0:
                    if attr['color'] == c:
                        colors[u] += 1.
                        colors[v] += 1.
                else:
                    if attr['color'] == c:
                        colors[u] = 1.
                        colors[v] = 1.

        if tpe == 0:
            for i in range(n_nodes):
                colors[i] /= G.degree[i]

        pos = nx.drawing.nx_agraph.graphviz_layout(G)
        nx.draw(G, pos, node_color=colors,
                vmin=np.min(colors), vmax=np.max(colors),
                with_labels=False, node_size=8)

        plt.show()

def draw_node_embeddings(G, model, n_components=5):
    n_nodes = G.number_of_nodes()
    model.eval()
    inp = torch.LongTensor(np.arange(n_nodes))
    emb = model.node_embeddings(inp).cpu().data.numpy()
    print(emb.shape)
    tsne = manifold.TSNE(n_components=n_components, random_state=501)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    y = np.zeros(n_nodes)
    print(y.shape)

    plt.close()
    for i in range(X_norm.shape[0]):
        plt.plot(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()
