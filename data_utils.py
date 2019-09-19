import re
import pickle as pkl
import collections
import os
import networkx as nx 
import numpy as np
import scipy.sparse as sp
# import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.cluster import KMeans
#import community
import pandas as pd
from tqdm import tqdm

def load_data(dataset_str):
    if dataset_str == 'blog':
        G, adj, features = graph_reader('./data/BlogCatalog-dataset/data/edges.csv')
    elif dataset_str == 'flickr':
        G, adj, features = graph_reader('./data/Flickr-dataset/data/edges.csv')
    elif dataset_str in ['cora', 'citeseer']:
        G, adj, features = load_cc(dataset_str)
    elif dataset_str == 'wiki':
        import scipy.io as sio
        A = sio.loadmat('./data/POS.mat')['network']
        G = nx.from_scipy_sparse_matrix(A)
        adj = nx.adjacency_matrix(G)
        features = None
    elif 'dblp' in dataset_str:
        G, adj, edge_labels = read_dblp_small('./data/dblp-small/net_co_author.txt')
        features = None
    else:
        assert False
    n_nodes = adj.shape[0]

    return G, adj, features

def load_cc(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/link-pred/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/link-pred/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print('adj.shape', adj.shape)
    print('type(adj)', type(adj))
    print('features.shape', features.shape)
    print('type(features)', type(features))

    return nx.from_dict_of_lists(graph), adj, features

def graph_reader(fpath):
    """
    Function to read a csv edge list and transform it to a networkx graph object.
    """    
    edges = pd.read_csv(fpath)
    graph = nx.convert_node_labels_to_integers(nx.from_edgelist(edges.values.tolist()))
    assert list(graph.nodes())[0] == 0
    print('number of nodes', graph.number_of_nodes())
    print('number of edges', graph.number_of_edges())
    adj = nx.adjacency_matrix(graph)
    return graph, adj, None

    # degs = np.expand_dims(np.sum(np.array(adj.todense()), 1), 1)
    # # clusterings = np.array(list(nx.clustering(graph).values()))
    # # features = np.hstack([degs, clusterings])
    # features = degs
    # features = torch.FloatTensor(np.array(features))

    # print('adj.shape', adj.shape)
    # print('type(adj)', type(adj))
    # print('features.shape', features.shape)
    # print('type(features)', type(features))
    # return graph, adj, features

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    print('num_val', num_val)
    print('num_test', num_test)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    for e in tqdm(test_edges):
        idx_i = e[0]
        while True:
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])
            break

    val_edges_false = []
    for e in tqdm(val_edges):
        idx_i = e[0]
        while True: #len(val_edges_false) < len(val_edges):
            # idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], test_edges):
                continue
            if ismember([idx_j, idx_i], test_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])
            break

    # print( ~ismember(test_edges_false, edges_all))
    # print( ~ismember(val_edges_false, edges_all))
    # print( ~ismember(val_edges, train_edges))
    # print( ~ismember(test_edges, train_edges))
    # print( ~ismember(val_edges, test_edges))
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score_v2(adj_rec, adj_orig, edges_pos, edges_neg):

    # Predict on test set of edges
    # adj_rec = np.dot(emb, emb.T)

    size = adj_orig.shape[0]
    preds = [[] for _ in range(size)]
    labels = [[] for _ in range(size)]

    idx = 0
    for e in edges_pos:
        preds[e[0]].append(adj_rec[idx, e[1]])
        idx+=1
        labels[e[0]].append(1)

    for e in edges_neg:
        preds[e[0]].append(adj_rec[idx, e[1]])
        idx+=1
        labels[e[0]].append(0)

    # preds_all = np.hstack([preds, preds_neg])
    # labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    tot = 0
    final_roc_score = 0
    final_ap_score = 0
    for i in range(size):
        if len(preds[i]) > 0:
            roc_score = roc_auc_score(labels[i], preds[i])
            ap_score = average_precision_score(labels[i], preds[i])

            ln = len(preds[i])
            final_roc_score += roc_score * ln
            final_ap_score += ap_score * ln
            tot += ln

    return final_roc_score / tot, final_ap_score / tot

def get_roc_score(adj_rec, adj_orig, edges_pos, edges_neg):

    # Predict on test set of edges
    # adj_rec = np.dot(emb, emb.T)

    size = adj_orig.shape[0]
    preds = [[] for _ in range(size)]
    labels = [[] for _ in range(size)]

    for e in edges_pos:
        preds[e[0]].append(adj_rec[e[0], e[1]])
        labels[e[0]].append(1)

    for e in edges_neg:
        preds[e[0]].append(adj_rec[e[0], e[1]])
        labels[e[0]].append(0)

    # preds_all = np.hstack([preds, preds_neg])
    # labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    tot = 0
    final_roc_score = 0
    final_ap_score = 0
    for i in range(size):
        if len(preds[i]) > 0:
            roc_score = roc_auc_score(labels[i], preds[i])
            ap_score = average_precision_score(labels[i], preds[i])

            ln = len(preds[i])
            final_roc_score += roc_score * ln
            final_ap_score += ap_score * ln
            tot += ln

    return final_roc_score / tot, final_ap_score / tot

def read_dblp_small2(fpath):
    G = nx.Graph()
    edge_labels = collections.defaultdict(list)
    mapping = {}
    with open(fpath, 'r') as f:
        for line in f:
            e0, e1, lab, _= line.strip().split('\t')
            if not (int(lab) >= 1 and int(lab) <= 2):
                continue

            try:
                e0 = mapping[e0]
            except:
                mapping[e0] = len(mapping)
                e0 = mapping[e0]

            try:
                e1 = mapping[e1]
            except:
                mapping[e1] = len(mapping)
                e1 = mapping[e1]

            G.add_edge(e0, e1)

            # edge_labels[(e0,e1)].append(int(lab))
            # edge_labels[(e1,e0)].append(int(lab))

    return G, nx.adjacency_matrix(G), edge_labels

def load_cora_citeseer(ds):
    dirpath = '../data/{}'.format(ds)
    edge_path = dirpath + '/{}.cites'.format(ds)
    content_path = dirpath + '/{}.content'.format(ds)
    
    with open(content_path, 'r') as f:
        content = f.readlines()
        
    mapping = {}
    label_mapping = {}
    membership = {}
    for line in content:
        tmp = line.strip().split('\t')
        mapping[tmp[0]] = len(mapping)
        try:
            lab = label_mapping[tmp[-1]]
        except:
            label_mapping[tmp[-1]] = len(label_mapping)
            lab = label_mapping[tmp[-1]]

        membership[mapping[tmp[0]]] = lab
    assert len(membership) == len(mapping)

    G = nx.Graph()
    with open(edge_path, 'r') as f:
        for line in f:
            e0, e1 = line.strip().split('\t')
            try:
                e0 = mapping[e0]
                e1 = mapping[e1]
            except:
                continue

            G.add_edge(e0, e1)

    assert len(mapping) ==G.number_of_nodes()
    
    assert max(list(G.nodes())) == G.number_of_nodes()-1
    membership = [membership[i] for i in range(G.number_of_nodes())]
    return G, nx.adjacency_matrix(G), membership

def load_webkb(ds):
    dirpath = '../data/WebKB'
    edge_path = dirpath + '/{}.cites'.format(ds)
    content_path = dirpath + '/{}.content'.format(ds)
    G = nx.Graph()

    mapping = {}
    with open(edge_path, 'r') as f:
        for line in f:
            e0, e1 = line.strip().split(' ')
            try:
                e0 = mapping[e0]
            except:
                mapping[e0] = len(mapping)
                e0 = mapping[e0]

            try:
                e1 = mapping[e1]
            except:
                mapping[e1] = len(mapping)
                e1 = mapping[e1]

            G.add_edge(e0, e1)

    assert len(mapping) ==G.number_of_nodes()
    label_mapping = {}
    membership = [-1 for _ in range(G.number_of_nodes())]
    with open(content_path, 'r') as f:
        for line in f:
            tmp = line.strip().split('\t')
            try:
                lab = label_mapping[tmp[-1]]
            except:
                label_mapping[tmp[-1]] = len(label_mapping)
                lab = label_mapping[tmp[-1]]

            membership[mapping[tmp[0]]] = lab
    
    membership = [membership[i] for i in range(G.number_of_nodes())]
    return G, nx.adjacency_matrix(G), membership

def read_all_facebook():
    G = nx.Graph()
    communities = []
    for fb_num in [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]:
        print('joining facebook', fb_num)
        cur_node_idx = G.number_of_nodes()
        if fb_num != 0:
            assert max(list(G.nodes())) == cur_node_idx-1
            assert min(list(G.nodes())) == 0

        mapping = {}
        edge_fpath = '../data/facebook/{}.edges'.format(fb_num)
        label_fpath =  '../data/facebook/{}.circles'.format(fb_num)
        with open(edge_fpath, 'r') as f:
            for line in f:
                e0, e1 = line.strip().split('\t')
                # print(e0, e1)
                try:
                    tmp = mapping[e0]
                except:
                    mapping[e0] = len(mapping)

                try:
                    tmp = mapping[e1]
                except:
                    mapping[e1] = len(mapping)

                G.add_edge(cur_node_idx + mapping[e0], cur_node_idx + mapping[e1])

        with open(label_fpath, 'r') as f:
            for line in f:
                nodes = re.split(' |\t', line.strip())[1:]
                for n in nodes:
                    try:
                        tmp = mapping[n]
                    except:
                        mapping[n] = len(mapping)

                communities.append(list([cur_node_idx+mapping[x] for x in nodes]))

    return G, nx.adjacency_matrix(G), communities

def read_facebook(ds, relabel=True):
    fb_num = int(ds[8:])
    print('using facebook', fb_num)
    edge_fpath = '../data/facebook/{}.edges'.format(fb_num)
    label_fpath =  '../data/facebook/{}.circles'.format(fb_num)
    if relabel:
        mapping = {}
        G = nx.Graph()
        with open(edge_fpath, 'r') as f:
            for line in f:
                e0, e1 = line.strip().split('\t')
                # print(e0, e1)
                try:
                    tmp = mapping[e0]
                except:
                    mapping[e0] = len(mapping)

                try:
                    tmp = mapping[e1]
                except:
                    mapping[e1] = len(mapping)

                G.add_edge(e0, e1)
                G.add_edge(e1, e0)

        communities = []
        with open(label_fpath, 'r') as f:
            for line in f:
                nodes = re.split(' |\t', line.strip())[1:]
                for n in nodes:
                    try:
                        tmp = mapping[n]
                    except:
                        mapping[n] = len(mapping)

                communities.append(list([mapping[x] for x in nodes]))


        G = nx.relabel_nodes(G, mapping)
        return G, nx.adjacency_matrix(G), communities
    else:
        assert False
        G = nx.Graph()
        with open(fpath, 'r') as f:
            for line in f:
                e0, e1 = line.strip().split('\t')
                G.add_edge(e0, e1)

        communities = []
        with open(label_fpath, 'r') as f:
            for line in f:
                nodes = re.split(' |\t', line.strip())[1:]
                communities.append([x for x in nodes])

        return G, nx.adjacency_matrix(G), communities

def load_bigds(ds, relabel=False):
    if ds.startswith('amazon'):
        ds, num = ds[:6], int(ds[6:])
    elif ds.startswith('youtube'):
        ds, num = ds[:7], int(ds[7:])
    elif ds.startswith('dblp'):
        ds, num = ds[:4], int(ds[4:])
    else:
        assert False
    edges_path = '../data/{}/com-{}.ungraph.txt'.format(ds, ds)
    community_path = '../data/{}/com-{}.top5000.cmty.txt'.format(ds, ds)

    raw_communities = []
    with open(community_path, 'r') as f:
        for line in tqdm(f):
            tmp = line.strip().split('\t')
            raw_communities.append(tmp)
    raw_communities.sort(key=lambda x: len(x), reverse=True)
    # print([len(x) for x in raw_communities])
    # input()

    communities = []
    mapping = {}
    for tmp in raw_communities[:num]:
        res = []
        for x in tmp:
            try:
                xx = mapping[x]
            except:
                xx = mapping[x] = len(mapping)

            res.append(xx)

        communities.append(res)

    print('number of communities', len(communities))
    G = nx.Graph()
    with open(edges_path, 'r') as f:
        for line in tqdm(f):
            if line.startswith('#'):
                continue
            e0, e1 = line.strip().split('\t')
            try:
                G.add_edge(mapping[e0], mapping[e1])
            except Exception as e:
                continue

    print('number of nodes', G.number_of_nodes())
    return G, nx.adjacency_matrix(G), communities
    
def read_data_author(dirpath, relabel=True):
    G = nx.Graph()

    mapping = {}
    with open('{}/net_co_author.txt'.format(dirpath), 'r') as f:
        for line in f:
            e0, e1 = line.strip().split('\t')

            if relabel:
                try:
                    e0 = mapping[e0]
                except:
                    mapping[e0] = len(mapping)
                    e0 = mapping[e0]

                try:
                    e1 = mapping[e1]
                except:
                    mapping[e1] = len(mapping)
                    e1 = mapping[e1]

            G.add_edge(e0, e1)


    gt_communities = [[] for _ in range(5)]
    with open('{}/author_conf.txt'.format(dirpath), 'r') as f:
        for line in f:
            tmp = line.strip().split(' ')
            p = tmp[0]
            for lb in tmp[1:]:
                if relabel:
                    try:
                        gt_communities[int(lb)].append(mapping[p])
                    except:
                        pass

                else:
                    gt_communities[int(lb)].append(p)

    gt_communities = [list(set(x)) for x in gt_communities]
    print('cool')
    return G, nx.adjacency_matrix(G), gt_communities

def read_data_paper(dirpath, relabel=True):
    print('reading data paper ...')
    G = nx.Graph()

    ub = [500, 100, 200, 300]
    ub = [4*x for x in ub]
    mapping = {}
    gt_communities = [[] for _ in range(5)]
    with open('{}/paper_label.txt'.format(dirpath), 'r') as f:
        lines = f.readlines()
        # np.random.shuffle(lines)
        for line in lines:
            p, lab = line.strip().split('\t')
            if int(lab) == 1:
                continue
            elif int(lab) > 1:
                lab = int(lab) - 1

            if len(gt_communities[int(lab)]) > ub[int(lab)]:
                continue

            mapping[p] = len(mapping)
            gt_communities[int(lab)].append(mapping[p])
            G.add_node(mapping[p], community=int(lab))# attr_dict={'community':int(lab)})

    print([len(x) for x in gt_communities])
    with open('{}/paper_reference.txt'.format(dirpath), 'r') as f:
        for line in f:
            e0, e1, _ = line.strip().split('\t')
            try:
                e0 = mapping[e0]
                e1 = mapping[e1]
            except Exception as e:
                continue

            G.add_edge(e0, e1)



    assignment = {}
    with open('{}/paper_title.txt'.format(dirpath), 'r') as f:
        for line in f:
            e0, ln = line.strip().split('\t')
            try:
                assignment[mapping[e0]] = ln
            except:
                pass

    # ub = [1000, 500, 300, 300]
    # ub = [5000, 2500, 1500, 1500]
    # ub = [10000, 5000, 3000, 3000]
    # cnt = [0, 0, 0, 0]
    # new_nodes = []
    # for n in G.nodes():
        # com = G.node[n]['community']
        # if cnt[com] > ub[com]:
            # continue
        # # if G.degree(n) > 10:
        # new_nodes.append(n)
        # cnt[com] += 1

    # G = G.subgraph(new_nodes)

    nx.set_node_attributes(G, assignment, 'label')
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    nodes = [n for n in Gc]
    G = G.subgraph(nodes)

    mapping = {}
    mapping = {n:idx for idx, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    new_gt_communities = []
    for y in gt_communities:
        tmp = []
        for x in y:
            try:
                tmp.append(mapping[x])
            except:
                pass
        new_gt_communities.append(tmp)

    print([len(y) for y in new_gt_communities])
    # input()
    return G, nx.adjacency_matrix(G), new_gt_communities

def read_flickr(num_communities, relabel=True):
    edge_fpath = '../data/Flickr-dataset/data/edges.csv'
    gt_fpath = '../data/Flickr-dataset/data/group-edges.csv'
    G = nx.Graph()
    mapping = {}

    gt_communities = [[] for _ in range(num_communities)]
    with open(gt_fpath, 'r') as f:
        for line in tqdm(f):
            p, lab = line.strip().split(',')
            if int(lab) > num_communities:
                break
            try:
                tmp = mapping[p] 
            except:
                tmp = mapping[p] = len(mapping)
            gt_communities[int(lab)-1].append(tmp)

    G.add_nodes_from([i for i in range(len(mapping))])
    with open(edge_fpath, 'r') as f:
        for line in tqdm(f):
            e0, e1 = line.strip().split(',')

            try:
                e0 = mapping[e0]
                e1 = mapping[e1]
            except:
                continue

            assert e0 < len(mapping) and e1 < len(mapping)
            G.add_edge(e0, e1)

    # print(G.nodes())
    # input()
    return G, nx.adjacency_matrix(G), gt_communities

def load_dataset(ds, relabel=True):
    if 'flickr' in ds:
        G, adj, gt_communities = read_flickr(int(ds[6:]))

    elif ds == 'facebook':
        G, adj, gt_communities = read_all_facebook()

    elif 'facebook' in ds:
        G, adj, gt_communities = read_facebook(ds)

    elif ds == 'dblp-small':
        G, adj, gt_communities = read_data_author('../data/data_author', relabel=relabel)

    elif 'youtube' in ds or 'amazon' in ds or 'dblp' in ds:
        G, adj, gt_communities = load_bigds(ds, relabel=True)
    elif ds in ['cora', 'citeseer']:
        G, adj, gt_communities = load_cora_citeseer(ds)
    elif ds in ['cornell', 'texas', 'washington', 'wisconsin']:
        G, adj, gt_communities = load_webkb(ds)

    else:
        assert False

    print('n_nodes', G.number_of_nodes())
    print('n_edges', G.number_of_edges())
    print('number of communities', len(gt_communities))

    return G, adj, gt_communities

if __name__ == '__main__':
    G, adj, membership = load_amazon()
    assert len(membership) == adj.shape[0]
    assert -1 not in membership
    print(adj.shape)

    G, adj, membership = load_cora_citeseer('cora')
    assert len(membership) == adj.shape[0]
    assert -1 not in membership
    print(adj.shape)

    G, adj, membership = load_cora_citeseer('citeseer')
    assert len(membership) == adj.shape[0]
    assert -1 not in membership
    print(adj.shape)

    G, adj, membership = load_webkb('cornell')
    assert len(membership) == adj.shape[0]
    assert -1 not in membership
    print(adj.shape)

    G, adj, membership = load_webkb('texas')
    assert len(membership) == adj.shape[0]
    assert -1 not in membership
    print(adj.shape)

    G, adj, membership = load_webkb('washington')
    assert len(membership) == adj.shape[0]
    assert -1 not in membership
    print(adj.shape)

    G, adj, membership = load_webkb('wisconsin')
    assert len(membership) == adj.shape[0]
    assert -1 not in membership
    print(adj.shape)
