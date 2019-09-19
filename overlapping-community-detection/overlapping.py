from __future__ import division
from __future__ import print_function

import argparse
import time
from tqdm import tqdm
import math
import numpy as np
from subprocess import check_output
import sys
sys.path.append('../')
from score_utils import calc_f1, calc_overlap_nmi, calc_jaccard, calc_omega
from score_utils import normalized_overlap
from data_utils import load_dataset

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import optim

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import collections
import re

import community
import torch
import numpy as np
from sklearn.cluster import KMeans import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='s', help="models used")
parser.add_argument('--lamda', type=float, default=0, help="")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5001, help='Number of epochs to train.')
parser.add_argument('--embedding-dim', type=int, default=128, help='')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='facebook0', help='type of dataset.')
parser.add_argument('--log-file', type=str, default='tmplog', help='log path')
# parser.add_argument('--task', type=str, default='community', help='type of dataset.')


def logging(args, epochs, cur_loss, f1, nmi, jaccard, modularity):
    if args.log_file is None:
        return
    with open(args.log_file, 'a+') as f:
        f.write('{},{},{},{},{},{},{},{},{},{}\n'.format('gcn_vae',
            args.dataset_str,
            args.lr,
            cur_loss, args.lamda, epochs, f1, nmi, jaccard, modularity))

def write_to_file(fpath, clist):
    with open(fpath, 'w') as f:
        for c in clist:
            f.write(' '.join(map(str, c)) + '\n')

def preprocess(fpath): 
    clist = []
    with open(fpath, 'rb') as f:
        for line in f:
            tmp = re.split(b' |\t', line.strip())[1:]
            clist.append([x.decode('utf-8') for x in tmp])
    
    write_to_file(fpath, clist)
            

def get_assignment(G, model, num_classes=5, tpe=0):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    assignment = {}

    n_nodes = G.number_of_nodes()

    res = np.zeros((n_nodes, num_classes))
    for idx, e in enumerate(edges):
        if tpe == 0:
            res[e[0], :] += q[idx, :].cpu().data.numpy()
            res[e[1], :] += q[idx, :].cpu().data.numpy()
        else:
            res[e[0], q_argmax[idx]] += 1
            res[e[1], q_argmax[idx]] += 1

    res = res.argmax(axis=-1)
    assignment = {i : res[i] for i in range(res.shape[0])}
    return assignment

def classical_modularity_calculator(graph, embedding, model='gcn_vae', cluster_number=5):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """    
    if model == 'gcn_vae':
        assignments = embedding
    else:
        kmeans = KMeans(n_clusters=cluster_number, random_state=0, n_init = 1).fit(embedding)
        assignments = {i: int(kmeans.labels_[i]) for i in range(0, embedding.shape[0])}

    modularity = community.modularity(assignments, graph)
    return modularity

def loss_function(recon_c, q_y, prior, c, norm=None, pos_weight=None):
    
    BCE = F.cross_entropy(recon_c, c, reduction='sum') / c.shape[0]
    # BCE = F.binary_cross_entropy_with_logits(recon_c, c, pos_weight=pos_weight)
    # return BCE

    log_qy = torch.log(q_y  + 1e-20)
    KLD = torch.sum(q_y*(log_qy - torch.log(prior)),dim=-1).mean()

    ent = (- torch.log(q_y) * q_y).sum(dim=-1).mean()
    return BCE + KLD

class GCNModelGumbel(nn.Module):
    def __init__(self, size, embedding_dim, categorical_dim, dropout, device):
        super(GCNModelGumbel, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.size = size

        self.community_embeddings = nn.Linear(embedding_dim, categorical_dim, bias=False).to(device)
        self.node_embeddings = nn.Embedding(size, embedding_dim)
        #self.contextnode_embeddings = nn.Embedding(size, embedding_dim)

        self.decoder = nn.Sequential(
          nn.Linear( embedding_dim, size),
        ).to(device)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, w, c, temp):

        w = self.node_embeddings(w).to(self.device)
        c = self.node_embeddings(c).to(self.device)

        q = self.community_embeddings(w*c)
        # q.shape: [batch_size, categorical_dim]
        # z = self._sample_discrete(q, temp)
        if self.training:
            z = F.gumbel_softmax(logits=q, tau=temp, hard=True)
        else:
            tmp = q.argmax(dim=-1).reshape(q.shape[0], 1)
            z = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.)

        prior = self.community_embeddings(w)
        prior = F.softmax(prior, dim=-1)
        # prior.shape [batch_num_nodes, 

        # z.shape [batch_size, categorical_dim]
        new_z = torch.mm(z, self.community_embeddings.weight)
        recon = self.decoder(new_z)
            
        return recon, F.softmax(q, dim=-1), prior



def get_overlapping_community(G, model, tpe=1):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    assignment = {}

    n_nodes = G.number_of_nodes()

    
    res = np.zeros((n_nodes, num_classes))
    for idx, e in enumerate(edges):
        if tpe == 0:
            res[e[0], :] += q[idx, :].cpu().data.numpy()
            res[e[1], :] += q[idx, :].cpu().data.numpy()
        else:
            res[e[0], q_argmax[idx]] += 1
            res[e[1], q_argmax[idx]] += 1

    communities = [[] for _ in range(num_classes)]
    for i in range(n_nodes):
        for j in range(num_classes):
            if res[i, j] > 0:
                communities[j].append(i)
    
    return communities

if __name__ == '__main__':
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    lr = args.lr
    epochs = args.epochs
    temp = 1.
    temp_min = 0.1
    ANNEAL_RATE = 0.00003

    G, adj, gt_communities = load_dataset(args.dataset_str)
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    categorical_dim = len(gt_communities)
    n_nodes = G.number_of_nodes()
    print(n_nodes, categorical_dim)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCNModelGumbel(adj.shape[0], embedding_dim, categorical_dim, args.dropout, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hidden_emb = None
    history_valap = []
    history_mod = []

    #train_edges = np.concatenate([train_edges, val_edges, test_edges])
    train_edges = [(u,v) for u,v in G.edges()]
    n_nodes = G.number_of_nodes()
    print('len(train_edges)', len(train_edges))
    print('calculating normalized_overlap')
    overlap = torch.Tensor([normalized_overlap(G,u,v) for u,v in train_edges]).to(device)
    # overlap = torch.Tensor([(G.degree(u)-G.degree(v))**2 for u,v in train_edges]).to(device)
    # overlap = torch.Tensor([1. for u,v in train_edges]).to(device)
    # overlap = torch.Tensor([float(max(G.degree(u), G.degree(v))**2) for u,v in train_edges]).to(device)
    cur_lr = args.lr
    for epoch in range(epochs):
        #np.random.shuffle(train_edges)

        t = time.time()
        batch = torch.LongTensor(train_edges)
        assert batch.shape == (len(train_edges), 2)

        model.train()
        optimizer.zero_grad()

        w = torch.cat((batch[:, 0], batch[:, 1]))
        c = torch.cat((batch[:, 1], batch[:, 0]))
        recon, q, prior = model(w, c, temp)
        loss = loss_function(recon, q, prior, c.to(device), None, None)

        if args.lamda > 0:
            tmp_w, tmp_c = batch[:, 0], batch[:, 1]
            res = torch.zeros([n_nodes, categorical_dim], dtype=torch.float32, requires_grad=True).to(device)
            for idx, e in enumerate(train_edges):
                res[e[0], :] += q[idx, :]
                res[e[1], :] += q[idx, :]
                #res[e[0], :] += q[idx, :]/G.degree(e[0])
                #res[e[1], :] += q[idx, :]/G.degree(e[1])

            # res /= res.sum(dim=-1).unsqueeze(-1).detach()
            # tmp = F.mse_loss(res[tmp_w], res[tmp_c])

            tmp = ((res[tmp_w] - res[tmp_c])**2).mean(dim=-1)
            assert overlap.shape == tmp.shape
            smoothing_loss = (overlap*tmp).mean()
            loss += args.lamda * smoothing_loss

        loss.backward()
        cur_loss = loss.item()

        optimizer.step()

        if np.isnan(loss.item()):
           break
        
        if epoch % 10 == 0:
            temp = np.maximum(temp*np.exp(-ANNEAL_RATE*epoch),temp_min)

        if epoch % 100 == 0:
            
            model.eval()
            assert not model.training 
            
            assignment = get_assignment(G, model, categorical_dim)
            modularity = classical_modularity_calculator(G, assignment)
            
            communities = get_overlapping_community(G, model)

            nmi = calc_overlap_nmi(n_nodes, communities, gt_communities)
            f1 = calc_f1(n_nodes, communities, gt_communities)
            jaccard = calc_jaccard(n_nodes, communities, gt_communities)
            omega = calc_omega(n_nodes, communities, gt_communities)

            if args.lamda > 0:
                print("Epoch:", '%04d' % (epoch + 1),
                              "lr:", '{:.5f}'.format(cur_lr),
                              "temp:", '{:.5f}'.format(temp),
                              "train_loss=", "{:.5f}".format(cur_loss),
                              "smoothing_loss=", "{:.5f}".format(args.lamda * smoothing_loss.item()),
                              "modularity=", "{:.5f}".format(modularity),
                              "nmi", nmi, "f1", f1, 'jaccard', jaccard, "omega", omega)
            else:
                print("Epoch:", '%04d' % (epoch + 1),
                              "lr:", '{:.5f}'.format(cur_lr),
                              "temp:", '{:.5f}'.format(temp),
                              "train_loss=", "{:.5f}".format(cur_loss),
                              "modularity=", "{:.5f}".format(modularity),
                              "nmi", nmi, "f1", f1, 'jaccard', jaccard, "omega", omega)
            logging(args, epoch, cur_loss, f1, nmi, jaccard, modularity)

            cur_lr = cur_lr * .95
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
            
    print("Optimization Finished!")
