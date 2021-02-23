import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import defaultdict
import pandas as pd
import scipy
from subprocess import check_output
from itertools import combinations
from collections import Counter
import networkx as nx

def normalized_overlap(g, node_1, node_2):
    """
    Function to calculate the normalized neighborhood overlap.
    :param g: NX graph.
    :param node_1: Node 1. of a pair.
    :param node_2: Node 2. of a pair.
    """    
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    unio = len(set(nx.neighbors(g, node_1)).union(set(nx.neighbors(g, node_2))))
    return float(inter)/float(unio)


class Omega:
    def __init__(self, comms1, comms2):
        comms1 = {idx:x for idx,x in enumerate(comms1)}
        comms2 = {idx:x for idx,x in enumerate(comms2)}

        self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(set().union([node for i, com in comms2.items() for node in com],
                                      [node for i, com in comms1.items() for node in com]))
        # J, K, N, obs, tuples1, tuples2 = self.observed()
        # exp = self.expected(J, K, N, tuples1, tuples2)
        self.omega_score = self.calc_omega()

    def get_node_assignment(self, comms):
        """
        returns a dictionary with node-cluster assignments of the form {node_id :[cluster1, cluster_3]}
        :param comms:
        :return:
        """
        nodes = {}
        for i, com in comms.items():
            for node in com:
                try:
                    nodes[node].append(i)
                except KeyError:
                    nodes[node] = [i]
        return nodes

    def num_of_common_clusters(self, u, v, nodes_dict):
        """
        return the number of clusters in which the pair u,v appears in the
        :param u:
        :param v:
        :param nodes_dict:
        :return:
        """
        try:
            _sum = len(set(nodes_dict[u]) & set(nodes_dict[v]))
        except KeyError:
            _sum = 0
        return _sum

    def calc_omega(self):
        tot = 0.
        N = 0
        for u, v in combinations(self.nodes, 2):
            tot += (self.num_of_common_clusters(u,v,self.nodes1) == self.num_of_common_clusters(u,v,self.nodes2))
            N += 1
        return tot/N


class NF1(object):
    def __init__(self, communities, ground_truth):
        self.matched_gt = {}
        self.gt_count = 0
        self.id_count = 0
        self.gt_nodes = {}
        self.id_nodes = {}
        self.communities = communities
        self.ground_truth = ground_truth
        self.prl = []
        #self.__compute_precision_recall()

    def get_f1(self):
        """

        :param prl: list of tuples (precision, recall)
        :return: a tuple composed by (average_f1, std_f1)
        """

        gt_coms = {cid: nodes for cid, nodes in enumerate(self.ground_truth)}
        ext_coms = {cid: nodes for cid, nodes in enumerate(self.communities)}

        f1_list = []
        for cid, nodes in gt_coms.items():
            tmp = [self.__compute_f1(nodes2, nodes) for _, nodes2 in ext_coms.items()]
            f1_list.append(np.max(tmp))

        f2_list = []
        for cid, nodes in ext_coms.items():
            tmp = [self.__compute_f1(nodes, nodes2) for _, nodes2 in gt_coms.items()]
            f2_list.append(np.max(tmp))

        
        # print(f1_list, f2_list)
        return (np.mean(f1_list) + np.mean(f2_list))/2

    def __compute_f1(self, c, gt):
        c = set(c)
        gt = set(gt)

        try:
            precision = len([x for x in c if x in gt ])/len(c)
            recall = len([x for x in gt if x in c ])/len(gt) 
            x, y = precision, recall
            z = 2 * (x * y) / (x + y)
            z = float("%.2f" % z)
            return z
        except ZeroDivisionError:
            return 0.


def calc_jaccard(num_vertices, result_comm_list, ground_truth_comm_list):
    def func(s1, s2):
        s1, s2 = set(s1), set(s2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    gt_coms = {cid: nodes for cid, nodes in enumerate(ground_truth_comm_list)}
    ext_coms = {cid: nodes for cid, nodes in enumerate(result_comm_list)}

    f1_list = []
    for _, nodes in gt_coms.items():
        tmp = [func(nodes2, nodes) for _, nodes2 in ext_coms.items()]
        f1_list.append(np.max(tmp))

    f2_list = []
    for _, nodes in ext_coms.items():
        tmp = [func(nodes, nodes2) for _, nodes2 in gt_coms.items()]
        f2_list.append(np.max(tmp))
    return (np.mean(f1_list) + np.mean(f2_list))/2


def calc_f1(num_vertices, result_comm_list, ground_truth_comm_list):
    # print(len(result_comm_list), len(ground_truth_comm_list))
    assert len(result_comm_list) == len(ground_truth_comm_list)
    nf = NF1(result_comm_list, ground_truth_comm_list)
    return nf.get_f1()


def calc_omega(num_vertices, result_comm_list, ground_truth_comm_list):
    # print(len(result_comm_list), len(ground_truth_comm_list))
    assert len(result_comm_list) == len(ground_truth_comm_list)
    return Omega(result_comm_list, ground_truth_comm_list).omega_score


def calc_overlap_nmi(num_vertices, result_comm_list, ground_truth_comm_list):
    assert len(result_comm_list) == len(ground_truth_comm_list)
    def write_to_file(fpath, clist):
        with open(fpath, 'w') as f:
            for c in clist:
                f.write(' '.join(map(str, c)) + '\n')

    try:
        write_to_file('./pred', result_comm_list)
        write_to_file('./gt', ground_truth_comm_list)
        assert len(result_comm_list) == len(ground_truth_comm_list)
        ret = check_output(["./bin/onmi", "pred", "gt"]).decode('utf-8')
        return float(ret.split('\n')[0].split()[-1])
    except:
        print('calc_overlap_nmi failed.\n Please refer to this repo: https://github.com/eXascaleInfolab/OvpNMI')

def calc_nonoverlap_nmi(pred_membership, gt_membership):
    from clusim.clustering import Clustering
    import clusim.sim as sim

    pred = Clustering()
    pred.from_membership_list(pred_membership)

    gt= Clustering()
    gt.from_membership_list(gt_membership)

    ret = sim.nmi(pred, gt, norm_type='sum')
    return ret
