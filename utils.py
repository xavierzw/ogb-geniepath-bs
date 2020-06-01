import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from scipy.sparse.linalg.eigen.arpack import eigsh
import os
import sys
import time
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import norm as sparsenorm
from scipy.linalg import qr
from sklearn.metrics import f1_score
from ogb.nodeproppred import NodePropPredDataset


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    if len(shape) == 2:
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    elif len(shape) == 1:
        init_range = np.sqrt(6.0 / shape[0])
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.get_variable(initializer=initial, name=name)


def calc_f1(task_type, y_true, y_pred):
    if task_type == "multi-label":
        #y_pred[y_pred > 0.5] = 1
        #y_pred[y_pred <= 0.5] = 0
        y_pred[y_pred > 0.] = 1
        y_pred[y_pred <= 0.] = 0
        return f1_score(y_true, y_pred, average="micro")
    else:
        y_pred = np.argmax(y_pred, 1)
        y_true = np.argmax(y_true, 1)
        f1_micro = f1_score(y_true, y_pred, average="micro")
        return f1_micro


def accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, 1)
    y_true = np.argmax(y_true, 1)
    return np.sum(y_pred == y_true) / float(len(y_true))


def preprocess(name, root, dataset):
    splitted_idx = dataset.get_idx_split()
    graph = dataset[0][0]
    labels = dataset[0][1]

    # create gnn_bs path
    dir_gnn_bs = os.path.join(root, "gnn_bs")
    os.mkdir(dir_gnn_bs)

    num_node = len(graph['node_feat'])
    edge_index = graph['edge_index']
    edge_feat = graph['edge_feat']
    num_edge = len(edge_index[0])
    feature_dim = len(graph['node_feat'][0])
    train_indices = splitted_idx['train']

    # add self-loop edges
    edge_index_list = []
    edge_index_list.append(edge_index[0].tolist())
    edge_index_list.append(edge_index[1].tolist())
    for node in range(num_node):
        edge_index_list[0].append(node)
        edge_index_list[1].append(node)
        num_edge += 1

    # adj_full
    adj_full = sp.coo_matrix((np.ones(num_edge), (edge_index_list[0], edge_index_list[1])),
                             shape=(num_node, num_node))
    sp.save_npz("{}/adj_full.npz".format(dir_gnn_bs), sp.csr_matrix(adj_full))

    # node feats
    node_feat = None
    if name == "ogbn-proteins":
        # average edge feats to get node feats
        node_feat_list = []
        adj_sum = adj_full.sum(axis=1)
        print("generate node feats...")
        for i in tqdm(range(edge_feat.shape[1])):
            edge_feat_i = sp.coo_matrix((edge_feat[:,i], (edge_index[0], edge_index[1])),
                                        shape=(num_node, num_node))
            node_feat_list.append(edge_feat_i.sum(axis=1)/adj_sum)
        node_feat = np.concatenate(node_feat_list, axis=1)
    elif name == "ogbn-products":
        node_feat = graph['node_feat']
    else:
        print("unknown data name: {}".format(name))
        sys.exit(0)

    # labels
    if name == "ogbn-products":
        label_idx = labels.reshape([-1])
        labels = np.zeros((label_idx.size, label_idx.max()+1))
        labels[np.arange(label_idx.size), label_idx] = 1
    np.save(open("{}/labels.npy".format(dir_gnn_bs), "wb"), labels)

    # feats
    np.save(open("{}/feats.npy".format(dir_gnn_bs), "wb"), node_feat)

    # splitted_idx
    np.save(open("{}/splitted_idx.npy".format(dir_gnn_bs), "wb"), splitted_idx)

def load_data(name):
    dir_name = "_".join(name.split("-"))
    root = os.path.join("dataset", dir_name)
    dir_gnn_bs = os.path.join(root, "gnn_bs")

    if not os.path.exists(root):
        os.mkdir(root)

    if not os.path.exists(dir_gnn_bs):
        dataset = NodePropPredDataset(name)
        print("data preprocess...")
        preprocess(name, root, dataset)
        adj_full = sp.load_npz('{}/adj_full.npz'.format(dir_gnn_bs)).astype(np.bool)
    else:
        adj_full = sp.load_npz('{}/adj_full.npz'.format(dir_gnn_bs)).astype(np.bool)
    splitted_idx = np.load('{}/splitted_idx.npy'.format(dir_gnn_bs), allow_pickle=True).item()
    feats = np.load('{}/feats.npy'.format(dir_gnn_bs))
    labels = np.load('{}/labels.npy'.format(dir_gnn_bs))

    ## ---- normalize feats ----
    #if dataset == "reddit" or dataset == "flickr" or dataset == "yelp" or dataset == "amazon" or dataset == "ppi-large":
    #    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    #    train_feats = feats[train_nodes]
    #    scaler = StandardScaler()
    #    scaler.fit(train_feats)
    #    feats = scaler.transform(feats)

    adj_full = adj_full.tolil()
    train_nodes = splitted_idx['train']
    y_train = labels[train_nodes]
    valid_nodes = splitted_idx['valid']
    y_valid = labels[valid_nodes]
    test_nodes = splitted_idx['test']
    y_test = labels[test_nodes]

    return adj_full, feats, train_nodes, y_train, \
           valid_nodes, y_valid, test_nodes, y_test

