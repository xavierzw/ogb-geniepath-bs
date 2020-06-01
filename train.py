from __future__ import division
from __future__ import print_function

import os
os.environ['KMP_WARNINGS'] = '0'
import time
import math
import copy
import itertools
import tensorflow as tf
import scipy.sparse as sp

from utils import *
from models import GAT, GeniePath
from cython_sampler import BanditSampler, BanditLinearSampler, BanditMPSampler
from scipy.sparse.linalg import norm as sparsenorm
from ogb.nodeproppred import Evaluator

# Set random seed
#seed = int(time.time())
#np.random.seed(seed)
#tf.set_random_seed(seed)
epsilon = 1e-6

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('model', 'gat', 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('neighbor_limit', 10, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batchsize', 256, 'Batch size.')
flags.DEFINE_integer('residual', 1, 'Residual.')
flags.DEFINE_float('eta', 0.4, 'Eta.')
flags.DEFINE_float('delta', 0.01, 'Delta.')
flags.DEFINE_string('sampler', 'bandit', 'Sampler type.')
flags.DEFINE_float('max_reward', 1.0, 'Max reward.')
flags.DEFINE_integer('num_proc', 12, 'Number of process.')


def iterate_minibatches(inputs, batchsize, shuffle=False):
    assert inputs is not None
    num_samples = inputs[-1].shape[0]
    nodes = inputs[0]
    labels = inputs[1]
    if shuffle:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
    if num_samples <= batchsize:
        yield np.array(nodes, np.int32), labels
    else:
        num_batch = int(math.ceil(float(num_samples) / batchsize))
        for idx in range(num_batch):
            if shuffle:
                if (idx+1)*batchsize < num_samples:
                    excerpt = indices[idx*batchsize:(idx+1)*batchsize]
                else:
                    excerpt = indices[idx*batchsize:]
                    excerpt = np.concatenate((excerpt,indices[:(idx+1)*batchsize-num_samples]), axis=0)
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield np.array(nodes[excerpt], np.int32), labels[excerpt]


def gen_subgraph(sampler, selected_nodes, adj, num_layer=2, neighbor_limit=10):
    edges = sampler.sample_graph(selected_nodes)
    edges = sorted(edges, key=lambda element: (element[0], element[1]))

    expand_list = set()
    for (src, dst) in edges:
        expand_list.add(src)
        expand_list.add(dst)
    for nod in selected_nodes:
        expand_list.add(nod)
    expand_list = list(expand_list)
    expand_list = sorted(expand_list)

    node_map = {}
    inverse_node_map = {}
    m_id = 0
    for nod in expand_list:
        node_map[nod] = m_id
        inverse_node_map[m_id] = nod
        m_id += 1

    src_list = []
    dst_list = []
    n2n_indices_batch=[]
    n2n_values_batch=[]
    
    sample_degree = {}
    for src in set([e[0] for e in edges]):
        sample_degree[src] = 0
    for (src, dst) in edges:
        sample_degree[src] += 1

    for (src, dst) in edges:
        n2n_indices_batch.append([node_map[src], node_map[dst]])
        src_list.append(src)
        dst_list.append(dst)
        n2n_values_batch.append(1.)
    n2n_indices_batch = np.array(n2n_indices_batch)
    n2n_values_batch = np.array(n2n_values_batch)

    left_indices_batch = [None]*len(n2n_indices_batch)
    left_values_batch = np.ones(len(n2n_indices_batch))
    right_indices_batch = [None]*len(n2n_indices_batch)
    right_values_batch = np.ones(len(n2n_indices_batch))
    ii = 0
    for n1, n2 in n2n_indices_batch:
        left_indices_batch[ii] = [ii, n1]
        right_indices_batch[ii] = [ii, n2]
        ii += 1

    node_indices_batch = []
    node_values_batch = np.ones(len(selected_nodes))
    ii = 0
    for nod in selected_nodes:
        node_indices_batch.append([ii, node_map[nod]])
        ii += 1
    node_indices_batch = np.array(node_indices_batch)
    node_values_batch = np.array(node_values_batch)

    n2n = tf.SparseTensorValue(n2n_indices_batch, n2n_values_batch, [m_id, m_id])
    left = tf.SparseTensorValue(left_indices_batch, left_values_batch, [len(left_indices_batch), m_id])
    right = tf.SparseTensorValue(right_indices_batch, right_values_batch, [len(right_indices_batch), m_id])
    node_select = tf.SparseTensorValue(node_indices_batch, node_values_batch, [len(node_indices_batch), m_id])
    return expand_list, n2n, left, right, node_select, src_list, dst_list, node_map


def gen_fullgraph(selected_nodes, adj, n2n_values_batch):
    adj_coo = adj.tocoo()
    m_id = adj.shape[0]

    n_edges = len(adj_coo.row)
    n2n_indices_batch=np.concatenate(
            [adj_coo.row[:,np.newaxis], adj_coo.col[:,np.newaxis]], axis=1)
    #n2n_values_batch=np.ones(len(adj_coo.row))

    left_indices_batch = np.concatenate(
            [np.arange(n_edges)[:,np.newaxis],
             adj_coo.row[:,np.newaxis]], axis=1)
    left_values_batch = np.ones(len(n2n_indices_batch))
    right_indices_batch = np.concatenate(
            [np.arange(n_edges)[:,np.newaxis],
             adj_coo.col[:,np.newaxis]], axis=1)
    right_values_batch = np.ones(len(n2n_indices_batch))

    node_indices_batch = np.concatenate(
            [np.arange(len(selected_nodes))[:,np.newaxis],
             selected_nodes[:,np.newaxis]], axis=1)
    node_values_batch = np.ones(len(selected_nodes))

    n2n = tf.SparseTensorValue(n2n_indices_batch, n2n_values_batch, [m_id, m_id])
    left = tf.SparseTensorValue(left_indices_batch, left_values_batch, [len(left_indices_batch), m_id])
    right = tf.SparseTensorValue(right_indices_batch, right_values_batch, [len(right_indices_batch), m_id])
    node_select = tf.SparseTensorValue(node_indices_batch, node_values_batch, [len(node_indices_batch), m_id])
    return n2n, left, right, node_select


def construct_feed_dict(n_nd, features, node_select, support, left, right, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['node_select']: node_select})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['left']: left})
    feed_dict.update({placeholders['right']: right})
    feed_dict.update({placeholders['n_nd']: n_nd})
    return feed_dict


def convert_support(support, left, right):
    indices = []
    for i in range(len(support[0])):
        indices.append([left[support[0][i][0]], right[support[0][i][1]]])
    return (np.array(indices), support[1], support[2])


def main():
    assert FLAGS.sampler in ["bandit", "bandit-linear", "bandit-mp", "random"], "Unknown sampler type"

    data_name = "_".join(FLAGS.dataset.split("-"))
    adj_full, features, train_nodes, y_train, \
        valid_nodes, y_valid, test_nodes, y_test = load_data(FLAGS.dataset)
    print("Finish loading data.")

    ## load valid & test split data
    #split_num = 10
    #valid_subgraph_nodes, \
    #valid_support, \
    #valid_left, \
    #valid_right, \
    #valid_node_select, \
    #valid_node_map, \
    #test_subgraph_nodes, \
    #test_support, \
    #test_left, \
    #test_right, \
    #test_node_select, \
    #test_node_map = load_subgraph(data_name, split_num)
    #print("Finish loading test split data.")

    # adj_train equals adj_full in ogb dataset
    adj_train = adj_full

    evaluator = Evaluator(name = FLAGS.dataset)
    eval_key = None
    if FLAGS.dataset == "ogbn-proteins":
        eval_key = "rocauc"
    elif FLAGS.dataset == "ogbn-products":
        eval_key = "acc"

    sampler = None
    if FLAGS.sampler == "bandit":
        sampler = BanditSampler()
        sampler.init(adj_train)
    elif FLAGS.sampler == "bandit-linear" or FLAGS.sampler == "random":
        sampler = BanditLinearSampler()
        sampler.init(adj_train)
    elif FLAGS.sampler == "bandit-mp":
        sampler = BanditMPSampler()
        sampler.init(adj_train)
    print("Finish init sampler.")

    ## compute full n2n
    ##n2n_values_name = "./data/{}/n2n_values_s.npy".format(FLAGS.dataset)
    #n2n_values_name = "./dataset/{}/gnn_bs/n2n_values.npy".format(FLAGS.dataset)
    #if not os.path.exists(n2n_values_name):
    #    n2n_values = []
    #    adj_full_coo = adj_full.tocoo()
    #    for src, dst in zip(adj_full_coo.row, adj_full_coo.col):
    #        degree = len(adj_full[src].rows[0])
    #        n2n_values.append(1./degree)
    #    n2n_values = np.array(n2n_values)
    #    np.save(open(n2n_values_name, "wb"), n2n_values)
    #else:
    #    n2n_values = np.load(open(n2n_values_name, "rb"))
    n2n_values = np.ones(adj_full.count_nonzero(), dtype=np.float32)

    feature_dim = features.shape[-1]
    label_dim = y_train.shape[-1]
    num_supports = 2
    numNode = adj_full.shape[0]

    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32),
        #'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None,feature_dim)),
        'node_select': tf.sparse_placeholder(tf.float32),
        'labels': tf.placeholder(tf.float32, shape=(None,label_dim)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'left': tf.sparse_placeholder(tf.float32),
        'right': tf.sparse_placeholder(tf.float32),
        'n_nd': tf.placeholder(tf.int32, shape=[]),
    }

    # Define task type
    task_type_dict = {
        "ogbn-proteins": "multi-label",
        "ogbn-products": "exclusive-label",
    }
    task_type = task_type_dict[FLAGS.dataset]

    # Create model
    model = None
    if FLAGS.model == "gat":
        model = GAT(task_type, placeholders, input_dim=features.shape[-1], label_dim=label_dim)
    elif FLAGS.model == "geniepath":
        model = GeniePath(task_type, placeholders, input_dim=features.shape[-1], label_dim=label_dim)
    else:
        print("Unknown model:", FLAGS.model)
        sys.exit()

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(feats, adj, nodes, labels, placeholders):
        preds = []

        support, left, right, node_select = gen_fullgraph(nodes, adj, n2n_values)

        # Construct feed dictionary
        feed_dict = construct_feed_dict(
                adj.shape[0], feats, node_select, support, left, right, labels, placeholders)
        feed_dict.update({placeholders['dropout']: 0.})

        outs = sess.run([model.outputs], feed_dict=feed_dict)
        preds.extend(outs[0].tolist())
        eval_true = np.array(labels)
        eval_pred = np.array(preds)

        # evaluate
        if task_type == "exclusive-label":
            eval_true = np.argmax(eval_true, axis=1).reshape([-1,1])
            eval_pred = np.argmax(eval_pred, axis=1).reshape([-1,1])
        eval_res = evaluator.eval({"y_true": eval_true, "y_pred": eval_pred})[eval_key]

        return eval_res

    # Define model evaluation function
    def inc_evaluate(mode, feats, adj, nodes, labels, placeholders):
        preds = []
    
        num_parts = 10
        part_size = int(len(nodes)/float(num_parts))
        splits = []
        for i in range(num_parts):
            splits.append(part_size*i)
        splits.append(len(nodes))

        node_map = None
        subgraph_nodes = None
        node_select = None
        support = None
        left = None
        right = None
        for i in range(num_parts):
            part_nodes = np.array(nodes[splits[i]:splits[i+1]], np.int32)
            part_labels = labels[splits[i]:splits[i+1], :]

            # Get split subgraph
            if mode == "valid":
                node_map = valid_node_map[i]
                subgraph_nodes = valid_subgraph_nodes[i]
                node_select = valid_node_select[i]
                support = valid_support[i]
                left = valid_left[i]
                right = valid_right[i]
            elif mode == "test":
                node_map = test_node_map[i]
                subgraph_nodes = test_subgraph_nodes[i]
                node_select = test_node_select[i]
                support = test_support[i]
                left = test_left[i]
                right = test_right[i]

            # Construct feed dictionary
            features_inputs = features[subgraph_nodes, :]
            feed_dict = construct_feed_dict(
                    len(node_map), features_inputs, node_select, support, left, right, part_labels, placeholders)
            feed_dict.update({placeholders['dropout']: 0.})

            outs = sess.run([model.outputs], feed_dict=feed_dict)
            preds.extend(outs[0].tolist())

        eval_true = np.array(labels)
        eval_pred = np.array(preds)

        # evaluate
        if task_type == "exclusive-label":
            eval_true = np.argmax(eval_true, axis=1).reshape([-1,1])
            eval_pred = np.argmax(eval_pred, axis=1).reshape([-1,1])
        eval_res = evaluator.eval({"y_true": eval_true, "y_pred": eval_pred})[eval_key]

        return eval_res

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    best_eval = 0.
    train_true = []
    train_pred = []

    last_time = time.time()
    print("train_size =", len(train_nodes))
    # Train model
    for epoch in range(FLAGS.epochs):
        n = 0
        train_losses = []
        for batch in iterate_minibatches(
                [train_nodes, y_train], batchsize=FLAGS.batchsize, shuffle=True):
            batch_nodes, y_batch = batch

            subgraph_nodes, support, left, right, node_select, src_list, dst_list, node_map = \
                gen_subgraph(sampler, batch_nodes, adj_train, neighbor_limit=FLAGS.neighbor_limit)

            features_inputs = features[subgraph_nodes, :]

            # Construct feed dictionary
            feed_dict = construct_feed_dict(
                    len(node_map), features_inputs, node_select, support, left, right, y_batch, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.sparse_attention_l0, model.outputs], feed_dict=feed_dict)
            train_losses.append(outs[1])

            # Update sample probs
            if FLAGS.sampler != "random":
                sampler.update(np.array(src_list, dtype=np.int32), np.array(dst_list, dtype=np.int32), outs[2])

            train_true.extend(y_batch.tolist())
            train_pred.extend(outs[3].tolist())

        # compute Train eval
        if task_type == "exclusive-label":
            train_true = np.argmax(train_true, axis=1).reshape([-1,1])
            train_pred = np.argmax(train_pred, axis=1).reshape([-1,1])
        eval_tr = evaluator.eval({"y_true": np.array(train_true), "y_pred": np.array(train_pred)})[eval_key]
        train_true = []
        train_pred = []

        # Valid
        eval_va = evaluate(features, adj_full, valid_nodes, y_valid, placeholders)
        eval_te = evaluate(features, adj_full, test_nodes, y_test, placeholders)
        print("Epoch:", '%04d' % (epoch + 1),
              "loss=", "{:.5f}".format(np.mean(train_losses)),
              "{}_tr=".format(eval_key), "{:.5f}".format(eval_tr),
              "{}_va=".format(eval_key), "{:.5f}".format(eval_va),
              "{}_te=".format(eval_key), "{:.5f}".format(eval_te),
              "time=", "{:.2f}".format(time.time() - last_time))
        last_time = time.time()

        ## Save model
        #if eval_va > best_eval:
        #    best_eval = eval_va
        #    if not os.path.exists("./save_models/%s" % FLAGS.dataset):
        #        os.mkdir("./save_models/%s" % FLAGS.dataset)
        #    saver.save(sess, "./save_models/{}/{}_{}.ckpt".format(
        #                FLAGS.dataset, FLAGS.model, FLAGS.sampler))

    ## Testing
    #saver.restore(sess, "./save_models/{}/{}_{}.ckpt".format(
    #              FLAGS.dataset, FLAGS.model, FLAGS.sampler))
    #eval_te = inc_evaluate(features, adj_full, test_nodes, y_test, placeholders)
    #print("eval_te=", "{:.5f}".format(eval_te))


if __name__=="__main__":
    print("DATASET:", FLAGS.dataset)
    main()
