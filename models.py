from layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def attention_mechanism(name, v, W_s, W_d, V, cur_embed, left, right, n2n):
    # a_{i,j} \propto v^\top tanh (W_s (\mu_i + \mu_j))
    if name == 'linear':
        t = tf.sparse_tensor_dense_matmul(sp_a=edge, b=cur_embed) # edge \in \R^{m, n}
        t = tf.matmul(t, W_s) # m by 16
        t = tf.nn.tanh(t)
        t = tf.matmul(t, tf.reshape(v, [-1,1])) # m by 1
        sparse_attention = tf.SparseTensor(n2n.indices, tf.reshape(t, [-1]), n2n.dense_shape)
        sparse_attention = tf.sparse_softmax(sparse_attention)
    # a_{i,j} \propto v^\top tanh (W_s |\mu_i - \mu_j|)
    elif name == 'abs':
        t = tf.sparse_tensor_dense_matmul(sp_a=edge, b=cur_embed) # edge \in \R^{m, n}
        t = tf.abs(t)
        t = tf.matmul(t, W_s) # m by 16
        t = tf.nn.tanh(t)
        t = tf.matmul(t, tf.reshape(v, [-1,1])) # m by 1
        sparse_attention = tf.SparseTensor(n2n.indices, tf.reshape(t, [-1]), n2n.dense_shape)
        sparse_attention = tf.sparse_softmax(sparse_attention)
    # a_{i,j} \propto leakyrelu (\mu_i V \mu_j)
    elif name == 'bilinear':
        tl = tf.sparse_tensor_dense_matmul(sp_a=left, b=cur_embed) # m by k
        tl = tf.matmul(tl, V)
        tr = tf.sparse_tensor_dense_matmul(sp_a=right, b=cur_embed)
        t = tf.reduce_sum(tf.multiply(tl, tr), 1, keep_dims=True)
        t = tf.keras.layers.LeakyReLU(t)
        sparse_attention = tf.SparseTensor(n2n.indices, tf.reshape(t, [-1]), n2n.dense_shape)
        sparse_attention = tf.sparse_softmax(sparse_attention)
    # a_{i,j} \propto v^\top tanh (W_s \mu_i + W_d \mu_j)
    if name == 'generalized_linear':
        tl = tf.sparse_tensor_dense_matmul(sp_a=left, b=cur_embed) # m by k
        tl = tf.matmul(tl, W_s)
        tr = tf.sparse_tensor_dense_matmul(sp_a=right, b=cur_embed)
        tr = tf.matmul(tr, W_d)
        t = tf.nn.tanh(tf.add(tl,tr))
        t = tf.matmul(t, tf.reshape(v, [-1,1]))
        sparse_attention = tf.SparseTensor(n2n.indices, tf.reshape(t, [-1]), n2n.dense_shape)
        sparse_attention = tf.sparse_softmax(sparse_attention)
    else:
        sys.exit(-1)
    return sparse_attention


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    if len(shape)==2:
        init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    elif len(shape)==1:
        init_range = np.sqrt(6.0/shape[0])
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.task_type = None
        self.vars = {}
        self.placeholders = {}

        self.label_dim = None
        self.inputs = None
        self.layers = []
        self.activations = []
        self.outputs = None

        self.sparse_attention_l0 = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        #self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GeniePath(Model):
    def __init__(self, task_type, placeholders, input_dim, label_dim, **kwargs):
        super(GeniePath, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        assert task_type in ["exclusive-label", "multi-label"], "Unknown task type!"
        self.task_type = task_type
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        #for i in range(len(self.layers)):
        #    for var in self.layers[i].vars.values():
        #        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        l2_loss = 0
        for i in range(2):
            l2_loss += tf.nn.l2_loss(self.vars_wn[i])
            l2_loss += tf.nn.l2_loss(self.vars_bn[i])
            l2_loss += tf.nn.l2_loss(self.vars_ws[i])
            l2_loss += tf.nn.l2_loss(self.vars_wd[i])
            l2_loss += tf.nn.l2_loss(self.vars_v[i])
            l2_loss += tf.nn.l2_loss(self.vars_V[i])
            l2_loss += tf.nn.l2_loss(self.vars_wi[i])
            l2_loss += tf.nn.l2_loss(self.vars_wf[i])
            l2_loss += tf.nn.l2_loss(self.vars_wo[i])
            l2_loss += tf.nn.l2_loss(self.vars_wc[i])
            l2_loss += tf.nn.l2_loss(self.vars_bc[i])
            l2_loss += tf.nn.l2_loss(self.vars_bo[i])
            l2_loss += tf.nn.l2_loss(self.vars_bf[i])
            l2_loss += tf.nn.l2_loss(self.vars_bi[i])
        l2_loss += tf.nn.l2_loss(self.W_x)
        l2_loss += tf.nn.l2_loss(self.b_x)
        l2_loss += tf.nn.l2_loss(self.v_o)
        l2_loss += tf.nn.l2_loss(self.ws_o)
        l2_loss += tf.nn.l2_loss(self.wd_o)
        l2_loss += tf.nn.l2_loss(self.V_o)
        l2_loss += tf.nn.l2_loss(self.wn_o)
        l2_loss += tf.nn.l2_loss(self.b_o)
        self.loss += FLAGS.weight_decay * l2_loss

        # Cross entropy error
        if self.task_type == "exclusive-label":
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.placeholders['labels'],
                        logits=self.outputs))
        else:  # multi-label
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.placeholders['labels'],
                        logits=self.outputs))

    def _build(self):
        # placeholder
        self.n2n=self.placeholders['support']
        self.node_feat=self.placeholders['features']
        self.node_select=self.placeholders['node_select']
        self.left=self.placeholders['left']
        self.right=self.placeholders['right']
        self.n_nd=self.placeholders['n_nd']

        # parameters
        hidden_dim = FLAGS.hidden1
        input_dim=self.input_dim
        label_dim = self.label_dim
        self.vars_wn=[]
        self.vars_bn=[]
        self.vars_ws=[]
        self.vars_wd=[]
        self.vars_v=[]
        self.vars_V=[]
        self.vars_wi=[]
        self.vars_wf=[]
        self.vars_wo=[]
        self.vars_wc=[]
        self.vars_bc=[]
        self.vars_bo=[]
        self.vars_bf=[]
        self.vars_bi=[]
        collector=[]
        for i in range(2):
            self.vars_wn.append(glorot([hidden_dim, hidden_dim], 'W_n_%d'%i))
            self.vars_bn.append(tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_n_%d'%i))
            self.vars_ws.append(glorot([hidden_dim, hidden_dim], 'W_s_%d'%i))
            self.vars_wd.append(glorot([hidden_dim, hidden_dim], 'W_d_%d'%i))
            self.vars_v.append(glorot([hidden_dim], 'v_%d'%i))
            self.vars_V.append(glorot([hidden_dim, hidden_dim], 'V_%d'%i))

            self.vars_wi.append(glorot([hidden_dim*2, hidden_dim], 'W_i_%d'%i))
            self.vars_wf.append(glorot([hidden_dim*2, hidden_dim], 'W_f_%d'%i))
            self.vars_wo.append(glorot([hidden_dim*2, hidden_dim], 'W_o_%d'%i))
            self.vars_wc.append(glorot([hidden_dim*2, hidden_dim], 'W_c_%d'%i))

            self.vars_bc.append(tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_c_%d'%i))
            self.vars_bo.append(tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_o_%d'%i))
            self.vars_bf.append(tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_f_%d'%i))
            self.vars_bi.append(tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_i_%d'%i))
        self.W_x = glorot([input_dim, hidden_dim], 'W_x')
        self.b_x = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_x')
        self.v_o = glorot([hidden_dim], 'v_o')
        self.ws_o = glorot([hidden_dim, hidden_dim], 'W_s_o')
        self.wd_o = glorot([hidden_dim, hidden_dim], 'W_d_o')
        self.V_o = glorot([hidden_dim, hidden_dim], 'V_o')
        self.wn_o = glorot([hidden_dim, label_dim], 'W_n_o')
        self.b_o = tf.Variable(tf.zeros([label_dim], dtype=tf.float32), name='b_o')

        # inference
        #self.node_feat = tf.nn.dropout(self.node_feat, rate=1-self.keep_prob)
        node_embed = tf.matmul(self.node_feat, self.W_x) + self.b_x
        cur_embed = node_embed
        C = tf.zeros([self.n_nd, hidden_dim], tf.float32)

        for i in range(2):
            cur_embed = tf.nn.dropout(cur_embed, rate=FLAGS.dropout)

            # build sparse attention matrix a_{i,j}
            sparse_attention = attention_mechanism(
                    "generalized_linear", self.vars_v[i], self.vars_ws[i], self.vars_wd[i],
                    self.vars_V[i], cur_embed, self.left, self.right, self.n2n)
            if i == 0:
                self.sparse_attention_l0 = sparse_attention.values

            # propagation
            n2npool = tf.sparse_tensor_dense_matmul(sp_a=sparse_attention, b=cur_embed)
            node_linear = tf.matmul(n2npool, self.vars_wn[i]) + self.vars_bn[i]

            if FLAGS.residual == 1:
                merged_linear = tf.add(node_linear, node_embed)
            else:
                merged_linear = node_linear

            cur_embed = tf.nn.tanh(merged_linear)
            #cur_embed = tf.nn.dropout(cur_embed, rate=1-self.keep_prob)
            collector.append(cur_embed)

        for i in range(len(collector)):
            input_gate  = tf.nn.sigmoid(tf.matmul(tf.concat([collector[i], node_embed], 1), self.vars_wi[i])+self.vars_bi[i])
            forget_gate = tf.nn.sigmoid(tf.matmul(tf.concat([collector[i], node_embed], 1), self.vars_wf[i])+self.vars_bf[i])
            output_gate = tf.nn.sigmoid(tf.matmul(tf.concat([collector[i], node_embed], 1), self.vars_wo[i])+self.vars_bo[i])
            C_update = tf.nn.tanh(tf.matmul(tf.concat([collector[i], node_embed], 1), self.vars_wc[i])+self.vars_bc[i])
            C = tf.add(tf.multiply(forget_gate, C), tf.multiply(input_gate, C_update))
            node_embed = tf.multiply(output_gate, tf.nn.tanh(C))
        node_embed = tf.matmul(node_embed, self.wn_o)+self.b_o
        self.outputs = tf.sparse_tensor_dense_matmul(sp_a=self.node_select, b=node_embed)


class GAT(Model):
    def __init__(self, task_type, placeholders, input_dim, label_dim, **kwargs):
        super(GAT, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        assert task_type in ["exclusive-label", "multi-label"], "Unknown task type!"
        self.task_type = task_type
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        #for i in range(len(self.layers)):
        #    for var in self.layers[i].vars.values():
        #        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for i in range(2):
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.vars_wn[i])
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.vars_bn[i])
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.vars_ws[i])
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.vars_wd[i])
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.vars_v[i])
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.vars_V[i])
        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.W_x)
        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.b_x)
        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.wn_o)
        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.b_o)

        # Cross entropy error
        if self.task_type == "exclusive-label":
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.placeholders['labels'],
                        logits=self.outputs))
        else:  # multi-label
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.placeholders['labels'],
                        logits=self.outputs))

    def _build(self):
        # placeholder
        self.n2n=self.placeholders['support']
        self.node_feat=self.placeholders['features']
        self.node_select=self.placeholders['node_select']
        self.left=self.placeholders['left']
        self.right=self.placeholders['right']
        self.n_nd=self.placeholders['n_nd']

        # parameters
        hidden_dim = FLAGS.hidden1
        input_dim=self.input_dim
        label_dim = self.label_dim
        self.vars_wn=[]
        self.vars_bn=[]
        self.vars_ws=[]
        self.vars_wd=[]
        self.vars_v=[]
        self.vars_V=[]
        for i in range(2):
            if i == 0:
                self.vars_wn.append(glorot([hidden_dim, hidden_dim], 'W_n_%d'%i))
                self.vars_bn.append(tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_n_%d'%i))
                self.vars_ws.append(glorot([hidden_dim, hidden_dim], 'W_s_%d'%i))
                self.vars_wd.append(glorot([hidden_dim, hidden_dim], 'W_d_%d'%i))
                self.vars_v.append(glorot([hidden_dim], 'v_%d'%i))
                self.vars_V.append(glorot([hidden_dim, hidden_dim], 'V_%d'%i))
            else:
                self.vars_wn.append(glorot([hidden_dim, hidden_dim], 'W_n_%d'%i))
                self.vars_bn.append(tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_n_%d'%i))
                self.vars_ws.append(glorot([hidden_dim, hidden_dim], 'W_s_%d'%i))
                self.vars_wd.append(glorot([hidden_dim, hidden_dim], 'W_d_%d'%i))
                self.vars_v.append(glorot([hidden_dim], 'v_%d'%i))
                self.vars_V.append(glorot([hidden_dim, hidden_dim], 'V_%d'%i))

        self.W_x = glorot([input_dim, hidden_dim], 'W_x')
        self.b_x = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32), name='b_x')
        self.wn_o = glorot([hidden_dim, label_dim], 'W_n_o')
        self.b_o = tf.Variable(tf.zeros([label_dim], dtype=tf.float32), name='b_o')

        # inference
        #self.node_feat = tf.nn.dropout(self.node_feat, rate=1-self.keep_prob)
        node_embed = tf.matmul(self.node_feat, self.W_x)+self.b_x
        #node_embed = self.node_feat
        cur_embed = node_embed

        bp = 2
        for i in range(bp):
            cur_embed = tf.nn.dropout(cur_embed, rate=FLAGS.dropout)

            # build sparse attention matrix a_{i,j}
            sparse_attention = attention_mechanism(
                    "generalized_linear", self.vars_v[i], self.vars_ws[i], self.vars_wd[i],
                    self.vars_V[i], cur_embed, self.left, self.right, self.n2n)
            if i == 0:
                self.sparse_attention_l0 = sparse_attention.values

            # propagation
            n2npool = tf.sparse_tensor_dense_matmul(sp_a=sparse_attention, b=cur_embed)
            node_linear = tf.matmul(n2npool, self.vars_wn[i])+self.vars_bn[i]

            if FLAGS.residual == 1:
                merged_linear = tf.add(node_linear, node_embed)
            else:
                merged_linear = node_linear

            cur_embed = tf.nn.tanh(merged_linear)
            #cur_embed = tf.nn.dropout(cur_embed, rate=1-self.keep_prob)

        node_embed = tf.matmul(cur_embed, self.wn_o)+self.b_o
        self.outputs = tf.sparse_tensor_dense_matmul(sp_a=self.node_select, b=node_embed)
