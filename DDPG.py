import tensorflow as tf
import numpy as np
import os
import shutil

LOAD_MODEL = False

LR_A = 0.01    # learning rate for actor
LR_C = 0.05     # learning rate for critic
GAMMA = 0.9      # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1500
BATCH_SIZE = 128*3


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory_capacity = MEMORY_CAPACITY
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.DATA_PATH = './model_data'
        if LOAD_MODEL:
            all_ckpt = tf.train.get_checkpoint_state('./model_data', 'checkpoint').all_model_checkpoint_paths
            self.saver.restore(self.sess, all_ckpt[-1])
            print('Saved model used.')
        else:
            if os.path.isdir(self.DATA_PATH): shutil.rmtree(self.DATA_PATH)
            os.mkdir(self.DATA_PATH)
            self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement

        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        # print('ddpg.pointer:',self.pointer)

    def _build_a(self, s, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s, 256, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l1', trainable=trainable)
            #net2 = tf.layers.dense(net1, 256, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)
            a = tf.layers.dense(net1, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, bias_initializer=init_b, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            #net2 = tf.layers.dense(net1, 256, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)
            return tf.layers.dense(net1, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable) # Q(s,a)

    def save_model(self,step_count):
        ckpt_path = os.path.join(self.DATA_PATH, 'DDPG.ckpt')
        save_path = self.saver.save(self.sess, ckpt_path, global_step=step_count, write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)