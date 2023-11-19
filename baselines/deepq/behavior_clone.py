"""
The code is used to train BC imitator, or pretrained GAIL imitator
"""
import numpy as np
import tensorflow as tf

from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.a2c.utils import conv, fc, conv_to_fc


class BehaviorClone:
    def __init__(self, env, data_set, scope="adversary"):
        self.scope = scope
        self.env = env
        self.n_action = env.action_space.n
        self.observation_shape = env.observation_space.shape
        self.actions_shape = (1, )
        self.expert_data_set = data_set
        self.train_set = [int(info['id']) for info in env.train_set]
        self.test_set = [int(info['id']) for info in env.test_set]

        # placeholder
        self.e_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="expert_obs_ph")
        self.e_acs_ph = tf.placeholder(tf.int32, (None,), name="expert_act_ph")
        # graph
        logits = self.__build_graph(self.e_obs_ph)
        # loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.e_acs_ph)
        loss = tf.reduce_mean(loss)
        # update
        var_list = self.get_trainable_variables()
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        self.lossandgrad = U.function(
            [self.e_obs_ph, self.e_acs_ph],
            [loss, U.flatgrad(loss, var_list)]
        )
        self.action = U.function([self.e_obs_ph, ], [tf.nn.softmax(logits), ])

    def __build_graph(self, obs_ph, reuse=False, **kwargs):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            scaled_x = tf.cast(obs_ph/255., tf.float32)
            activ = tf.nn.relu

            x = activ(conv(scaled_x, 'c1', nf=4, rf=9, stride=2, init_scale=np.sqrt(2), **kwargs))
            x = activ(conv(x, 'c2', nf=8, rf=7, stride=2, init_scale=np.sqrt(2), **kwargs))
            x = activ(conv(x, 'c3', nf=16, rf=5, stride=2, init_scale=np.sqrt(2), **kwargs))
            x = activ(conv(x, 'c4', nf=32, rf=5, stride=1, init_scale=np.sqrt(2), **kwargs))

            x = conv_to_fc(x)
            x = fc(x, 'fc', nh=512, init_scale=np.sqrt(2))
            logits = tf.identity(fc(x, 'fc1', nh=self.n_action, init_scale=np.sqrt(2)))
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def load(self, path):
        U.load_variables(path)
        return self.act

    def learn(self, batch_size=128, max_iters=1e4, lr=5e-4, verbose=False, save_path=None):
        U.initialize()
        self.adam.sync()

        dataset = self.expert_data_set
        print("Pretraining with Behavior Cloning...")
        for iter_so_far in range(int(max_iters+1)):
            np.random.shuffle(self.train_set)

            ob_expert, ac_expert = dataset.get_next_batch(self.train_set[:batch_size])
            [train_loss, g, *_] = self.lossandgrad(ob_expert, ac_expert)

            if iter_so_far > 0:
                self.adam.update(g, lr)

            if verbose and iter_so_far % 5000 == 0:
                ob_expert, ac_expert = dataset.get_next_batch(self.test_set)
                val_loss, _ = self.lossandgrad(ob_expert, ac_expert)
                print("Iter: {}, Train loss: {}, Val loss: {}".format(iter_so_far, train_loss, val_loss))
                U.save_variables(save_path + 'my_model_{}.pkl'.format(iter_so_far))

    def act(self, obs):
        if len(obs.shape) == 3:
            obs = np.expand_dims(obs, 0)
        return self.action(obs)
