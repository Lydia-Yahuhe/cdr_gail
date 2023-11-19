"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
# import tensorflow as tf
import numpy as np

from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util as U
from baselines.a2c.utils import conv, fc, conv_to_fc

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def logsigmoid(a):
    """Equivalent to tf.log(tf.sigmoid(a))"""
    return -tf.nn.softplus(-a)


"""
Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51
"""


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, env, data_set, ent_coeff=0.001, scope="adversary"):
        self.scope = scope
        # obs_shape = list(env.observation_space.shape)
        # obs_shape[-1] *= 2
        self.observation_shape = env.observation_space.shape
        print(self.observation_shape)

        self.expert_data_set = data_set
        # Build placeholder
        self.g_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="obs_ph")
        self.e_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="expert_obs_ph")
        # Build graph
        g_logits = self.__build_graph(self.g_obs_ph)
        e_logits = self.__build_graph(self.e_obs_ph, reuse=True)
        # Build regression loss
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=g_logits, labels=tf.zeros_like(g_logits))
        g_loss = tf.reduce_mean(g_loss)
        e_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=e_logits, labels=tf.ones_like(e_logits))
        e_loss = tf.reduce_mean(e_loss)
        # Build entropy loss
        logits = tf.concat([g_logits, e_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        ent_loss = -ent_coeff * entropy
        # Loss terms
        t_loss = g_loss + e_loss + ent_loss
        # Build Reward for policy
        # self.reward_op = -tf.log(1 - tf.nn.sigmoid(g_logits) + 1e-8)
        self.reward_op = -tf.pow(1-tf.nn.sigmoid(g_logits), 0.5)

        var_list = self.get_trainable_variables()
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        self.loss_name = ["g_loss", "expert_loss", "ent_loss", "t_loss"]
        self.lossandgrad = U.function(
            [self.g_obs_ph, self.e_obs_ph],
            [g_loss, e_loss, ent_loss, t_loss, U.flatgrad(t_loss, var_list)]
        )

    def __build_graph(self, obs_ph, reuse=False, **kwargs):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            scaled_x = tf.cast(obs_ph / 255., tf.float32)
            activ = tf.nn.relu
            x = activ(conv(scaled_x, 'c1', nf=4, rf=9, stride=2, init_scale=np.sqrt(2), **kwargs))
            x = activ(conv(x, 'c2', nf=8, rf=7, stride=2, init_scale=np.sqrt(2), **kwargs))
            x = activ(conv(x, 'c3', nf=16, rf=5, stride=2, init_scale=np.sqrt(2), **kwargs))
            x = activ(conv(x, 'c4', nf=32, rf=5, stride=1, init_scale=np.sqrt(2), **kwargs))
            x = conv_to_fc(x)
            x = activ(fc(x, 'fc', nh=512, init_scale=np.sqrt(2)))
            x = activ(fc(x, 'fc1', nh=64, init_scale=np.sqrt(2)))
            logits = tf.identity(fc(x, 'fc2', nh=1, init_scale=np.sqrt(2)))
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def update(self, num_batch, ob_batch, n_ob_batch):
        # ob_batch_two = np.concatenate([ob_batch, n_ob_batch], axis=-1)[:]
        ob_batch_two = n_ob_batch - ob_batch
        ob_expert_two, _ = self.expert_data_set.get_next_batch(batch_samples=num_batch)
        *new_losses, g = self.lossandgrad(ob_batch_two, ob_expert_two)
        self.adam.update(g, 1e-4)
        return new_losses

    def compare(self, num, obs, ac, n_obs):
        obs_e, ac_e = self.expert_data_set.get_action(num)
        # ob_batch_two = np.concatenate([obs, n_obs], axis=-1)
        ob_batch_two = n_obs - obs
        rew = self.get_reward(ob_batch_two)[0][0]
        rew_e = self.get_reward(obs_e)[0][0]
        print('{:>+7.4f}, {:>+7.4f}, {:>4d}, {:>4d}'.format(rew, rew_e, ac, ac_e), end=' | ')
        return rew, rew_e, abs(ac - ac_e)

    def get_reward(self, obs):
        sess = tf.get_default_session()

        if len(obs.shape) == len(self.observation_shape):
            obs = np.expand_dims(obs, 0)

        reward = sess.run(self.reward_op, feed_dict={self.g_obs_ph: obs})
        return reward
