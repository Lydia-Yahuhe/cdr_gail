"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
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
        self.n_action = env.action_space.n
        self.observation_shape = env.observation_space.shape
        self.actions_shape = (self.n_action,)

        self.expert_data_set = data_set

        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="obs_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="act_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="expert_obs_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="expert_act_ph")

        # Build graph
        generator_logits = self.__build_graph(self.generator_obs_ph, self.generator_acs_ph)
        expert_logits = self.__build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)

        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=generator_logits,
            labels=tf.zeros_like(generator_logits)
        )
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=expert_logits,
            labels=tf.ones_like(expert_logits)
        )
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -ent_coeff * entropy

        # Loss + Accuracy terms
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.loss_name = ["generator_loss", "expert_loss", "entropy_loss", "total_loss"]

        # Build Reward for policy
        # self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.reward_op = -tf.pow(1 - tf.nn.sigmoid(generator_logits), 0.5)

        var_list = self.get_trainable_variables()
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        self.lossandgrad = U.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            [generator_loss, expert_loss, entropy_loss, self.total_loss, U.flatgrad(self.total_loss, var_list)])

    def __build_graph(self, obs_ph, acs_ph, reuse=False, **kwargs):
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
            x = fc(x, 'fc', nh=512 - self.n_action, init_scale=np.sqrt(2))
            combined = tf.concat([x, acs_ph], axis=1)
            x = activ(fc(combined, 'fc1', nh=64, init_scale=np.sqrt(2)))
            logits = tf.identity(fc(x, 'fc2', nh=1, init_scale=np.sqrt(2)))
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def save(self, save_path):
        U.save_variables(save_path)

    def update(self, num_batch, ob_batch, ac_batch):
        ob_expert, ac_expert = self.expert_data_set.get_next_batch(batch_samples=num_batch)
        *new_losses, g = self.lossandgrad(
            ob_batch,
            np.eye(self.n_action)[ac_batch],
            ob_expert,
            np.eye(self.n_action)[ac_expert]
        )
        self.adam.update(g, 5e-4)
        return new_losses

    def compare(self, num, obs, ac):
        obs_e, ac_e = self.expert_data_set.get_action(num)
        rew = self.get_reward(obs, np.eye(self.n_action)[ac])[0][0]
        rew_e = self.get_reward(obs_e, np.eye(self.n_action)[ac_e])[0][0]
        print('{:>+7.4f}, {:>+7.4f}, {:>4d}, {:>4d}, {}'.format(
            rew, rew_e, ac, ac_e, int((obs_e == obs).all())),
            end=' | ')
        return rew, rew_e, abs(ac - ac_e)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()

        if len(obs.shape) == 3:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
