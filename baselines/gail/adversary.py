"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import tensorflow as tf
import numpy as np

from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U


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
    def __init__(self, env, hidden_size, entcoeff=0.001, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = (1, )

        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="expert_actions_ph")

        # Build graph
        generator_logits = self.__build_graph(self.generator_obs_ph, self.generator_acs_ph, hidden_size=hidden_size)
        expert_logits = self.__build_graph(self.expert_obs_ph, self.expert_acs_ph, hidden_size=hidden_size, reuse=True)

        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.losses = [generator_loss, expert_loss, entropy_loss, self.total_loss]
        self.loss_name = ["generator_loss", "expert_loss", "entropy_loss", "total_loss"]

        # Build Reward for policy
        # self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.reward_op = tf.pow(tf.nn.sigmoid(generator_logits), 3)

        var_list = self.get_trainable_variables()
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        self.lossandgrad = U.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            self.losses + [U.flatgrad(self.total_loss, var_list)])

    def __build_graph(self, obs_ph, acs_ph, hidden_size=128, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std

            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, hidden_size, activation_fn=tf.nn.relu)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, hidden_size, activation_fn=tf.nn.relu)
            p_h3 = tf.contrib.layers.fully_connected(p_h2, hidden_size, activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(p_h3, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()

        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
