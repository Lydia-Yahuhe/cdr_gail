"""
The code is used to train BC imitator, or pretrained GAIL imitator
"""
import csv
from tqdm import tqdm
import tensorflow as tf

from baselines import logger
from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4, adam_epsilon=1e-5, optim_stepsize=5e-4,
          ckpt_dir=None, verbose=False):
    val_per_iter = int(max_iters / 100)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = tf.placeholder(name='expert_ac', shape=[None, ], dtype=tf.int64)
    stochastic = U.get_placeholder_cached(name="stochastic")
    # loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi.ac, labels=ac)
    loss = tf.reduce_mean(loss)
    # optimizer
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss] + [U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()

    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size)
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)

        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1)
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

            with open('bc_losses_{}.csv'.format(int(max_iters)), 'a+', newline='') as f:
                csv.writer(f).writerow([train_loss, val_loss])
            U.save_variables(ckpt_dir, variables=pi.get_variables())
