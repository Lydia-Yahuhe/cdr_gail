import math
import os
import tempfile
import time

import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)

    def load(self, path):
        load_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          batch_size=32,
          print_freq=100,
          learning_starts=1000,
          gamma=1.0,
          reward_giver=None,
          target_network_update_freq=500,
          load_path=None,
          save_path=None,
          **network_kwargs):
    # Create all the functions necessary to train the model
    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space = env.observation_space
    num_actions = env.action_space.n

    def make_obs_ph(name_):
        return ObservationInput(observation_space, name=name_)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=False
    )

    act = ActWrapper(act, act_params={'make_obs_ph': make_obs_ph, 'q_func': q_func, 'num_actions': num_actions})

    # Create the replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    obs = env.reset()

    if load_path is not None:
        logger.log('Loaded model from {}'.format(load_path))
        # load_variables(load_path)
        # act.load(load_path)
        return act

    episode_rewards, true_rewards, expert_rewards = [0.0], [0.0], [0.0]
    losses, d_losses = [], []
    similarity, sr, mse = [], [], []

    start = time.time()
    for t in range(1, total_timesteps + 1):
        # Take action and update exploration to the newest value
        action = act(obs, update_eps=exploration.value(t))[0]
        new_obs, true_rew, done, info = env.step(action)
        num = env.scenario.id
        rew, rew_e, delta_ac = reward_giver.compare(int(num), obs, action)
        # Store transition in the replay buffer.
        replay_buffer.add(int(num), obs, action, rew, new_obs, float(done))
        obs = new_obs
        # Collect raw data for analysis
        episode_rewards[-1] += rew
        true_rewards[-1] += true_rew
        expert_rewards[-1] += rew_e
        similarity.append(float(delta_ac <= 0))
        mse.append(math.pow(delta_ac, 2))

        if t > learning_starts:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            nums, obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None
            [_, error] = train(obses_t, actions, rewards, obses_tp1, dones, weights)
            losses.append(error)
            # Update discriminator network
            d_loss = reward_giver.update(nums, obses_t, actions)
            d_losses.append(d_loss)
            # Update target network periodically.
            if t % target_network_update_freq == 0:
                update_target()

        # 测试训练集和测试集
        if (t == learning_starts or t % 5000 == 0) and save_path is not None:
            act.save(save_path + 'my_model_{}.pkl'.format(t))
            reward_giver.save(save_path + 'my_model_gan_{}.pkl'.format(t))
            # env.evaluate(act, save_path=save_path + 'dqn_actions_train_{}'.format(t), use_set='train')
            # env.evaluate(act, save_path=save_path + 'dqn_actions_test_{}'.format(t), use_set='test')

        # Change new scenario and output the processed analysis data
        if done:
            print('episode: {}'.format(t))
            obs = env.reset()
            sr.append(int(info['result']))

            if t % print_freq == 0:
                if len(d_losses) > 0:
                    for (name, loss) in zip(reward_giver.loss_name, np.mean(d_losses, axis=0)):
                        logger.record_tabular(name, loss)

                end = time.time()
                logger.record_tabular("episodes", t)
                logger.record_tabular("episode reward", np.mean(episode_rewards[-print_freq:]))
                logger.record_tabular("true reward", np.mean(true_rewards[-print_freq:]))
                logger.record_tabular("expert reward", np.mean(expert_rewards[-print_freq:]))
                logger.record_tabular("loss", np.mean(losses))
                logger.record_tabular('similarity', np.mean(similarity))
                logger.record_tabular('time', end - start)
                logger.record_tabular('sr', np.mean(sr))
                logger.record_tabular('mse', math.sqrt(np.mean(mse)))
                logger.record_tabular("exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

                losses, d_losses = [], []
                similarity, sr, mse = [], [], []
                start = end

            episode_rewards.append(0.0)
            true_rewards.append(0.0)
            expert_rewards.append(0.0)

    return act
