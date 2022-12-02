from abc import ABC

import gym
from gym import spaces
from tqdm import tqdm

from flightEnv.scene import ConflictScene
from flightEnv.cmd import CmdCount, reward_for_cmd

from flightSim.load import load_and_split_data
from flightSim.visual import *


def calc_reward(solved, cmd_info):
    if not solved:  # failed
        reward = -5.0
    else:  # solved
        rew = reward_for_cmd(cmd_info)
        reward = 0.5+min(rew, 0)

    print('{:>+4.2f}'.format(reward), end=', ')
    return reward


class ConflictEnv(gym.Env, ABC):
    def __init__(self, limit=30, **kwargs):
        self.limit = limit
        self.train, self.test = load_and_split_data('scenarios_gail_final', **kwargs)

        self.action_space = spaces.Discrete(CmdCount)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(350, ), dtype=np.float64)

        print('----------env----------')
        print('    train size: {:>6}'.format(len(self.train)))
        print(' validate size: {:>6}'.format(len(self.test)))
        print('  action shape: {}'.format((self.action_space.n,)))
        print('   state shape: {}'.format(self.observation_space.shape))
        print('-----------------------')

        self.scene = None

    def shuffle_data(self):
        np.random.shuffle(self.train)

    def reset(self):
        info = self.train.pop(0)
        self.scene = ConflictScene(info, limit=self.limit)
        self.train.append(info)
        return self.scene.get_states()

    def step(self, action, scene=None):
        if scene is None:
            scene = self.scene

        solved, cmd_info = scene.do_step(action)
        rewards = calc_reward(solved, cmd_info)
        states = scene.get_states()
        return states, rewards, True, {'result': solved}

    def evaluate(self, act, save_path='policy', **kwargs):
        num_array = []
        obs_array = []
        act_array = []
        rew_array = []
        n_obs_array = []

        # 选择训练集场景还是训练集场景
        tmp = self.train if 'evaluate' in save_path else self.test

        # 对每一个场景进行测试
        for info in tqdm(tmp, desc='Test from '+save_path):
            obs_collected = {'num': [], 'obs': [], 'act': [], 'rew': [], 'n_obs': []}

            scene = ConflictScene(info, limit=self.limit)
            obs, done, result = scene.get_states(), False, {'result': True}
            while not done:
                if 'gail' in save_path:
                    action, _ = act(kwargs['stochastic'], obs)
                    action = np.argmax(action)
                    # print('\tgail', action)
                elif 'dqn' in save_path:
                    action = act(np.array(obs)[None])[0]
                    # print('\tdqn', action)
                else:
                    action = np.random.randint(0, CmdCount)
                    # print('\trandom', action)
                next_obs, rew, done, result = self.step(action, scene=scene)

                obs_collected['num'].append(info.id)
                obs_collected['obs'].append(obs)
                obs_collected['act'].append(action)
                obs_collected['rew'].append(rew)
                obs_collected['n_obs'].append(next_obs)
                obs = next_obs

            if result['result']:
                num_array += obs_collected['num']
                obs_array += obs_collected['obs']
                act_array += obs_collected['act']
                rew_array += obs_collected['rew']
                n_obs_array += obs_collected['n_obs']

        num_array = np.array(num_array)
        obs_array = np.array(obs_array, dtype=np.float64)
        act_array = np.array(act_array, dtype=np.float64)
        rew_array = np.array(rew_array, dtype=np.float64)
        n_obs_array = np.array(n_obs_array, dtype=np.float64)

        # 将测试过程中生成的数据写入npz文件中
        print(save_path)
        np.savez(save_path+'.npz', num=num_array, obs=obs_array, acs=act_array, rews=rew_array, n_obs=n_obs_array)

    def close(self):
        pass
