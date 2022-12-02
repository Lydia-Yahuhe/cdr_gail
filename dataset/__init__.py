"""
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
"""
import numpy as np


class Dset(object):
    def __init__(self, names, inputs, labels, randomize):
        self.names = list(names)
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.pointer = None
        self.__init_pointer()

    def __init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx]
            self.names = [self.names[i] for i in idx]

    def get_next_batch(self, batch_samples):
        if isinstance(batch_samples, int):
            batch_size = batch_samples
            if batch_size < 0:
                return self.inputs, self.labels

            if self.pointer + batch_size >= self.num_pairs:
                self.__init_pointer()
            end = self.pointer + batch_size
            inputs = self.inputs[self.pointer:end, :]
            labels = self.labels[self.pointer:end]
            self.pointer = end
            return inputs, labels

        idx = []
        for name in batch_samples:
            idx.append(-1 if name not in self.names else self.names.index(name))
        return self.inputs[idx, :], self.labels[idx]


class OurSet(object):
    def __init__(self, expert_path, randomize=True, track=False):
        traj_data = np.load(expert_path)
        self.nums = traj_data['num']
        if not track:
            self.obs = traj_data['obs']
        else:
            self.obs = np.hstack([traj_data['obs'], traj_data['n_obs']])
        self.acs = traj_data['acs'].astype(int)
        assert len(self.obs) == len(self.acs)

        self.dset = Dset(self.nums, self.obs, self.acs, randomize)

        print('----------dataset----------')
        print(' nums:', self.nums.shape, self.nums.dtype)
        print('  obs:', self.obs.shape, self.obs.dtype)
        print('  acs:', self.acs.shape, self.acs.dtype)
        print('---------------------------')

    def get_next_batch(self, batch_samples):
        return self.dset.get_next_batch(batch_samples=batch_samples)

    def get_action(self, num):
        obs_n, act_n = self.dset.get_next_batch(batch_samples=[num])
        return obs_n[0], int(obs_n[0])


def split_expert_samples(load_path, ratio):
    """
    将专家策略分为训练集和测试集.
    """
    traj_data = np.load(load_path)

    expert1, expert2 = {}, {}
    for key, value in traj_data.items():
        length = int(value.shape[0] * ratio)
        expert1[key] = value[:length]
        expert2[key] = value[length:]

    np.savez('random_policy_5000_train.npz', **expert1)
    np.savez('random_policy_5000_test.npz', **expert2)


if __name__ == '__main__':
    from parameters import data_path, load_size

    split_expert_samples(load_path=data_path + '/random_policy_{}.npz'.format(load_size),
                         ratio=0.8)
