import cv2
import numpy as np

from parameters import expert_path


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
            self.inputs = self.inputs[idx]
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
            inputs = self.inputs[self.pointer:end]
            labels = self.labels[self.pointer:end]
            self.pointer = end
            return inputs, labels

        idx = []
        for name in batch_samples:
            idx.append(-1 if name not in self.names else self.names.index(name))
        return self.inputs[idx], self.labels[idx]


class OurSet(object):
    def __init__(self, expert_path, seq=False, randomize=True):
        traj_data = np.load(expert_path)

        self.names = traj_data['name']
        self.actions = traj_data['action']
        if seq:
            track = traj_data['track']
            # self.tracks = np.concatenate([track[:, 0], track[:, -1]], axis=-1)
            self.tracks = track[:, -1] - track[:, 0]
            print(self.tracks.shape)
        else:
            self.tracks = traj_data['track'][:, 0]
        self.dset = Dset(self.names, self.tracks, self.actions, randomize)
        print('----------dataset----------')
        print('  name:', self.names.shape, self.names.dtype)
        print('action:', self.actions.shape, self.actions.dtype)
        print(' track:', self.tracks.shape, self.tracks.dtype)
        print('---------------------------')
        traj_data.close()

    def get_next_batch(self, batch_samples):
        return self.dset.get_next_batch(batch_samples=batch_samples)

    def get_action(self, num):
        obs_n, act_n = self.dset.get_next_batch(batch_samples=[num])
        return obs_n[0], act_n[0]


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

    # np.savez('random_policy_5000_train.npz', **expert1)
    # np.savez('random_policy_5000_test.npz', **expert2)


def main():
    for key, value in np.load(expert_path).items():
        print(key, value.shape)
        if key != 'track':
            continue

        for i, frame in enumerate(value):
            # frame = np.hstack(frame)
            frame = frame[-1] - frame[0]
            win_name = 'frame {}/{}'.format(i+1, len(value))
            cv2.imshow(win_name, frame)
            if cv2.waitKey(0) == 113:  # 按q键退出渲染
                cv2.destroyWindow(win_name)
                continue


if __name__ == '__main__':
    main()
