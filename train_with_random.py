import numpy as np

from env.environment import ConflictEnv
from parameters import *


def test():
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=ratio)
    root = './trained/random/'
    env.evaluate(None, save_path=root + 'random_train', use_set='train')
    env.evaluate(None, save_path=root + 'random_test', use_set='test')
    env.close()


def copy(start=0, end=100001, delta=5000):
    root = './trained/random/'
    train_data_dict = np.load(root+'random_train.npz')
    test_data_dict = np.load(root+'random_test.npz')

    for i in range(start, end, delta):
        np.savez(root+'random_train_{}'.format(i), **train_data_dict)
        np.savez(root+'random_test_{}'.format(i), **test_data_dict)


if __name__ == '__main__':
    # test()
    copy()

