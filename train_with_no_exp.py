from baselines.common import models

from env.environment import ConflictEnv
from parameters import *

from baselines.deepq.deepqForNoExp import learn


def train():
    print('Run train with no exp')
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=ratio)
    learn(env,
          network=models.cnn_small(),
          save_path=dqn_no_exp_path,
          **common_params)
    env.close()


def test(start=0, end=100001, delta=5000):
    print('Run test with no exp')
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=ratio)
    act = learn(env,
                network=models.cnn_small(),
                load_path=dqn_no_exp_path,
                **common_params)
    for step in range(start, end, delta):
        print('{}/{}'.format(step, end-1))
        load_path = dqn_no_exp_path + 'my_model_{}.pkl'.format(step)
        print('Loaded model from {}'.format(load_path))
        act.load(path=load_path)
        env.evaluate(act, save_path=dqn_no_exp_path + 'dqn_no_exp_train_{}'.format(step), use_set='train')
        env.evaluate(act, save_path=dqn_no_exp_path + 'dqn_no_exp_test_{}'.format(step), use_set='test')
    env.close()


if __name__ == '__main__':
    # train()
    test(end=200001)
