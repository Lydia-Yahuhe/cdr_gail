from baselines.common import models
from data.data_set import OurSet

from env.environment import ConflictEnv
from parameters import *

from baselines.deepq.deepqForTracks import learn
from baselines.deepq.adversaryTracks import TransitionClassifier


def train():
    print('Run train with track')
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=ratio, simple=True)
    expert_set = OurSet(expert_path=expert_path, seq=True)
    learn(env,
          network=models.cnn_small(),
          reward_giver=TransitionClassifier(env, expert_set, ent_coeff=1e-3),
          save_path=gail_tracks_path,
          **common_params)
    env.close()


def test(start=0, end=100001, delta=5000):
    print('Run test with action')
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=ratio, simple=True)
    act = learn(env,
                network=models.cnn_small(),
                load_path=gail_tracks_path,
                **common_params)
    for step in range(start, end, delta):
        print('{}/{}'.format(step, end-1))
        load_path = gail_tracks_path + 'my_model_{}.pkl'.format(step)
        print('Loaded model from {}'.format(load_path))
        act.load(path=load_path)
        env.evaluate(act, save_path=gail_tracks_path + 'dqn_tracks_train_{}'.format(step), use_set='train')
        env.evaluate(act, save_path=gail_tracks_path + 'dqn_tracks_test_{}'.format(step), use_set='test')
    env.close()


if __name__ == '__main__':
    train()
    # test()
