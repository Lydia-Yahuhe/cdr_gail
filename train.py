from baselines.common import models
from dataset import OurSet
from flightEnv import ConflictEnv, CmdCount


def train_with_action(expert_path, path, **kwargs):
    from baselines.deepq.deepqForActions import learn
    from common.adversaryActions import TransitionClassifier

    env = ConflictEnv(limit=0, **kwargs)
    learn(env,
          network=models.mlp(num_hidden=128, num_layers=3),
          lr=5e-4,
          batch_size=32,
          total_timesteps=100000,
          buffer_size=10000,
          learning_starts=100,
          reward_giver=TransitionClassifier(env, hidden_size=128, entcoeff=1e-3),
          expert_dataset=OurSet(expert_path=expert_path),
          save_path=path)
    env.close()


def train_with_track(expert_path, path, **kwargs):
    from baselines.deepq.deepqForTracks import learn
    from common.adversaryTracks import TransitionClassifier

    env = ConflictEnv(limit=0, **kwargs)
    learn(env,
          network=models.mlp(num_hidden=128, num_layers=3),
          lr=5e-4,
          batch_size=32,
          total_timesteps=100000,
          buffer_size=10000,
          learning_starts=100,
          reward_giver=TransitionClassifier(env, hidden_size=128, entcoeff=1e-3),
          expert_dataset=OurSet(expert_path=expert_path, track=True),
          save_path=path)
    env.close()


def train_no_exp(path, **kwargs):
    from baselines.deepq.deepqForNoExp import learn

    env = ConflictEnv(limit=0, **kwargs)
    learn(env,
          network=models.mlp(num_hidden=128, num_layers=3),
          lr=5e-4,
          batch_size=32,
          total_timesteps=100000,
          buffer_size=10000,
          learning_starts=100,
          save_path=path)
    env.close()


if __name__ == '__main__':
    from analysis import Analyzer
    from parameters import *

    analyzer = Analyzer(bins=CmdCount,
                        folder=figure_path,
                        expert_train=expert_train,
                        expert_test=expert_test)
    while True:
        input_from_cmd = input('Which train process (1: action, 2: track, 3: no exp, else: exit): ')
        if input_from_cmd == 1:
            print('Run train with action')
            save_path = data_path + 'withActions/'
            train_with_action(expert_path=expert,
                              path=save_path,
                              size=load_size,
                              ratio=split_ratio)
            analyzer.analysis(save_path, keyword='actions')
        elif input_from_cmd == 2:
            print('Run train with track')
            save_path = data_path + 'withTracks/'
            train_with_track(expert_path=expert,
                             path=save_path,
                             size=load_size,
                             ratio=split_ratio)
            analyzer.analysis(save_path, keyword='tracks')
        elif input_from_cmd == 3:
            print('Run train with no exp')
            save_path = data_path + 'withNoExp/'
            train_no_exp(path=save_path,
                         size=load_size,
                         ratio=split_ratio)
            analyzer.analysis(save_path, keyword='no_exp')
        else:
            print('No run process is chosen!')
            break

