import csv
import os

import numpy as np

from env.environment import ConflictScenario, ConflictEnv
from parameters import *
from baselines.common import models


def load_act_gail_action(env, step=0):
    from baselines.deepq.deepqForNoExp import learn

    print('Run test with gail action')
    act = learn(env,
                network=models.cnn_small(),
                load_path=gail_actions_path,
                **common_params)
    load_path = gail_actions_path + 'my_model_{}.pkl'.format(step)
    if not os.path.exists(load_path):
        return None

    print('Loaded model from {}'.format(load_path))
    act.load(path=load_path)
    return act


def load_act_gail_track(env, step=0):
    from baselines.deepq.deepqForTracks import learn

    print('Run test with gail track')
    act = learn(env,
                network=models.cnn_small(),
                load_path=gail_tracks_path,
                **common_params)
    load_path = gail_tracks_path + 'my_model_{}.pkl'.format(step)
    if not os.path.exists(load_path):
        return None

    print('Loaded model from {}'.format(load_path))
    act.load(path=load_path)
    return act


def load_act_bc_action(env, step=0):
    from baselines.deepq.behavior_clone import BehaviorClone
    from data.data_set import OurSet

    print('Run test with bc action')
    expert_set = OurSet(expert_path=expert_path)
    learner = BehaviorClone(env, expert_set)

    load_path = bc_actions_path + 'my_model_{}.pkl'.format(step)
    if not os.path.exists(load_path):
        return None
    print('Loaded model from {}'.format(load_path))
    act = learner.load(path=load_path)
    return act


def load_act_no_exp(env, step=0):
    from baselines.deepq.deepqForNoExp import learn

    print('Run test with no exp')
    act = learn(env,
                network=models.cnn_small(),
                load_path=dqn_no_exp_path,
                **common_params)
    load_path = dqn_no_exp_path + 'my_model_{}.pkl'.format(step)
    if not os.path.exists(load_path):
        return None
    print('Loaded model from {}'.format(load_path))
    act.load(path=load_path)
    return act


def main(actor='gail_action', step=100000):
    env = ConflictEnv(ratio=1.0)
    if actor == 'gail_action':
        act = load_act_gail_action(env, step=step)
    elif actor == 'gail_track':
        act = load_act_gail_track(env, step=step)
    elif actor == 'bc_action':
        act = load_act_bc_action(env, step=step)
    elif actor == 'no_exp':
        act = load_act_no_exp(env, step=step)
    else:
        act = None

    info_array, action_array, result_array = [], [], []
    for i, info in enumerate(env.train_set):
        print('{}/{}'.format(i+1, len(env.train_set)), end=' ')
        scenario = ConflictScenario(info, read_csv=True)
        state = scenario.get_states()
        if 'bc' in actor:
            action = np.argmax(act(state)[0][0])
        elif 'random' in actor:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = act(np.array(state)[None])[0]
        _, _, result = scenario.step(action)

        info_array.append(int(info['id']))
        action_array.append(action)
        result_array.append(int(result['result']))
        if result['result']:
            print('Accepted!', scenario.now(), scenario.now() - scenario.start)
        else:
            print('Failed!')

    np.savez(
        './data/{}_{}.csv'.format(actor, step),
        name=np.array(info_array),
        action=np.array(action_array),
        result=np.array(result_array)
    )


def analysis(kwargs, steps):
    with open(limit_path, 'r', newline='') as f:
        limit = f.readline().strip('\r\n').split(',')

    tmp_dict_train, tmp_dict_test = {}, {}
    for k, arg in enumerate(kwargs):
        data = np.load('./data/{}_{}.csv.npz'.format(arg, steps[k]))
        name = data['name']
        action = data['action']
        result = data['result']
        for i, f_id in enumerate(name):
            if str(f_id) in limit:
                if f_id not in tmp_dict_train.keys():
                    tmp_dict_train[f_id] = [action[i], result[i]]
                else:
                    tmp_dict_train[f_id] += [action[i], result[i]]
            else:
                if f_id not in tmp_dict_test.keys():
                    tmp_dict_test[f_id] = [action[i], result[i]]
                else:
                    tmp_dict_test[f_id] += [action[i], result[i]]

    train_array, test_array = [], []
    for f_id, value in tmp_dict_train.items():
        train_array.append(value)
    for f_id, value in tmp_dict_test.items():
        test_array.append(value)

    sr_train = np.array(train_array).mean(axis=0)
    sr_test = np.array(test_array).mean(axis=0)
    sr_total = np.array(train_array+test_array).mean(axis=0)
    for i, arg in enumerate(kwargs):
        idx = 2*i+1
        print('train:{:>7.6f},test:{:>7.6f},total:{:>7.6f}'.format(
            sr_train[idx], sr_test[idx], sr_total[idx]
        ), arg)


if __name__ == '__main__':
    kwargs_ = ['bc_action', 'no_exp', 'gail_action', 'gail_track', 'random']
    steps_ = [100000, 100000, 25000, 190000, 100000]
    # main(actor=kwargs_[0], step=steps_[0])
    # main(actor=kwargs_[1], step=steps_[1])
    # main(actor=kwargs_[2], step=steps_[2])
    # main(actor=kwargs_[3], step=steps_[3])
    # main(actor=kwargs_[4], step=steps_[4])

    analysis(kwargs_, steps_)
