import os
import csv

import numpy as np

from env.environment import ConflictScenario, ConflictEnv
from env.render import convert_into_image

from parameters import *

from baselines.deepq.deepqForNoExp import learn
from baselines.common import models


def load_act(env, step=0):
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


def main():
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=1.0)
    act = load_act(env, step=45000)

    count = 0
    info_array, action_array, track_array = [], [], []
    for info in env.train_set:
        scenario = ConflictScenario(info, read_csv=True)
        state = scenario.get_states()
        action = act(np.array(state)[None])[0]
        _, _, result = scenario.step(action)

        if result['result']:
            print('Accepted!', scenario.now(), scenario.now() - scenario.start)
            info_array.append(int(info['id']))
            action_array.append(action)
            track_array.append(
                convert_into_image(
                    tracks=result['track'],
                    limit=scenario.conflict_ac,
                    fx=0.5, fy=0.5, span=75,
                )
            )
            count += 1
        else:
            print('Failed!')

    with open('./data/names_{}.csv'.format(count), 'w', newline='') as f:
        csv.writer(f).writerow(info_array)
    np.savez(
        './data/exp_samples_{}'.format(count),
        name=np.array(info_array),
        action=np.array(action_array),
        track=np.array(track_array)
    )


if __name__ == '__main__':
    main()
