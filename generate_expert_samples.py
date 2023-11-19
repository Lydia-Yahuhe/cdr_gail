import os
import csv

import numpy as np

from env.core import AircraftAgentSet
from env.environment import ConflictScenario
from env.load import load_and_split_data
from env.render import convert_into_image

bins = 42
interval = 4


def get_action_list(num, bins_):
    lst = (np.random.randn(num) / 4 + 1) * bins_ / 2
    lst_ceil = []
    for v in lst:
        if v >= 0.0:
            min_int = int(v)
            max_int = min_int + 1
            if max_int - v >= v - min_int:
                lst_ceil.append(min_int)
            else:
                lst_ceil.append(min(max_int, bins_ - 1))
        else:
            lst_ceil.append(0)
    return lst_ceil


def main(limit=-1, other_traj=False):
    train_set, _ = load_and_split_data(col='scenarios_gail_small', ratio=1.0)

    count = 0
    info_array, action_array, track_array = [], [], []
    for info in train_set:
        print('{}/{}'.format(count, limit), end=' | ')

        filename = './data/trajectory/{}.csv'.format(info['id'])
        if not os.path.exists(filename) and other_traj:
            print('write', end=' | ')

            agent_set = AircraftAgentSet(
                fpl_list=info['fpl_list'],
                candi=info['candi'],
            )
            agent_set.step(duration=info['start'] + 1, basic=True)

            # 检查动作的解脱效果，并返回下一部状态
            tracks = {}
            while True:
                now = agent_set.time
                tracks[now] = agent_set.points
                agent_set.step(interval)
                if now + interval >= info['start'] + 610:
                    tracks[now + interval] = agent_set.points
                    break

            with open(filename, 'w', newline='') as f:
                f = csv.writer(f)
                for clock, points in tracks.items():
                    for p in points:
                        f.writerow([clock, ] + p)
        print()

        action_list = get_action_list(20, bins_=bins)
        for i, action in enumerate(action_list):
            if action >= bins:
                print('Action error!')
                continue

            print('\t>>> {:>2d}/{:>2d}'.format(i + 1, len(action_list)), end='  |  ')
            scenario = ConflictScenario(info, read_csv=True)
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
                break
            print('Failed!')

        if 0 < limit <= count:
            break

    with open('./data/names_{}.csv'.format(count), 'w', newline='') as f:
        csv.writer(f).writerow(info_array)
    np.savez(
        './data/exp_samples_{}'.format(count),
        name=np.array(info_array),
        action=np.array(action_array),
        track=np.array(track_array)
    )


if __name__ == '__main__':
    main(4000, other_traj=True)
