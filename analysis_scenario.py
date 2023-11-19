import matplotlib.pyplot as plt
import numpy as np

from baselines.common import models

from env.environment import ConflictEnv, ConflictScenario
from parameters import *


def act_gail_action(env, step):
    from baselines.deepq.deepqForActions import learn
    act = learn(env,
                network=models.cnn_small(),
                load_path=gail_actions_path,
                **common_params)
    load_path = gail_actions_path + 'my_model_{}.pkl'.format(step)
    print('Loaded model from {}'.format(load_path))
    return act


def act_bc_action(env, step):
    from data.data_set import OurSet
    from baselines.deepq.behavior_clone import BehaviorClone

    expert_set = OurSet(expert_path=expert_path)
    learner = BehaviorClone(env, expert_set)
    load_path = bc_actions_path + 'my_model_{}.pkl'.format(step)
    print('Loaded model from {}'.format(load_path))
    act = learner.load(path=load_path)
    return act


def act_gail_tracks(env, step):
    from baselines.deepq.deepqForTracks import learn

    act = learn(env,
                network=models.cnn_small(),
                load_path=gail_tracks_path,
                **common_params)
    load_path = gail_tracks_path + 'my_model_{}.pkl'.format(step)
    print('Loaded model from {}'.format(load_path))
    act.load(path=load_path)
    return act


def act_dqn(env, step):
    from baselines.deepq.deepqForNoExp import learn

    act = learn(env,
                network=models.cnn_small(),
                load_path=dqn_no_exp_path,
                **common_params)
    load_path = dqn_no_exp_path + 'my_model_{}.pkl'.format(step)
    print('Loaded model from {}'.format(load_path))
    act.load(path=load_path)
    return act


def draw_traj(ax, tracks, actions):
    for i, (key, track) in tracks.items():
        for ac, pos in track.items():
            pos_array = np.array(pos)
            ax.plot(pos_array[:, 0], pos_array[:, 1:], label=key+'_'+ac)


def main(env, step):
    act_dict = {
        'origin': None,
        'bc_action': act_bc_action(env, step),
        'act_gail_a': act_gail_action(env, step),
        'act_gail_t': act_gail_tracks(env, step),
        'act_no_exp': act_dqn(env, step)
    }

    fig, axes = plt.subplots(2, 3)

    count = 0
    for i, info in enumerate(env.test_set):
        print('{}'.format(i + 1), end=' | ')
        tracks, actions = {}, []
        ok = True
        for j, (key, act) in enumerate(act_dict.items()):
            scenario = ConflictScenario(info, read_csv=True)

            obs = scenario.get_states()
            if 'bc' in key:
                action = np.argmax(act(obs)[0][0])
            elif key == 'origin':
                action = None
            else:
                action = act(np.array(obs)[None])[0]
            _, _, result = scenario.step(action)

            print('\t>>>', j, action, result['result'])
            if not result['result']:
                ok = False
                break

            track = {}
            for clock, [ac, *status] in result['track']:
                if ac not in scenario.conflict_ac:
                    continue

                if ac in track.keys():
                    track[ac].append([clock, ] + status[:3])
                else:
                    track[ac] = [[clock, ] + status[:3], ]
            tracks[key] = track
            actions.append(action)

        if ok:
            count += 1
            draw_traj(axes[count // 3, count % 2], tracks, actions)
        if count >= 6:
            break

    [ax.legend() for ax in axes]
    fig.text(0.5, 0.02, 'The number of iteration/5e3', ha='center', fontsize=16)
    fig.text(0.05, 0.5, 'Percentage', va='center', rotation='vertical', fontsize=16)
    plt.show()
    # fig.savefig('./trained/Total.png', dpi=300)
    fig.savefig('./trained/Total.pdf')


if __name__ == '__main__':
    main(
        env=ConflictEnv(limit_path=limit_path, size=size, ratio=ratio),
        step=50000
    )
