import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from data.data_set import OurSet
from env.environment import ConflictEnv
from parameters import *


def draw(env, expert_set):
    train_num_list = [int(info['id']) for info in env.train_set]
    test_num_list = [int(info['id']) for info in env.test_set]
    _, train_action_list = expert_set.get_next_batch(train_num_list)
    _, test_action_list = expert_set.get_next_batch(test_num_list)
    total_action_list = list(train_action_list) + list(test_action_list)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
    axes[0].hist(train_action_list, bins=bins)
    axes[0].xaxis.set_visible(False)
    ax_twin = axes[0].twinx()
    ax_twin.plot(*stat(train_action_list), 'red', label='train')
    ax_twin.ticklabel_format(style='sci', scilimits=(0, 1), axis='y')
    ax_twin.xaxis.set_visible(False)
    ax_twin.legend(fontsize=14)

    axes[1].hist(test_action_list, bins=bins)
    axes[1].xaxis.set_visible(False)
    ax_twin = axes[1].twinx()
    ax_twin.plot(*stat(test_action_list), 'green', label='test')
    ax_twin.ticklabel_format(style='sci', scilimits=(0, 1), axis='y')
    ax_twin.xaxis.set_visible(False)
    ax_twin.legend(fontsize=12)

    axes[2].hist(total_action_list, bins=bins)
    axes[2].xaxis.set_visible(False)
    ax_twin = axes[2].twinx()
    ax_twin.plot(*stat(total_action_list), 'blue', label='total')
    ax_twin.ticklabel_format(style='sci', scilimits=(0, 1), axis='y')
    ax_twin.xaxis.set_visible(False)
    ax_twin.legend(fontsize=12)

    fig.text(0.5, 0.05, 'Action Index', ha='center', fontsize=14)
    fig.text(0.05, 0.5, 'Count', va='center', rotation='vertical', fontsize=14)
    fig.text(0.95, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=14)
    fig.savefig('./trained/Sample Distribution.pdf')
    plt.show()


def stat(lst):
    length = len(lst)

    result = {i: 0 for i in range(bins)}
    for v in lst:
        assert v in result.keys()
        result[v] += 1

    x, y = [], []
    for key, value in result.items():
        x.append(key)
        y.append(value / length)
    return x, y


def kl_divergence(p, q):
    p = np.array(stat(p))
    q = np.array(stat(q))
    return np.sum(np.where(p != 0, p * np.log(p / (q+1e-8)), 0))


def get_acr_mse(int_num_list, int_act, exp_num_list, exp_act):
    acr_list = []
    mse_list = []
    for num, act in zip(int_num_list, int_act):
        e_act = exp_act[exp_num_list.index(num)]
        delta = act - e_act
        acr_list.append(int(delta == 0))
        mse_list.append(delta * delta)
    return np.mean(acr_list), math.sqrt(np.mean(mse_list))


def _analysis(path, prefix, start=0, end=10001, delta=1000):
    traj_data = np.load(expert_path)
    print(list(traj_data.keys()))
    expert_num = list(traj_data['name'])
    expert_act = traj_data['action']

    train_idx, test_idx = None, None
    expert_train_act, expert_test_act = None, None
    train_acr_list, test_acr_list, total_acr_list = [], [], []
    train_mse_list, test_mse_list, total_mse_list = [], [], []
    kl_train_lst, kl_test_lst = [], []
    sr_train_lst, sr_test_lst = [], []
    steps = list(range(start, end, delta))
    ad_lst = []
    for step in steps:
        print(path + prefix + '_train_{}.npz'.format(step))
        train_data = np.load(path + prefix + '_train_{}.npz'.format(step))
        test_data = np.load(path + prefix + '_test_{}.npz'.format(step))
        train_num, test_num = train_data['name'], test_data['name']
        train_act, test_act = train_data['action'], test_data['action']

        if train_idx is None:
            train_idx = [expert_num.index(num) for num in train_num]
            test_idx = [expert_num.index(num) for num in test_num]
            expert_train_act = expert_act[train_idx]
            expert_test_act = expert_act[test_idx]

        if step in [0, 50000, 100000]:
            with open('act_{}.csv'.format(step), 'w', newline='') as f:
                f = csv.writer(f)
                for row in zip(train_num, train_act, expert_train_act, train_data['result']):
                    f.writerow(list(row))
                for row in zip(test_num, test_act, expert_test_act, test_data['result']):
                    f.writerow(list(row))
        # AD
        kl_train = kl_divergence(expert_train_act, train_act)
        kl_train_lst.append(kl_train)
        kl_test = kl_divergence(expert_test_act, test_act)
        kl_test_lst.append(kl_test)
        ad_lst.append([stat(train_act), stat(test_act)])
        # SR
        sr_train_lst.append(np.mean(train_data['result']))
        sr_test_lst.append(np.mean(test_data['result']))
        # ACR, MSE
        acr, mse = get_acr_mse(train_num, train_act, expert_num, expert_act)
        train_acr_list.append(acr)
        train_mse_list.append(mse)
        print(step, acr, mse, end=' ')
        acr, mse = get_acr_mse(test_num, test_act, expert_num, expert_act)
        test_acr_list.append(acr)
        test_mse_list.append(mse)
        print(acr, mse)

    fig, axes = plt.subplots(2, 3)

    ax00 = axes[0, 0]
    ax10 = axes[1, 0]
    ax00.plot(*stat(expert_train_act), label='expert_train')
    ax10.plot(*stat(expert_test_act), label='expert_test')

    for i, [train_ad, test_ad] in enumerate(ad_lst):
        step = steps[i]
        if step not in [1000, 10000, end-1, ]:
            continue
        ax00.plot(*train_ad, label='train_{}'.format(step))
        ax10.plot(*test_ad, label='test_{}'.format(step))

    ax01 = axes[0, 1]
    ax11 = axes[1, 1]
    ax01.plot(steps, kl_train_lst, label='train_kl')
    ax01.plot(steps, kl_test_lst, label='test_kl')
    ax11.plot(steps, sr_train_lst, label='train_sr')
    ax11.plot(steps, sr_test_lst, label='test_sr')

    ax02 = axes[0, 2]
    ax12 = axes[1, 2]
    ax02.plot(steps, train_acr_list, label='train_acr')
    ax02.plot(steps, test_acr_list, label='test_acr')
    ax12.plot(steps, train_mse_list, label='train_mse')
    ax12.plot(steps, test_mse_list, label='test_mse')

    [ax.legend() for ax in [ax00, ax10, ax01, ax11, ax02, ax12]]
    plt.show()
    fig.savefig('./trained/'+prefix+'.pdf')


if __name__ == '__main__':
    kwargs = {
        "start": 50000,
        "end": 100001,
        "delta": 5000
    }

    # draw(
    #     env=ConflictEnv(limit_path=limit_path, size=size, ratio=ratio),
    #     expert_set=OurSet(expert_path=expert_path)
    # )

    # _analysis(path=bc_actions_path, prefix='bc_actions', **kwargs)
    # _analysis(path=dqn_no_exp_path, prefix='dqn_no_exp', **kwargs)
    # _analysis(path=gail_actions_path, prefix='dqn_actions', **kwargs)
    # _analysis(path=gail_tracks_path, prefix='dqn_tracks', **kwargs)
