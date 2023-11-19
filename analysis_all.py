import math
import os

import matplotlib.pyplot as plt
import numpy as np

from parameters import *


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
    p = np.array(stat(p)[1])
    q = np.array(stat(q)[1])
    return np.sum(np.where(p != 0, p * np.log(p / (q+1e-4)), 0))


def get_acr_mse(int_num_list, int_act, exp_num_list, exp_act):
    acr_list = []
    mse_list = []
    for num, act in zip(int_num_list, int_act):
        e_act = exp_act[exp_num_list.index(num)]
        delta = act - e_act
        acr_list.append(int(delta == 0))
        mse_list.append(delta * delta)
    return np.mean(acr_list), math.sqrt(np.mean(mse_list))


def draw_ad(expert_train_act, expert_test_act, labels, ad_train_lst, ad_test_lst):
    fig, axes = plt.subplots(1, 2)

    # plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

    ax0 = axes[0]
    ax1 = axes[1]

    train_tmp, test_tmp = [], []
    new_labels = []
    for label in labels:
        train_tmp.append(expert_train_act)
        test_tmp.append(expert_test_act)
        new_labels.append('\n'.join(label.split(',')))

    ax0.violinplot(train_tmp, showmedians=True)
    ax0.violinplot(ad_train_lst[-1], showmedians=True)
    ax0.set_xticks(list(range(1, len(labels)+1)), labels=new_labels)

    ax1.violinplot(test_tmp, showmedians=True)
    ax1.violinplot(ad_test_lst[-1], showmedians=True)
    ax1.set_xticks(list(range(1, len(labels)+1)), labels=new_labels)

    # [ax.legend(fontsize=12) for ax in axes]
    [ax.yaxis.set_visible(False) for ax in axes]
    fig.text(0.5, 0.02, 'Algorithm and Knowledge Type', ha='center', fontsize=16)
    fig.text(0.07, 0.5, 'Action Distribution',  va='center', rotation='vertical', fontsize=16)
    plt.show()
    # fig.savefig('./trained/Action Distribution.png', dpi=300)
    fig.savefig('./trained/Action Distribution.pdf')


def draw_others(steps, labels, y_label, train_lst, test_lst):
    fig, axes = plt.subplots(1, 2)
    ax0 = axes[0]
    ax1 = axes[1]
    train_array = np.array(train_lst)
    test_array = np.array(test_lst)
    steps = [int(step/5000) for step in steps]
    for i, label in enumerate(labels):
        ax0.plot(steps, train_array[:, i], label=label)
        ax1.plot(steps, test_array[:, i], label=label)
    [ax.legend() for ax in axes]
    # [ax.xaxis.set_visible(False) for ax in axes]
    # [ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x') for ax in axes]
    fig.text(0.5, 0.02, 'The number of iteration/5e3', ha='center', fontsize=16)
    fig.text(0.05, 0.5, y_label,  va='center', rotation='vertical', fontsize=16)
    plt.show()
    # fig.savefig('./trained/'+y_label+'.png', dpi=300)
    fig.savefig('./trained/'+y_label+'.pdf')


def _total(total):
    steps, sr, sim = [], [], []
    for step, result_array in enumerate(total):
        length = result_array.shape[0]
        exp_act = result_array[0, :]
        act_same = None
        solved = None
        steps.append(step)
        for i in range(1, length, 2):
            act = np.asarray(result_array[i, :] - exp_act == 0, dtype=np.int8)
            if act_same is None:
                act_same = act
            else:
                act_same += act
            result = result_array[i+1, :]
            if solved is None:
                solved = result
            else:
                solved += result

        sr.append(np.mean(np.where(solved > 0.0, 1, 0)))
        sim.append(np.mean(np.where(act_same > 0.0, 1, 0)))
    return steps, sr, sim


def draw_total(total_train, total_test):
    print(total_train.shape, total_test.shape)
    steps, train_sr, train_sim = _total(total_train)
    _, test_sr, test_sim = _total(total_test)

    fig, axes = plt.subplots(1, 2)
    ax0 = axes[0]
    ax1 = axes[1]
    ax0.plot(steps, train_sr, label='train_sr')
    ax0.plot(steps, train_sim, label='train_sim')
    ax1.plot(steps, test_sr, label='test_sr')
    ax1.plot(steps, test_sim, label='test_sim')
    [ax.legend() for ax in axes]
    fig.text(0.5, 0.02, 'The number of iteration/5e3', ha='center', fontsize=16)
    fig.text(0.05, 0.5, 'Percentage',  va='center', rotation='vertical', fontsize=16)
    plt.show()
    # fig.savefig('./trained/Total.png', dpi=300)
    fig.savefig('./trained/Total.pdf')


def _analysis(paths, prefixes, labels, start=0, end=10001, delta=1000):
    traj_data = np.load(expert_path)
    print(list(traj_data.keys()))
    expert_num = list(traj_data['name'])
    expert_act = traj_data['action']
    steps = list(range(start, end, delta))

    train_idx, test_idx = None, None
    expert_train_act, expert_test_act = None, None

    acr_train_lst, acr_test_lst = [], []
    mse_train_lst, mse_test_lst = [], []
    kl_train_lst, kl_test_lst = [], []
    sr_train_lst, sr_test_lst = [], []
    ad_train_lst, ad_test_lst = [], []
    total_train_lst, total_test_lst = [], []
    for step in steps:
        tmp_train_kl, tmp_test_kl = [], []
        tmp_train_sr, tmp_test_sr = [], []
        tmp_train_mse, tmp_test_mse = [], []
        tmp_train_acr, tmp_test_acr = [], []
        tmp_train_ad, tmp_test_ad = [], []
        tmp_train_total, tmp_test_total = [], []
        for path, prefix in zip(paths, prefixes):
            file_name = path + prefix + '_train_{}.npz'.format(step)
            print(file_name)
            if not os.path.exists(file_name):
                continue
            train_data = np.load(path + prefix + '_train_{}.npz'.format(step))
            test_data = np.load(path + prefix + '_test_{}.npz'.format(step))
            train_num, test_num = train_data['name'], test_data['name']
            train_act, test_act = train_data['action'], test_data['action']

            if train_idx is None:
                train_idx = [expert_num.index(num) for num in train_num]
                test_idx = [expert_num.index(num) for num in test_num]
                expert_train_act = expert_act[train_idx]
                expert_test_act = expert_act[test_idx]

            # Integrated Model
            if len(tmp_train_total) <= 0:
                tmp_train_total.append(expert_train_act)
                tmp_test_total.append(expert_test_act)
            tmp_train_total += [train_act, train_data['result']]
            tmp_test_total += [test_act, test_data['result']]

            # AD
            kl_train = kl_divergence(expert_train_act, train_act)
            tmp_train_kl.append(kl_train)
            kl_test = kl_divergence(expert_test_act, test_act)
            tmp_test_kl.append(kl_test)
            tmp_train_ad.append(train_act)
            tmp_test_ad.append(test_act)
            # SR
            tmp_train_sr.append(np.mean(train_data['result']))
            tmp_test_sr.append(np.mean(test_data['result']))
            # ACR, MSE
            acr, mse = get_acr_mse(train_num, train_act, expert_num, expert_act)
            tmp_train_acr.append(acr)
            tmp_train_mse.append(mse)
            print(step, acr, mse, end=' ')
            acr, mse = get_acr_mse(test_num, test_act, expert_num, expert_act)
            tmp_test_acr.append(acr)
            tmp_test_mse.append(mse)
            print(acr, mse)
        kl_train_lst.append(tmp_train_kl)
        kl_test_lst.append(tmp_test_kl)
        sr_train_lst.append(tmp_train_sr)
        sr_test_lst.append(tmp_test_sr)
        mse_train_lst.append(tmp_train_mse)
        mse_test_lst.append(tmp_test_mse)
        acr_train_lst.append(tmp_train_acr)
        acr_test_lst.append(tmp_test_acr)
        ad_train_lst.append(tmp_train_ad)
        ad_test_lst.append(tmp_test_ad)
        total_train_lst.append(tmp_train_total)
        total_test_lst.append(tmp_test_total)

    draw_total(np.array(total_train_lst), np.array(total_test_lst))
    draw_ad(expert_train_act, expert_test_act, labels, ad_train_lst, ad_test_lst)
    draw_others(steps, labels, 'KL Divergence', kl_train_lst, kl_test_lst)
    draw_others(steps, labels, 'Success Rate', sr_train_lst, sr_test_lst)
    draw_others(steps, labels, 'Action Overlap Rate', acr_train_lst, acr_test_lst)
    draw_others(steps, labels, 'RMSE', mse_train_lst, mse_test_lst)


if __name__ == '__main__':
    kwargs = {
        "start": 50000,
        "end": 100001,
        "delta": 5000
    }

    # _analysis(
    #     paths=[gail_actions_path, gail_tracks_path, bc_actions_path, dqn_no_exp_path, random_path],
    #     prefixes=['dqn_actions', 'dqn_tracks', 'bc_actions', 'dqn_no_exp', 'random'],
    #     labels=['GAIL+DQN,State-Action', 'GAIL+DQN,State-State', 'BC,State-Action', 'DQN,NoExp', 'Random,NoExp'],
    #     **kwargs
    # )
    _analysis(
        paths=[gail_actions_path, gail_tracks_path],
        prefixes=['dqn_actions', 'dqn_tracks'],
        labels=['GAIL+DQN,State-Action', 'GAIL+DQN,State-State'],
        **kwargs
    )

