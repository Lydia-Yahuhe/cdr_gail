import os

load_size = 5000
split_ratio = 0.8
show_fig = True

# path
root = os.path.dirname(__file__)
data_path = root + 'dataset/'
expert = data_path + 'random_policy_{}.npz'.format(load_size)
expert_train = data_path + 'random_policy_{}_train.npz'.format(load_size)
expert_test = data_path + 'random_policy_{}_test.npz'.format(load_size)
figure_path = data_path + 'figures/'
