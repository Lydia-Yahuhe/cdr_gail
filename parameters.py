common_params = {
    'lr': 5e-4,
    'batch_size': 32,
    'total_timesteps': 100000,
    'buffer_size': 10000,
    'learning_starts': 1000,
    'exploration_fraction': 0.2
}
size = 2000
ratio = 0.7
limit_path = './data/names_{}.csv'.format(size)
expert_path = './data/exp_samples_{}.npz'.format(size)
bc_actions_path = './trained/bc_actions/'
dqn_no_exp_path = './trained/dqn_no_exp/'
gail_actions_path = './trained/gail_actions/'
gail_tracks_path = './trained/gail_tracks/'
random_path = './trained/random/'
fx = 0.5
fy = 0.5
span = 4
bins = 42
