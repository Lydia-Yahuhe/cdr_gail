import numpy as np

from flightEnv import ConflictScene


def execute_action(expert_path='policy', size=5000):
    print(expert_path)
    traj_data = np.load(expert_path + '.npz')

    nums = list(traj_data['num'])
    acs = list(traj_data['acs'].astype(int))

    env = ConflictEnv(limit=0, size=size, ratio=1.0)
    info_dict = {info.id: info for info in env.train}

    num_array, act_array, tra_array = [], [], []
    count = 0
    for idx, num in enumerate(nums):
        action = acs[idx]
        info = info_dict[num]
        print('>>>', idx, info.id, num, action)

        scene = ConflictScene(info)
        c_time = scene.info.time

        _, _, _, result = env.step(action, scene=scene)

        ac = scene.get_conflict_ac(0)

        trajs = []
        for clock, track in ac.tracks.items():
            if c_time + 300 >= clock > c_time - 300:
                trajs.append(track[:3])

        if sum([len(v) for v in trajs]) == 1800:
            num_array.append(info.id)
            act_array.append(action)
            tra_array.append(trajs)

        if result['result']:
            count += 1

    print(count / len(acs) * 100)
    num_array = np.array(num_array)
    act_array = np.array(act_array)
    tra_array = np.array(tra_array)

    print(num_array.shape, act_array.shape, tra_array.shape)
    np.savez(expert_path + '_traj.npz', num=num_array, act=act_array, traj=tra_array)


def generate_traj(folder, size):
    print(folder)
    for file_or_dir in os.listdir(folder):
        file_or_dir = os.path.join(folder, file_or_dir)
        if os.path.isfile(file_or_dir) and file_or_dir.endswith('.npz'):
            execute_action(file_or_dir.split('.')[0], size=size)
