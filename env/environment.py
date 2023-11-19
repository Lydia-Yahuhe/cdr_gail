import numpy as np
import gym
from gym import spaces

from .core import CmdCount, AircraftAgentSet, parse_cmd
from .load import load_and_split_data
from .render import convert_into_image


def read_from_csv(file_name, limit):
    if file_name is None:
        return [{}, None]

    with open('./data/trajectory/{}.csv'.format(file_name), 'r', newline='') as f:
        tracks = {}
        for line in f.readlines():
            [time_, fpl_id, *line] = line.strip('\r\n').split(',')
            if fpl_id in limit:
                continue

            time_ = int(time_)
            if time_ in tracks.keys():
                tracks[time_].append([fpl_id, ] + [float(x) for x in line])
            else:
                tracks[time_] = [[fpl_id, ] + [float(x) for x in line]]
    return [tracks, limit]


class ConflictScenario:
    def __init__(self, info, read_csv=False):
        self.info = info

        self.id = info['id']
        self.conflict_ac = info['conflict_ac']
        self.start = info['start']
        # print('{:>5s}, {:>5d}'.format(self.id, self.start), end=' | ')

        self.shin: AircraftAgentSet = AircraftAgentSet(
            fpl_list=info['fpl_list'],
            candi=info['candi'],
            supply=None if not read_csv else read_from_csv(self.id, self.conflict_ac)
        )
        self.shin.step(duration=self.start + 1, basic=True)

    def now(self) -> int:
        return self.shin.time

    def step(self, action=None, interval=4):
        end_time = self.start + 601

        # 解析、分配动作（模拟）
        cmd_list = []
        if action is not None:
            [idx, *cmd_list] = parse_cmd(self.now() + 30, action)
            ac = self.conflict_ac[idx]
            self.shin.agents[ac].assign_cmd(cmd_list)

        # 检查动作的解脱效果，并返回下一部状态
        conflicts, tracks = [], {}
        while True:
            now = self.shin.time
            tracks[now] = self.shin.points
            self.shin.step(interval)
            conflicts = self.shin.detect(search=self.conflict_ac)
            if (action is not None and len(conflicts) > 0) or now + interval >= end_time:
                tracks[now+interval] = self.shin.points
                break

        # 根据指令和解脱效果，计算奖励
        return (
            self.get_states(),
            cmd_list,
            {'result': len(conflicts) <= 0, 'track': tracks}
        )

    def get_states(self, radius=5, fx=0.5, fy=0.5, wait=None):
        track = self.shin.points
        return convert_into_image(
            tracks={0: track},
            limit=self.conflict_ac,
            radius=radius,
            fx=fx,
            fy=fy,
            wait=wait,
        )


class ConflictEnv(gym.Env):
    def __init__(self, size=None, ratio=0.8, limit_path=None, simple=False):
        self.train_set, self.test_set = load_and_split_data(size=size, ratio=ratio, limit_path=limit_path)

        self.action_space = spaces.Discrete(CmdCount * 2)
        self.observation_space = spaces.Box(low=-1.0, high=+1.0, shape=(300, 400, 3))
        print('----------env------------')
        print('|   split ratio: {:<6.2f} |'.format(ratio))
        print('|    train size: {:<6} |'.format(len(self.train_set)))
        print('| validate size: {:<6} |'.format(len(self.test_set)))
        print('|  action shape: {}   |'.format((self.action_space.n,)))
        print('|   state shape: {} |'.format(self.observation_space.shape))
        print('-------------------------')

        self.scenario = None
        self.simple = simple

    def reset(self):
        info = self.train_set.pop(0)
        self.scenario = ConflictScenario(info, read_csv=True)
        self.train_set.append(info)
        return self.scenario.get_states()

    def step(self, action):
        next_states, cmd_list, info = self.scenario.step(action)
        is_solved = info['result']
        reward = self.__calc_reward(is_solved, cmd_list)
        print('{} {:>+4.2f}'.format(int(is_solved), reward), end=' | ')
        return next_states, reward, True, info

    def render(self, mode='human', wait=1):
        pass

    def __calc_reward(self, solved: bool, cmd_list):
        if solved:
            if self.simple:
                rew = 0.0
            else:
                [alt_cmd, hdg_cmd, *spd_cmd] = cmd_list
                rew_alt = 0.3 - abs(alt_cmd.delta) / 3000.0
                rew_hdg = 0.4 - abs(hdg_cmd.delta) / 150.0
                # rew_spd = 0.3 - abs(spd_cmd.delta) / 100.0
                # rew = rew_alt + rew_spd + rew_hdg
                rew = rew_alt + rew_hdg
        else:
            rew = -1.0
        return rew

    def evaluate(self, act, save_path=None, use_set='test'):
        if use_set == 'all':
            test_set = self.train_set + self.test_set
        elif use_set == 'test':
            test_set = self.test_set
        else:
            test_set = self.train_set

        info_array, action_array, result_array = [], [], []
        for i, info in enumerate(test_set):
            print('{}/{}'.format(i+1, len(test_set)), end=' | ')
            scenario = ConflictScenario(info, read_csv=True)
            obs = scenario.get_states()
            if 'bc' in save_path:
                action = np.argmax(act(obs)[0][0])
            elif 'random' in save_path:
                action = np.random.randint(0, self.action_space.n)
            else:
                action = act(np.array(obs)[None])[0]
            _, _, result = scenario.step(action)
            info_array.append(int(info['id']))
            action_array.append(action)
            result_array.append(int(result['result']))
            print(result['result'])

        print('sr: ', np.mean(result_array))
        np.savez(
            save_path,
            name=np.array(info_array),
            action=np.array(action_array),
            result=np.array(result_array)
        )

    def close(self):
        pass
