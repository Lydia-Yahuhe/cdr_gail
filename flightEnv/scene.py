import numpy as np

from flightEnv.agent_Set import AircraftAgentSet
from flightEnv.cmd import int_2_atc_cmd, check_cmd


def read_from_csv(file_name, limit):
    if file_name is None:
        return [{}, None]

    file_name = '/Volumes/Documents/Trajectories/No.{}.csv'.format(file_name)
    with open(file_name, 'r', newline='') as f:
        ret = {}
        for line in f.readlines():
            [fpl_id, time_, *line] = line.strip('\r\n').split(',')
            if fpl_id in limit:
                continue

            time_ = int(time_)
            if time_ in ret.keys():
                ret[time_].append([fpl_id] + [float(x) for x in line])
            else:
                ret[time_] = [[fpl_id] + [float(x) for x in line]]

    return [ret, limit]


class ConflictScene:
    def __init__(self, info, limit=0, read=True):
        self.info = info
        self.conflict_ac, self.clock = info.conflict_ac, info.time
        self.conflict_pos = info.other[0]

        if read:
            self.agentSet = AircraftAgentSet(fpl_list=info.fpl_list, start=info.start,
                                             supply=read_from_csv(info.id, self.conflict_ac))
        else:
            self.agentSet = AircraftAgentSet(fpl_list=info.fpl_list, start=info.start)
        self.agentSet.do_step(self.clock - 300 + limit, basic=True)

        self.cmd_check_dict = {ac: {'HDG': [], 'ALT': [], 'SPD': []} for ac in self.conflict_ac}
        self.cmd_info = {}

    def now(self):
        return self.agentSet.time

    def get_conflict_ac(self, idx):
        ac_id = self.conflict_ac[idx]
        return self.agentSet.agents[ac_id]

    # def get_state(self, ac_en, limit=50):
    #     states = [[0.0 for _ in range(7)] for _ in range(limit)]
    #
    #     j = 0
    #     for [agent, *state] in ac_en:
    #         ele = [int(agent in self.conflict_ac),
    #                state[0] - self.conflict_pos[0],
    #                state[1] - self.conflict_pos[1],
    #                (state[2] - self.conflict_pos[2]) / 3000,
    #                (state[3] - 150) / 100,
    #                state[4] / 20,
    #                state[5] / 180]
    #         states[min(limit - 1, j)] = ele
    #         j += 1
    #     return states
    #
    # def get_states(self):
    #     state_1 = self.get_state(self.agentSet.agent_en_)
    #
    #     ghost = AircraftAgentSet(other=self.agentSet)
    #     ghost.do_step(duration=60)
    #     state_2 = self.get_state(ghost.agent_en_)
    #
    #     ghost.do_step(duration=60)
    #     state_3 = self.get_state(ghost.agent_en_)
    #
    #     return np.concatenate(np.vstack([state_1, state_2, state_3]))

    def get_states(self, limit=50):
        states = [[0.0 for _ in range(7)] for _ in range(limit)]

        j = 0
        for [agent, *state] in self.agentSet.agent_en_:
            ele = [int(agent in self.conflict_ac),
                   state[0] - self.conflict_pos[0],
                   state[1] - self.conflict_pos[1],
                   (state[2] - self.conflict_pos[2]) / 3000,
                   (state[3] - 150) / 100,
                   state[4] / 20,
                   state[5] / 180]
            states[min(limit - 1, j)] = ele
            j += 1
        return np.concatenate(states)

    def do_step(self, action):
        agent_id, idx = self.conflict_ac[0], action

        # 指令解析
        now = self.now()
        agent = self.agentSet.agents[agent_id]
        [hold, *cmd_list] = int_2_atc_cmd(now + 1, idx, agent)
        # print('{:>4d}, {:>4d}'.format(idx, hold), end=', ')

        # 执行hold，并探测冲突
        self.agentSet.do_step(duration=hold)

        # 分配动作
        for cmd in cmd_list:
            cmd.ok, reason = check_cmd(cmd, agent, self.cmd_check_dict[agent_id])
            # print(now, hold, cmd.assignTime, self.now())
            # print('{:>+5d}, {}'.format(int(cmd.delta), int(cmd.ok)), end=', ')
            agent.assign_cmd(cmd)
        cmd_info = {'agent': agent_id, 'cmd': cmd_list, 'hold': hold}
        self.cmd_info[now] = cmd_info

        # 执行动作并探测冲突
        has_conflict = self.__do_step(self.clock + 300, duration=30)
        return not has_conflict, cmd_info  # solved, done, cmd

    def __do_step(self, end_time, duration):
        hold = end_time - self.now()
        for dur in [30 for _ in range(hold // duration)] + [hold % duration]:
            self.agentSet.do_step(duration=dur)
            conflicts = self.agentSet.detect_conflict_list(search=self.conflict_ac)
            if len(conflicts) > 0:
                return True
        return False
