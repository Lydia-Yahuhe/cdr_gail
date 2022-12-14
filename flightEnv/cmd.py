from flightSim.aircraft import atccmd
from flightSim.utils import convert_with_align

CmdCount = 16*9
KT2MPS = 0.514444444444444
NM2M = 1852
flight_level = [i*300.0 for i in range(29)]
flight_level += [i*300.0 + 200.0 for i in range(29, 50)]


def calc_level(alt, v_spd, delta):
    delta = int(delta / 300.0)
    lvl = int(alt / 300.0) * 300.0

    if alt < 8700.0:
        idx = flight_level.index(lvl)
        if (v_spd > 0 and alt - lvl != 0) or (v_spd == 0 and alt - lvl > 150):
            idx += 1

        return flight_level[idx+delta]

    lvl += 200.0
    idx = flight_level.index(lvl)
    if v_spd > 0 and alt - lvl > 0:
        idx += 1
    elif v_spd < 0 and alt - lvl < 0:
        idx -= 1

    return flight_level[idx+delta]


def check_cmd(cmd, a, check_dict):
    if not a.is_enroute() or not a.next_leg:
        return False, '1'

    if cmd.cmdType == atccmd.ATCCmdType.Heading:
        return True, 'Hdg'

    if cmd.cmdType == atccmd.ATCCmdType.Speed:
        return True, 'Spd'

    if cmd.cmdType == atccmd.ATCCmdType.Altitude:
        check = check_dict['ALT']

        # 最高12000m，最低6000m
        target_alt = cmd.targetAlt
        if target_alt > 12000 or target_alt < 6000:
            return False, '2'

        # 下降的航空器不能上升，或上升的航空器不能下降
        v_spd, delta = a.status.vSpd, cmd.delta
        if v_spd * delta < 0:
            return False, '3'

        # 调过上升，又调下降，或调过下降，又调上升
        if delta == 0.0:
            prefix = int(abs(v_spd) / v_spd) if v_spd != 0.0 else 0
        else:
            prefix = int(abs(delta) / delta)

        if prefix == 0:
            return True, '0'

        if len(check) > 0 and prefix not in check:
            return False, '4'
        check.append(prefix)
        return True, '0'

    raise NotImplementedError


def int_2_atc_cmd(time: int, idx: int, target):
    # 将idx转化成三进制数
    time_idx, cmd_idx = idx // 9, idx % 9
    # print(idx, alt_idx, spd_idx, hdg_idx, time_idx)

    # time cmd
    time_cmd = 240-int(time_idx) * 15

    [alt_idx, hdg_idx] = convert_with_align(cmd_idx, x=3, align=2)
    # alt cmd
    delta = (int(alt_idx) - 1) * 600.0
    targetAlt = calc_level(target.status.alt, target.status.vSpd, delta)
    alt_cmd = atccmd.AltCmd(delta=delta, targetAlt=targetAlt, assignTime=time+time_cmd)
    # hdg cmd
    delta = (int(hdg_idx) - 1) * 45
    hdg_cmd = atccmd.HdgCmd(delta=delta, assignTime=time+time_cmd)
    return [time_cmd, alt_cmd, hdg_cmd]


def reward_for_cmd(cmd_info):
    reward = 0.0
    for cmd in cmd_info['cmd']:
        if not cmd.ok:
            reward += -0.5
        else:
            reward += int(cmd.delta != 0.0) * (-0.2)
    return reward


def parse_idx_2_action(idx):
    time_idx, cmd_idx = idx // 9, idx % 9
    time_cmd = 240 - int(time_idx) * 15  # time cmd
    [alt_idx, hdg_idx] = convert_with_align(cmd_idx, x=3, align=2)  # 将idx转化成三进制数
    alt_cmd = (int(alt_idx) - 1) * 600.0  # alt cmd
    hdg_cmd = (int(hdg_idx) - 1) * 45  # hdg cmd
    return time_cmd, alt_cmd, hdg_cmd


def analysis_two_actions(a1, a2):
    cmd_list_a1 = parse_idx_2_action(a1)
    cmd_list_a2 = parse_idx_2_action(a2)

    # print(a1, a2, cmd_list_a1, cmd_list_a2)
    time_diff = abs(cmd_list_a1[0] - cmd_list_a2[0]) / 240
    alt_diff = abs(cmd_list_a1[1] - cmd_list_a2[1]) / 600
    hdg_diff = abs(cmd_list_a1[2] - cmd_list_a2[2]) / 45

    return time_diff ** 2 + alt_diff ** 2 + hdg_diff ** 2
