import time

import numpy as np
import pymongo
import matplotlib.pyplot as plt

from env.core import AircraftAgentSet
from env.load import load_data_set, load_and_split_data
from env.model import Routing, FlightPlan, Aircraft
from env.render import plot_line, make_random_color, make_color, draw
from env.utils import pnpoly


# --------------------------------------------
# -- Generate Continuous Conflict Scenarios --
# --------------------------------------------

"""
 1. 找到所有经过武汉扇区（vertices）的航路 → wh_routing_list；
 2. 截取wh_routing_list航路中在武汉扇区（vertices）里的航段；
 3. 随机抽取120个routing，构建飞行计划和AgentSet；
 4. 运行AgentSet，并进行冲突探测；
 5. 剔除冲突时间-起飞时间<=600的飞行计划，并重建AgentSet；
 6. 运行AgentSet，并进行冲突探测；
 7. 记录冲突信息和飞行计划；
 8. 记录各个冲突信息和飞行计划；
"""

shift = 360

vertices = ((114.7, 32.0),
            (114.67, 31.37),
            (115.67, 30.45),
            (115.24, 30.07),
            (112.7, 30.54),
            (113.27, 31.05),
            (113.26, 31.64))

# vertices = ((109.51666666666667, 31.9),
#             (110.86666666666666, 33.53333333333333),
#             (114.07, 32.125),
#             (115.81333333333333, 32.90833333333333),
#             (115.93333333333334, 30.083333333333332),
#             (114.56666666666666, 29.033333333333335),
#             (113.12, 29.383333333333333),
#             (109.4, 29.516666666666666),
#             (109.51666666666667, 31.9),
#             (109.51666666666667, 31.9))

data_set = load_data_set()
flight_level = {i - 20: i * 300.0 + int(i >= 29) * 200.0 for i in range(20, 41)}  # 6000~12500


# step 1和2
def search_routing_in_wuhan():
    import simplekml

    kml = simplekml.Kml()
    folder = kml.newfolder(name='Polygons')
    plot_line(folder,
              (p + (8100.0,) for p in vertices),
              name='border')

    folder = kml.newfolder(name='Lines')
    inner_routes, route_id = [], []
    for r_id, routing in data_set.rou_dict.items():
        wpt_list = routing.wpt_list

        plot_line(folder,
                  (wpt.location.tuple() + (8100.0,) for wpt in wpt_list),
                  name=r_id,
                  color=make_random_color())

        i, idx = 0, []
        for i, wpt in enumerate(wpt_list):
            if pnpoly(vertices, wpt.location.tuple()):
                idx.append(i)

        if len(idx) > 0:
            min_v, max_v = min(idx), max(idx)
            if min_v == 0:
                mode = 'start'
            elif max_v == i:
                mode = 'end'
            else:
                mode = 'part'

            wpt_list_part = wpt_list[max(0, min_v - 1): min(i + 1, max_v + 2)]
            inner_routes.append([r_id, wpt_list_part, mode])
            route_id.append(r_id)

            plot_line(folder,
                      (wpt.location.tuple() + (8100.0,) for wpt in wpt_list_part),
                      name=r_id + '_part',
                      color=make_color(255, 0, 0))
    kml.save('wuhan.kml')

    count = 0
    for fpl in data_set.fpl_dict.values():
        if fpl.routing.id in route_id:
            count += 1
    print(count)

    return inner_routes


# step 3
def get_fpl_random(routes, interval=30, number=100, max_time=43200):
    fpl_set = list(data_set.fpl_dict.values())
    np.random.shuffle(fpl_set)

    count = 0
    fpl_list, candi = [], {}
    for j in range(0, max_time, interval):
        np.random.shuffle(routes)
        candi[j] = []

        for [r_id, wpt_list, mode] in routes[:number]:
            fpl_template = fpl_set[count % len(fpl_set)]

            # routing
            routing = Routing(id=r_id, wpt_list=wpt_list)
            # aircraft
            aircraft = fpl_template.aircraft
            # min_alt
            if mode == 'start':  # 在扇区内起飞的航班，上升后平飞
                min_alt = 6000.0
            else:
                min_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]
            # max_alt
            if mode == 'end':  # 在扇区内落地的航班，下降
                max_alt = 6000.0
            elif np.random.randint(0, 60) % 2 == 0:  # 1/2的航班起始高度等于目标高度
                max_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]
            else:
                max_alt = min_alt
            # flight plan
            fpl = FlightPlan(id=str(count),
                             routing=routing,
                             aircraft=aircraft,
                             startTime=j,
                             min_alt=min_alt,
                             max_alt=max_alt)
            fpl_list.append(fpl)
            candi[j].append(fpl.id)
            count += 1

    return fpl_list, candi


# step 4
def run_scenario(fpl_list, candi):
    print('\t>>>', len(fpl_list))
    agent_set = AircraftAgentSet(fpl_list=fpl_list, candi=candi)

    all_conflicts, record = [], {'flow': [], 'conflict': []}
    shift_list, check_list = [], []
    start = time.time()
    while True:
        agent_set.step(duration=1)
        conflicts = []
        for c in agent_set.detect():
            c_id = c.id
            [a0_id, a1_id] = c_id.split('-')
            c_id_reverse = a1_id + '-' + a0_id
            if c_id in check_list or c_id_reverse in check_list:
                continue

            check_list.append(c_id)
            check_list.append(c_id_reverse)
            fpl0 = agent_set.agents[a0_id].fpl
            fpl1 = agent_set.agents[a1_id].fpl
            if c.time - fpl0.startTime < shift and a0_id not in shift_list:
                shift_list.append(a0_id)
            if c.time - fpl1.startTime < shift and a1_id not in shift_list:
                shift_list.append(a1_id)
            conflicts.append(c)

        if len(conflicts) > 0:
            all_conflicts += conflicts

        now = agent_set.time
        if now % 1000 == 0:
            print('\t>>>', now, len(agent_set.agent_id_en), len(all_conflicts))
        record['flow'].append([now, len(agent_set.agent_id_en)])
        record['conflict'].append([now, len(conflicts)])

        if agent_set.is_done():
            print('场景运行结束：', now, len(conflicts), time.time() - start)
            break

    return all_conflicts, record, shift_list


# Step 7
def write_in_db(name, conflict_info, fpl_info, col='scenarios_meta_small'):
    database = pymongo.MongoClient('localhost')['admin']
    collection = database[col]

    c_times = [c.time for c in conflict_info]
    conflicts = [c.to_dict() for c in conflict_info]
    fpl_list = [fpl.to_dict() for fpl in fpl_info]
    collection.insert_one(dict(id=name, start=min(c_times) - 301, conflicts=conflicts, fpl_list=fpl_list))


def analysis(record, new_record, candi, new_candi, name):
    fig, axes = plt.subplots(3, 1)
    x, y = [], []
    for [t, flow, *_] in record['flow']:
        x.append(t)
        y.append(flow)
    axes[0].plot(x, y, label='before')
    x, y = [], []
    for [t, flow, *_] in new_record['flow']:
        x.append(t)
        y.append(flow)
    axes[0].plot(x, y, label='after')
    axes[0].set_xlabel('Time Axis')
    axes[0].set_ylabel('Flow')
    axes[0].legend()

    x, y = [], []
    for [t, flow, *_] in record['conflict']:
        x.append(t)
        y.append(flow)
    axes[1].plot(x, y, label='before')
    x, y = [], []
    for [t, flow, *_] in new_record['conflict']:
        x.append(t)
        y.append(flow)
    axes[1].plot(x, y, label='after')
    axes[1].set_xlabel('Time Axis')
    axes[1].set_ylabel('Conflict')
    axes[1].legend()

    x, y = [], []
    for key in sorted(candi.keys()):
        x.append(key)
        y.append(len(candi[key]))
    axes[2].plot(x, y, label='before')
    x, y = [], []
    for key in sorted(new_candi.keys()):
        x.append(key)
        y.append(len(new_candi[key]))
    axes[2].plot(x, y, label='after')
    axes[2].set_xlabel('Time Axis')
    axes[2].set_ylabel('Candi')
    axes[2].legend()

    plt.subplots_adjust(hspace=0.5)
    fig.savefig('stats_{}.pdf'.format(name))
    # plt.show()


def make_random_scenario(num_start=0, num_end=10):
    # np.random.seed(1234)
    inner_routes = search_routing_in_wuhan()
    print('>>> 一共找到{}条经过武汉扇区的Routing（Step 1和2）'.format(len(inner_routes)))

    for i in range(num_start, num_end):
        print('No.{}/{}'.format(i+1, num_end))
        print('>>> 随机加载航空器（Step 3)')
        fpl_list, candi = get_fpl_random(inner_routes[:], interval=10, number=1, max_time=1000)

        print('>>> 开始运行场景，并进行冲突探测（Step 4和5）')
        conflicts, record, shift_list = run_scenario(fpl_list, candi)

        print(len(shift_list))
        new_fpl_list, new_candi = [], {}
        for fpl in fpl_list:
            if fpl.id not in shift_list:
                new_fpl_list.append(fpl)
                start = fpl.startTime
                if start in new_candi.keys():
                    new_candi[start].append(fpl.id)
                else:
                    new_candi[start] = [fpl.id]

        print('>>> 重新运行场景，并进行冲突探测（Step 5和6）')
        new_conflicts, new_record, shift_list = run_scenario(new_fpl_list, new_candi)
        assert len(shift_list) == 0
        # analysis(record, new_record, candi, new_candi, str(i + 1))

        print('>>> 记录冲突信息和飞行计划（Step 7）\n')
        write_in_db(i, new_conflicts, new_fpl_list)


# ------------------------------------------
# -- Generate Episodic Conflict Scenarios --
# ------------------------------------------
def run_scenario_1(fpl_list, candi):
    print('fpl_list:', len(fpl_list))
    agent_set = AircraftAgentSet(fpl_list=fpl_list, candi=candi)

    check_list, shift_list = [], []
    conflicts, structure = [], {}
    while not agent_set.is_done():
        agent_set.step(1)
        for c in agent_set.detect():
            c_id = c.id
            [a0_id, a1_id] = c_id.split('-')
            c_id_reverse = a1_id + '-' + a0_id
            if c_id in check_list or c_id_reverse in check_list:
                continue
            check_list.append(c_id)
            check_list.append(c_id_reverse)

            if a0_id in structure.keys():
                structure[a0_id].append((c.time, a1_id))
            else:
                structure[a0_id] = [(c.time, a1_id)]

            if a1_id in structure.keys():
                structure[a1_id].append((c.time, a0_id))
            else:
                structure[a1_id] = [(c.time, a0_id)]

            fpl0 = agent_set.agents[a0_id].fpl
            fpl1 = agent_set.agents[a1_id].fpl
            if c.time - fpl0.startTime < shift and a0_id not in shift_list:
                shift_list.append(a0_id)
            if c.time - fpl1.startTime < shift and a1_id not in shift_list:
                shift_list.append(a1_id)

            # c.printf()
            conflicts.append(c)

        now = agent_set.time
        if now % 1000 == 0:
            print('\t>>>', now, len(agent_set.agent_id_en))

    print(len(shift_list) > 0, shift_list)

    shift_list = []
    for key, value in structure.items():
        if len(value) <= 1:
            continue

        print(key, value)
        base = None
        for v in value:
            if base is None:
                base = v
                continue

            if abs(v[0] - base[0]) <= 360:
                shift_list.append(v[1])
                print('\t>>>', base, v, shift_list)
            else:
                base = v
    print(list(set(shift_list)))
    return shift_list, conflicts


def split_individual_conflict():
    train_set, _ = load_and_split_data(col='scenarios_meta_small', ratio=1.0)
    count, s_name = 497, 3628
    for info in train_set[497:]:
        fpl_list = info['fpl_list']
        shift_list, _ = run_scenario_1(fpl_list=fpl_list, candi=info['candi'])

        new_fpl_list, new_candi = [], {}
        for fpl in fpl_list:
            fpl_id, start = fpl.id, fpl.startTime
            if fpl_id not in shift_list:
                new_fpl_list.append(fpl)
                if start in new_candi.keys():
                    new_candi[start].append(fpl_id)
                else:
                    new_candi[start] = [fpl_id]

        new_shift_list, new_conflicts = run_scenario_1(fpl_list=new_fpl_list, candi=new_candi)
        assert len(new_shift_list) <= 0

        for c in new_conflicts:
            c.printf()
            write_in_db('{}_{}'.format(count, s_name), [c, ], new_fpl_list, col='scenarios_gail_small')
            s_name += 1
        count += 1


# ---------------------------------------
# -- Check Episodic Conflict Scenarios --
# ---------------------------------------

def check_conflict(col='scenarios_gail_small'):
    train_set, _ = load_and_split_data(col=col, ratio=1.0)

    count = 0
    for info in train_set:
        print(info['id'])
        conflict_ac, start = info['conflict_ac'], info['start']
        print('{}/{}'.format(count + 1, len(train_set)), info['id'], conflict_ac, start)
        print(info['conflict'])

        agent_set = AircraftAgentSet(fpl_list=info['fpl_list'], candi=info['candi'])
        agent_set.step(duration=start, basic=True)
        end_time = agent_set.time + 601

        check_list, conflicts = [], []
        while not agent_set.is_done():
            agent_set.step(1)

            for c in agent_set.detect(search=conflict_ac):
                c_id = c.id
                [a0_id, a1_id] = c_id.split('-')
                c_id_reverse = a1_id + '-' + a0_id
                if c_id in check_list or c_id_reverse in check_list:
                    continue
                check_list.append(c_id)
                check_list.append(c_id_reverse)
                conflicts.append(c)
                c.printf()

            if end_time is not None and agent_set.time > end_time:
                break


if __name__ == '__main__':
    # make_random_scenario(num_start=500, num_end=600)
    split_individual_conflict()
    # check_conflict()
    pass
