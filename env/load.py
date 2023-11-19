import pymongo
from tqdm import tqdm
import numpy as np

from env.model import *


def dict_to_type(e):
    """
    加载性能数据表格（各个高度对应的标称速度、转弯率等）
    """
    ret = {}
    for key, values in e.items():
        if key != 'flightPerformanceTable':
            ret[key] = values
        else:
            ret[key] = [Performance(**v) for v in values]
    return AircraftType(**ret)


def load_type(database):
    """
    加载机型数据（最大加减速度和性能数据表等）
    """
    ret = {}
    for e in database['AircraftType'].find():
        del e['_id']
        key = e['id']
        ret[key] = dict_to_type(e)
    return ret


def load_waypoint(database):
    """
    加载航路点数据
    """
    ret = {}
    for e in database['Waypoint'].find():
        key = e['id']
        ret[key] = Waypoint(id=key,
                            location=Point2D(e['point']['lng'], e['point']['lat']))
    for e in database["Airport"].find():
        key = e['id']
        ret[key] = Waypoint(id=key,
                            location=Point2D(e['location']['lng'], e['location']['lat']))
    return ret


def load_routing(database, wpt_dict):
    """
    加载航线数据（城市OD对和计划航路点集合）
    """
    ret = {}
    for e in database['Routing'].find():
        key = e['id']
        ret[key] = Routing(id=key,
                           wpt_list=[wpt_dict[wpt_id] for wpt_id in e['waypointList']])
    return ret


def load_aircraft(database, act_dict):
    """
    加载航空器数据（注册号和机型）
    """
    ret = {}
    for e in database['AircraftRandom'].find():
        key = e['id']
        ret[key] = Aircraft(id=key,
                            aircraftType=act_dict[e['aircraftType']])
    return ret


def load_flight_plan(database, aircraft, routes):
    """
    加载航班的飞行计划数据（呼号、起始高度、航线、起飞时刻、航空器和目标高度）
    """
    ret = {}
    for e in database['FlightPlan'].find():
        key = e['id']
        ret[key] = FlightPlan(id=key,
                              min_alt=0,
                              routing=routes[e['routing']],
                              startTime=e['startTime'],
                              aircraft=aircraft[e['aircraft']],
                              max_alt=e['flightLevel'])
    return ret


def load_data_set(host='localhost', db='admin'):
    """
    包含所有飞行数据的集合
    """
    connection = pymongo.MongoClient(host)
    database = connection[db]
    wpt_dict = load_waypoint(database)
    act_dict = load_type(database)
    air_dict = load_aircraft(database, act_dict)
    rou_dict = load_routing(database, wpt_dict)
    fpl_dict = load_flight_plan(database, air_dict, rou_dict)
    connection.close()
    return DataSet(
        wpt_dict=wpt_dict,
        rou_dict=rou_dict,
        fpl_dict=fpl_dict,
        air_dict=air_dict,
        act_dict=act_dict
    )


def load_and_split_data(host='localhost', db='admin', col='scenarios_gail_small', size=None, ratio=0.8, limit_path=None):
    """
    加载冲突场景，并将其按比例分为训练集和测试集
    """
    connection = pymongo.MongoClient(host)
    database = connection[db]
    wpt_dict = load_waypoint(database)
    act_dict = load_type(database)

    if limit_path is not None:
        with open(limit_path, 'r', newline='') as f:
            limit = f.readline().strip('\r\n').split(',')
    else:
        limit = []

    if size is None:
        size = int(1e6)

    scenarios, idx_list = [], []
    for info in tqdm(database[col].find(), desc='Loading from {}/{}'.format(db, col)):
        info_id = info['id']
        info_id = info_id.split('_')[-1] if isinstance(info_id, str) else info_id
        fpl_list, candi = [], {}
        for i, fpl in enumerate(info['fpl_list']):
            # call sign
            fpl_id = fpl['id']
            # start time
            start = fpl['startTime']
            # routing
            routing = Routing(
                id=fpl['route'],
                wpt_list=[wpt_dict[p_id] for p_id in fpl['wpt_list']]
            )
            # aircraft
            ac = Aircraft(
                id=fpl['register'],
                aircraftType=act_dict[fpl['type']]
            )
            # flight plan
            fpl_list.append(
                FlightPlan(
                    id=fpl_id,
                    routing=routing,
                    aircraft=ac,
                    startTime=start,
                    min_alt=fpl['min_alt'],
                    max_alt=fpl['max_alt']
                )
            )
            if start in candi.keys():
                candi[start].append(fpl_id)
            else:
                candi[start] = [fpl_id]

        # print(info_id, info_id in limit, limit)
        if len(limit) == 0 or info_id in limit:
            idx_list.append(info_id)
            conflict = info['conflicts'][0]
            scenarios.append(
                dict(
                    id=info_id,
                    fpl_list=fpl_list,
                    candi=candi,
                    conflict_ac=conflict['id'].split('-'),
                    start=info['start'],
                    conflict=conflict
                )
            )
        if len(scenarios) >= size:
            break
    connection.close()

    split_size = int(len(scenarios) * ratio)
    if len(limit) == 0:
        np.random.shuffle(scenarios)
        return scenarios[:split_size], scenarios[split_size:]

    return (
        [scenarios[idx_list.index(x)] for x in limit[:split_size]],
        [scenarios[idx_list.index(x)] for x in limit[split_size:]]
    )


if __name__ == '__main__':
    load_data_set()
