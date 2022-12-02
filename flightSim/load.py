import os.path

import pymongo
from tqdm import tqdm

from flightSim.model import Waypoint, Routing, Aircraft, aircraftTypes, FlightPlan, DataSet, Point2D, \
    ConflictScenarioInfo


def load_waypoint(db):
    wpts = {}
    for e in db['Waypoint'].find():
        wpts[e['id']] = Waypoint(id=e['id'], location=Point2D(e['point']['lng'], e['point']['lat']))

    cursor = db["Airport"].find()
    for pt in cursor:
        wpt = Waypoint(id=pt['id'], location=Point2D(pt['location']['lng'], pt['location']['lat']))
        wpts[wpt.id] = wpt

    return wpts


def load_routing(db, wpts):
    ret = {}
    cursor = db['Routing'].find()
    # for e in cursor:
    #     wptList = [wpts[e["departureAirport"]]]
    #     for wptId in e['waypointList']:
    #         wptList.append(wpts[wptId])
    #     wptList.append(wpts[e["arrivalAirport"]])
    #     r = Routing(e['id'], wptList)
    #     ret[r.id] = r

    for e in cursor:
        wptList = []
        for wptId in e['waypointList']:
            wptList.append(wpts[wptId])
        r = Routing(id=e['id'], waypointList=wptList)
        ret[r.id] = r

    return ret


def load_aircraft(db):
    ret = {}
    cursor = db['Aircraft'].find()
    for e in cursor:
        info = Aircraft(id=e['id'], aircraftType=aircraftTypes[e['aircraftType']])
        ret[info.id] = info

    return ret


def load_flight_plan(db, aircraft, routes):
    ret = {}
    cursor = db['FlightPlan'].find()
    for e in cursor:
        a = aircraft[e['aircraft']]

        fpl = FlightPlan(
            id=e['id'],
            min_alt=0,
            routing=routes[e['routing']],
            startTime=e['startTime'],
            aircraft=a,
            max_alt=e['flightLevel']
        )

        ret[fpl.id] = fpl

    return ret


def load_data_set():
    connection = pymongo.MongoClient('localhost')
    database = connection['admin']

    wpts = load_waypoint(database)
    aircrafts = load_aircraft(database)
    routes = load_routing(database, wpts)
    fpls = load_flight_plan(database, aircrafts, routes)

    connection.close()
    return DataSet(wpts, routes, fpls, aircrafts)


routings = load_data_set().routings


def load_and_split_data(col='scenarios_gail_final', size=None, ratio=0.8):
    file_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(file_path, 'no_solution_list.csv')
    with open(file_path, 'r', newline='') as f:
        no_solution_list = f.readline().strip('\r\n').split(',')

    if size is None:
        size = 1e6

    db = pymongo.MongoClient('localhost')['admin']
    data = load_data(db, col, size, no_solution_list)
    split_size = int(size * ratio)
    return data[:split_size], data[split_size:size]


def load_data(db, col, size, limit):
    scenes = []

    for i, e in enumerate(tqdm(list(db[col].find()), desc='Loading from ' + col)):
        if str(i+1) in limit:
            continue

        fpl_list, starts = [], []
        for f in e['fpl_list']:
            # aircraft
            ac = Aircraft(id=f['aircraft'], aircraftType=aircraftTypes[f['acType']])

            # routing
            r_other = f['other']
            routing = routings[f['routing']]
            wpt_list = routing.waypointList[r_other[0]:r_other[1]]
            routing = Routing(id=routing.id, waypointList=wpt_list, other=r_other)

            # fpl
            starts.append(f['startTime'])
            fpl_list.append(FlightPlan(id=f['id'], aircraft=ac, routing=routing,
                                       startTime=f['startTime'],
                                       min_alt=f['min_alt'], max_alt=f['max_alt']))
        scenes.append(ConflictScenarioInfo(id=str(i+1),
                                           time=e['time'], conflict_ac=e['id'].split('-'),
                                           other=[e['pos0'], e['pos1'], e['hDist'], e['vDist']],
                                           start=min(starts) - 1, end=max(starts), fpl_list=fpl_list))
        if len(scenes) >= size:
            break
    return scenes
