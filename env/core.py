from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

from rtree.index import Index, Property

from .utils import make_bbox_3d, distance, calc_level, calc_turn_prediction
from .model import Point2D, FlightPlan, Routing, Waypoint, Performance, AircraftType, Conflict, Segment

CmdCount = 21
alt_idx_list = [-600.0, 0.0, 600.0]
hdg_idx_list = [-60.0, -45.0, -30.0, 0.0, 30.0, 45.0, 60.0]


# ---------
# Command
# ---------
@dataclass
class ATCCmd:
    delta: float = 0.0
    assignTime: int = 0.0
    ok: bool = True
    cmdType: str = ""

    def to_dict(self):
        return {self.cmdType: '{},{}'.format(round(self.delta, 2), self.assignTime)}

    def __str__(self):
        return '%s: <TIME:%d, DELTA:%0.2f>' % (self.cmdType, self.assignTime, self.delta)

    def __repr__(self):
        return '%s: <TIME:%d, DELTA:%0.2f>' % (self.cmdType, self.assignTime, self.delta)


def parse_cmd(now: int, cmd_idx: int):
    idx: int = cmd_idx // CmdCount
    cmd_idx_ = cmd_idx % CmdCount
    # alt cmd
    alt_delta: float = alt_idx_list[cmd_idx_ // 7]
    alt_cmd = ATCCmd(delta=alt_delta, assignTime=now, cmdType='Altitude')
    # hdg cmd
    hdg_delta: float = hdg_idx_list[cmd_idx_ % 3]
    hdg_cmd = ATCCmd(delta=hdg_delta, assignTime=now, cmdType='Heading')
    print('{:>2d} {:>2d} {:>+6.1f} {:>+6.1f}'.format(cmd_idx, idx, alt_delta, hdg_delta), end=' | ')
    return [idx, alt_cmd, hdg_cmd]


# ---------
# Control
# ---------
@dataclass
class FlightControl(object):
    def __init__(self, fpl: FlightPlan = None, other: FlightControl = None):
        if fpl is not None:
            self.altCmd: ATCCmd = None
            self.spdCmd: ATCCmd = None
            self.hdgCmd: ATCCmd = None
            self.targetAlt: float = fpl.max_alt
            self.targetHSpd: float = 0.0
            self.targetCourse: float = 0.0
        else:
            self.altCmd: ATCCmd = other.altCmd
            self.spdCmd: ATCCmd = other.spdCmd
            self.hdgCmd: ATCCmd = other.hdgCmd
            self.targetAlt: float = other.targetAlt
            self.targetHSpd: float = other.targetHSpd
            self.targetCourse: float = other.targetCourse

    def assign(self, cmd_list: List[ATCCmd]):
        for cmd in cmd_list:
            if cmd.cmdType == "Altitude":
                self.altCmd = cmd
            elif cmd.cmdType == "Speed":
                self.spdCmd = cmd
            elif cmd.cmdType == "Heading":
                self.hdgCmd = cmd
            else:
                raise NotImplementedError

    def update(self, now: int, v_spd: float, performance: Performance, alt: float, hdg: float,
               hdg_to_target: float):
        self.__update_target_spd(now, v_spd, performance)
        self.__update_target_alt(now, v_spd, alt)
        self.__update_target_hdg(now, hdg, hdg_to_target)

    def __update_target_alt(self, now: int, v_spd: float, alt: float):
        alt_cmd = self.altCmd
        if alt_cmd is None or now != alt_cmd.assignTime:
            return

        delta: float = alt_cmd.delta
        target_alt: float = calc_level(alt, v_spd, delta)
        if v_spd * delta >= 0.0 and 12000.0 >= target_alt >= 6000.0:
            self.targetAlt = target_alt
        self.altCmd = None

    def __update_target_spd(self, now: int, v_spd: float, performance: Performance):
        spd_cmd = self.spdCmd
        if spd_cmd is None or now != spd_cmd.assignTime:
            if v_spd == 0.0:
                self.targetHSpd = performance.normCruiseTAS
            elif v_spd > 0.0:
                self.targetHSpd = performance.normClimbTAS
            else:
                self.targetHSpd = performance.normDescentTAS
            return
        self.spdCmd = None

    def __update_target_hdg(self, now: int, heading: float, hdg_to_target: float):
        hdg_cmd = self.hdgCmd
        if hdg_cmd is None:
            self.targetCourse = hdg_to_target
            return

        diff: int = now - hdg_cmd.assignTime
        if diff < 0:
            self.targetCourse = hdg_to_target
            return

        delta: float = hdg_cmd.delta
        if delta == 0 or diff == 240:  # 结束偏置（dogleg机动）
            self.targetCourse = hdg_to_target
            self.hdgCmd = None
        elif diff == 0:  # 以delta角度出航
            self.targetCourse = (delta + heading) % 360
        elif diff == 120:  # 转向后持续60秒飞行，之后以30°角切回航路
            self.targetCourse = (-abs(delta) / delta * (abs(delta) + 30) + heading) % 360


# ---------
# Profile
# ---------
@dataclass
class FlightProfile:
    def __init__(self, route: Routing = None, other: FlightProfile = None):
        if route is not None:
            self.segs: List[Segment] = route.segments
            self.idx: int = 0
            self.cur_seg: Segment = self.segs[0]
            self.__update_dist_course()
        else:
            self.segs: List[Segment] = other.segs
            self.idx: int = other.idx
            self.cur_seg: Segment = other.cur_seg
            self.distToTarget: float = other.distToTarget
            self.courseToTarget: float = other.courseToTarget

    def __update_dist_course(self, location=None):
        if location is None:
            self.distToTarget = self.cur_seg.distance
            self.courseToTarget = self.cur_seg.course
        else:
            target: Point2D = self.cur_seg.end.location
            self.distToTarget = location.distance_to(target)
            self.courseToTarget = location.bearing_to(target)

    def update(self, h_spd: float, heading: float, turn_rate: float, location: Point2D) -> bool:
        passed, seg = self.__target_passed(h_spd, heading, turn_rate)
        if passed:
            self.idx += 1
            self.cur_seg = seg
            if self.cur_seg is None:  # 如果cur seg是None，则飞行计划结束
                return False
            self.__update_dist_course()
        else:
            self.__update_dist_course(location)
        return True

    def __next_n_seg(self, n: int):
        idx: int = self.idx + n  # 若总共9个航段，则idx ∈ [0,8]
        return self.segs[idx] if idx <= len(self.segs) - 1 else None

    def __target_passed(self, h_spd: float, heading: float, turn_rate: float) -> Tuple[bool, Segment]:
        dist: float = self.distToTarget
        next_seg: Segment = self.__next_n_seg(1)
        if dist <= h_spd * 1:
            passed = True
        elif next_seg is not None:
            passed = dist <= calc_turn_prediction(h_spd, self.cur_seg.course, next_seg.course, turn_rate)
        else:
            passed = 270 > (heading - self.courseToTarget + 360) % 360 >= 90
        return passed, next_seg


# ---------
# Status
# ---------
class FlightStatus:
    def __init__(self, fpl: FlightPlan = None, other: FlightStatus = None):
        if fpl is not None:
            self.alt: float = fpl.min_alt
            self.performance: Performance = Performance()
            self.acType: AircraftType = fpl.aircraft.aircraftType
            self.acType.compute_performance(self.alt, self.performance)
            self.hSpd: float = self.performance.normCruiseTAS
            self.vSpd: float = 0
            wpt_list: List[Waypoint] = fpl.routing.wpt_list
            self.location: Point2D = wpt_list[0].location.copy()
            self.heading: float = wpt_list[0].bearing_to(wpt_list[1])
        else:
            self.alt: float = other.alt
            self.performance: Performance = Performance()
            self.performance.set(other.performance)
            self.acType: AircraftType = other.acType
            self.hSpd: float = other.hSpd
            self.vSpd: float = other.vSpd
            self.location: Point2D = other.location.copy()
            self.heading: float = other.heading

    def update(self, target_h_spd: float, target_course: float, target_alt: float):
        self.__move_horizontal(target_h_spd, target_course)
        self.__move_vertical(target_alt)
        self.__update_performance()

    def __move_horizontal(self, target_h_spd: float, target_hdg: float):
        # h_spd
        pre_h_spd: float = self.hSpd
        if pre_h_spd > target_h_spd:
            self.hSpd = max(pre_h_spd - self.acType.normDeceleration * 1, target_h_spd)
        elif pre_h_spd < target_h_spd:
            self.hSpd = min(pre_h_spd + self.acType.normAcceleration * 1, target_h_spd)
        # heading
        pre_hdg = self.heading
        diff: float = (target_hdg - pre_hdg) % 360
        if diff != 0.0:
            diff = diff - 360 if diff > 180 else diff
            if abs(diff) >= 90:
                turn: float = self.performance.maxTurnRate * 1
            else:
                turn: float = self.performance.normTurnRate * 1
            diff = min(max(diff, -turn), turn)
            self.heading = (pre_hdg + diff) % 360
        # location
        self.location.move_to(self.heading, (pre_h_spd + self.hSpd) * 1 / 2)

    def __move_vertical(self, target_alt: float):
        diff: float = target_alt - self.alt
        if diff < 0:
            v_spd: float = max(-self.performance.normDescentRate * 1, diff)
        elif diff > 0:
            v_spd: float = min(self.performance.normClimbRate * 1, diff)
        else:
            v_spd: float = 0.0
        self.alt += v_spd
        self.vSpd = v_spd

    def __update_performance(self):
        self.acType.compute_performance(self.alt, self.performance)

    def get_state(self) -> Tuple[float, float, float, float, float, float]:
        return self.location.tuple() + (self.alt, self.hSpd, self.vSpd, self.heading)

    def get_position(self) -> Tuple[float, float, float]:
        return self.location.tuple() + (self.alt,)


# ---------
# Agent
# ---------
class AircraftAgent:
    def __init__(self, fpl=None, other=None):
        if fpl is not None:
            self.fpl: FlightPlan = fpl
            self.id: str = fpl.id
            self.phase: str = 'Schedule'
            self.control = FlightControl(fpl)
            self.status = FlightStatus(fpl)
            self.profile = FlightProfile(fpl.routing)
        else:
            self.fpl: FlightPlan = other.fpl
            self.id: str = other.id
            self.phase: str = other.phase
            self.control = FlightControl(other=other.control)
            self.status = FlightStatus(other=other.status)
            self.profile = FlightProfile(other=other.profile)

    def is_enroute(self):
        return self.phase == 'EnRoute'

    def is_finished(self):
        return self.phase == "Finished"

    def state(self):
        return self.status.get_state()

    def position(self):
        return self.status.get_position()

    def step(self, now: int):
        if self.is_finished():  # 如果该航班飞行已经结束，则直接return
            return False

        if now == self.fpl.startTime:  # 如果当前时刻为起飞时刻，则状态改为EnRoute
            self.phase = 'EnRoute'

        if self.is_enroute():
            status, profile, control = self.status, self.profile, self.control
            control.update(now,
                           status.vSpd,
                           status.performance,
                           status.alt,
                           status.heading,
                           profile.courseToTarget)
            status.update(control.targetHSpd,
                          control.targetCourse,
                          control.targetAlt)
            if not profile.update(status.hSpd,
                                  status.heading,
                                  status.performance.normTurnRate,
                                  status.location):
                self.phase = 'Finished'
        return self.is_enroute()

    def assign_cmd(self, cmd_list: List[ATCCmd]):
        self.control.assign(cmd_list)


# ---------
# AgentSet
# ---------
class AircraftAgentSet:
    def __init__(self, fpl_list=None, candi=None, other=None, supply=None):
        if fpl_list is not None:
            self.agents = {}
            for fpl in fpl_list:
                if supply is not None and fpl.id not in supply[1]:
                    continue
                self.agents[fpl.id] = AircraftAgent(fpl)
            self.tracks = supply[0] if supply is not None else {}
            self.agent_id_en: List[str] = []
            self.candi: dict = candi
            self.time: int = min(candi.keys()) - 1
            self.end: int = max(candi.keys())
            self.points = []
        else:
            self.agents: Dict[str, AircraftAgent] = {a_id: AircraftAgent(other=agent)
                                                     for a_id, agent in other.agents.items()}
            self.tracks = other.tracks
            self.agent_id_en: List[str] = other.agent_id_en[:]
            self.candi: dict = other.candi
            self.time: int = other.time
            self.end: int = other.end
            self.points = other.points[:]

    def is_done(self) -> bool:
        return self.time > self.end and len(self.agent_id_en) <= 1

    def __pre_do_step(self, clock: int, read: bool = False) -> List[str]:
        if read:
            return list(self.agents.keys())

        if clock in self.candi.keys():
            return self.agent_id_en + self.candi[clock]
        return self.agent_id_en

    def step(self, duration: int, basic=False):
        now = self.time
        duration -= now * int(basic)
        read_csv = self.tracks is not None

        points = []
        for i in range(duration):
            clock = now + i + 1
            agent_id_en = []
            points = []
            for agent_id in self.__pre_do_step(clock, read=read_csv):
                agent = self.agents[agent_id]
                if agent.step(clock):
                    agent_id_en.append(agent_id)
                    points.append([agent_id, ] + list(agent.state()))
            self.agent_id_en = agent_id_en

        now += duration
        if read_csv and now in self.tracks.keys():
            points += self.tracks[now]

        self.time = now
        self.points = points
        return points

    def __build_rtree(self):
        points = self.points
        agents = self.agents

        idx = Index(properties=Property(dimension=3))
        for i, point in enumerate(points):
            idx.insert(i, make_bbox_3d(point[1:4], (0.0, 0.0, 0.0)))
        return idx, agents, points

    def detect(self, search=None):
        r_tree, agents, points = self.__build_rtree()
        if len(points) <= 1:
            return []

        if search is None:
            search = self.agent_id_en

        conflicts = []
        check_list = []
        for a0_id in search:
            a0 = agents[a0_id]
            if not a0.is_enroute():
                continue

            pos0 = a0.position()
            for i in r_tree.intersection(make_bbox_3d(pos0, (0.1, 0.1, 299))):
                a1_id, *pos1 = points[i]
                if a0_id == a1_id:
                    continue

                c_id = a0_id + '-' + a1_id
                if c_id in check_list:
                    continue

                h_dist = distance(pos0[:2], pos1[:2])
                v_dist = abs(pos0[2] - pos1[2])
                if h_dist < 10000 and v_dist < 300.0:
                    conflicts.append(
                        Conflict(
                            id=c_id,
                            time=self.time,
                            hDist=h_dist,
                            vDist=v_dist,
                            pos0=pos0,
                            pos1=pos1
                        )
                    )
                check_list.append(a1_id + '-' + a0_id)
        return conflicts
