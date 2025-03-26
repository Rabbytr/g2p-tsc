import numpy as np
from . import BaseGenerator
from world import world_cityflow, world_sumo  # , world_openengine
from agent.utils import group_by_first


class RoadMapping(object):
    def __init__(self, world):
        self.road_mapping = {}
        for inter in world.roadnet['intersections']:
            for road_link in inter['roadLinks']:
                attr = set([lane_link.get('startLaneIndex') for lane_link in road_link.get('laneLinks')])
                a, b = road_link.get('startRoad'), road_link.get('endRoad')
                self.road_mapping[f'{a}-{b}'] = attr

    def get_valid_lane(self, road1, road2):
        lane_inds = self.road_mapping[f'{road1}-{road2}']
        return [f'{road1}_{lane_ind}' for lane_ind in lane_inds]


class AllStateGenerator(BaseGenerator):

    def __init__(self, world, I, fns, in_only=False, average=None, negative=False):
        self.world = world
        self.I = I

        # get lanes of intersections
        self.lanes = []
        if in_only:
            roads = I.in_roads
        else:
            roads = I.roads

        # ---------------------------------------------------------------------------------------------------------------
        # TODO: register it in Registry
        if isinstance(world, world_sumo.World):
            for r in roads:
                if not self.world.RIGHT:
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
                else:
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
                self.lanes.append(tmp)
                # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        elif isinstance(world, world_cityflow.World):
            for road in roads:
                from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (
                        road["endIntersection"] == I.id)
                self.lanes.append(
                    [road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
        else:
            raise Exception('NOT IMPLEMENTED YET')

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        size = sum(len(x) for x in self.lanes)
        if average == "road":
            size = len(roads)
        elif average == "all":
            size = 1
        self.ob_length = len(fns) * size
        if self.ob_length == 3:
            self.ob_length = 4

        self.average = average
        self.negative = negative

        self.grouped_lanes = group_by_first(self.I.lanelinks)

        if not hasattr(self.__class__, 'road_mapping'):
            self.__class__.road_mapping = RoadMapping(self.world)

    def _advanc_and_routeaware(self, lane_id, vehicles):
        eff_run_num = 0
        to_lane_count = {k: 0 for k in self.grouped_lanes[lane_id]}
        for vehicle in vehicles:
            vec_info = self.world.eng.get_vehicle_info(vehicle)
            distance = float(vec_info['distance'])
            speed = float(vec_info['speed'])
            max_speed = self.world.all_lanes_speed[lane_id]
            max_length = self.world.lane_length[lane_id]

            # ================= effective running number ===================
            if distance + max_speed * 10 < max_length:
                continue
            if speed >= 0.1:
                eff_run_num += 1

            # ================= routeaware ===================
            route = vec_info['route'].split()
            if len(route) < 2:
                continue
            if len(route) == 2:
                next_lanes = self.grouped_lanes[lane_id]
            else:
                next_lanes = self.road_mapping.get_valid_lane(route[1], route[2])
            for lane in next_lanes:
                to_lane_count[lane] += 1.0 / len(next_lanes)

            # if self.world.eng.get_current_time() > 1000:
            #     pass

        info = self.world.get_info("lane_waiting_count")
        explicit = 0.0
        for to_lane, expected_num in to_lane_count.items():
            explicit += expected_num - info[to_lane]

        return eff_run_num, explicit

    def generate(self):
        # ============== Advance States ===================
        lane_vehicles = self.world.get_info('lane_vehicles')

        advance_ret = []
        routeaware_ret = []
        for road_lanes in self.lanes:
            for lane_id in road_lanes:
                run_effc_num, explicit = self._advanc_and_routeaware(lane_id, lane_vehicles[lane_id])
                advance_ret.append(run_effc_num)
                routeaware_ret.append(explicit)

        # ============== Efficient States ===================
        result = self.world.get_info("lane_waiting_count")
        lvw_ret = []
        for road_lanes in self.lanes:
            for lane_id in road_lanes:
                # Efficient State
                lvw_ret.append(result[lane_id] - np.mean([result[to] for to in self.grouped_lanes[lane_id]]))

        return np.array(lvw_ret + advance_ret + routeaware_ret)
