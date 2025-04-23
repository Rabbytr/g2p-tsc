import numpy as np
from . import BaseGenerator
from world import world_cityflow, world_sumo  # , world_openengine
from agent.utils import group_by_first
from common.registry import Registry


class AdvanceStateGenerator(BaseGenerator):
    '''
    Generate state or reward based on statistics of lane vehicles.

    :param world: World object
    :param I: Intersection object
    :param fns: list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay", "lane_pressure" and "pressure". 
        "lane_count": get number of running vehicles on each lane. 
        "lane_waiting_count": get number of waiting vehicles(speed less than 0.1m/s) on each lane. 
        "lane_waiting_time_count": get the sum of waiting time of vehicles on the lane since their last action. 
        "lane_delay": the delay of each lane: 1 - lane_avg_speed/speed_limit.
        "lane_pressure": the number of vehicles that in the in_lane minus number of vehicles that in out_lane.
        "pressure": difference of vehicle density between the in-coming lane and the out-going lane.

    :param in_only: boolean, whether to compute incoming lanes only. 
    :param average: None or str, None means no averaging, 
        "road" means take average of lanes on each road, 
        "all" means take average of all lanes.
    :param negative: boolean, whether return negative values (mostly for Reward).
    '''

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
        # ---------------------------------------------------------------------------------------------------------------

        # elif isinstance(world, world_openengine.World):
        #     for r in roads:
        #         if self.world.RIGHT:
        #             tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(str(ob)[-1]), reverse=True)
        #         else:
        #             tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(str(ob)[-1]))
        #         self.lanes.append(tmp)
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

    def _running_effective_num(self, lane_id, vehicles):
        ret = 0
        for vehicle in vehicles:
            vec_info = self.world.eng.get_vehicle_info(vehicle)
            distance = float(vec_info['distance'])
            speed = float(vec_info['speed'])
            max_speed = self.world.all_lanes_speed[lane_id]
            max_length = self.world.lane_length[lane_id]

            if distance + max_speed * (Registry.mapping['trainer_mapping']['setting']
                                               .param['action_interval'] + 5) > max_length and speed >= 0.1:
                ret += 1

        return ret

    def generate(self):
        # ============== Advance States ===================
        lane_vehicles = self.world.get_info('lane_vehicles')

        advance_ret = []
        for road_lanes in self.lanes:
            for lane_id in road_lanes:
                run_effc_num = self._running_effective_num(lane_id, lane_vehicles[lane_id])
                advance_ret.append(run_effc_num)

        # if self.world.eng.get_current_time() > 1000:
        #     pass

        # ============== Efficient States ===================
        result = self.world.get_info("lane_waiting_count")
        grouped_lanes = group_by_first(self.I.lanelinks)
        lvw_ret = []

        for road_lanes in self.lanes:
            for lane_id in road_lanes:
                # Efficient State
                lvw_ret.append(result[lane_id] - np.mean([result[to] for to in grouped_lanes[lane_id]]))
        return np.array(lvw_ret + advance_ret)
