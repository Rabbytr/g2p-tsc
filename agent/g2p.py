from . import MaxPressureAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, RouteAwareStateGenerator
import numpy as np
import gym
from .utils import group_by_first


@Registry.register_model('g2p')
class G2p(MaxPressureAgent):
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.world = world
        self.rank = rank
        self.model = None

        # get generator for each MaxPressure
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = RouteAwareStateGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count", "lane_vehicles"], in_only=True,
                                                     average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)

        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                          ["lane_waiting_count"], in_only=True,
                                          negative=False)

        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                          ["lane_delay"], in_only=True,
                                          negative=False)
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))

        # the minimum duration of time of one phase
        self.t_min = Registry.mapping['model_mapping']['setting'].param['t_min']
        # self.t_min = 10
        pass

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
        # get generator for each MaxPressure
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = RouteAwareStateGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count", "lane_vehicles"], in_only=True,
                                                     average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)

        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                          ["lane_waiting_count"], in_only=True,
                                          negative=False)

        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                          ["lane_delay"], in_only=True,
                                          negative=False)

    def get_action(self, ob, phase, test=True):
        lvw = self.world.get_info("lane_waiting_count")
        # lvw = self.world.get_info("lane_count")

        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        lane_ids = []
        for road_lanes in self.ob_generator.lanes:
            for lane_id in road_lanes:
                lane_ids.append(lane_id)
        eff_running = self.ob_generator.generate()[12:-12]
        eff_running_mapping = {k: v for k, v in zip(lane_ids, eff_running)}

        route_state = self.ob_generator.generate()[-12:]
        route_state_mapping = {k: v for k, v in zip(lane_ids, route_state)}

        route_p = []
        for phase_id in range(len(self.inter_obj.phases)):
            grouped_lanes = group_by_first(
                [(k, v) for k, v in self.inter_obj.phase_available_lanelinks[phase_id] if not k.endswith('2')])
            pressure = sum([route_state_mapping[fro] for fro in grouped_lanes.keys()])
            route_p.append(pressure)

        action = np.argmax(route_p)

        return action
