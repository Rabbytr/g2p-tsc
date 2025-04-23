from . import MaxPressureAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, AdvanceStateGenerator
import numpy as np
import gym
from .utils import group_by_first


@Registry.register_model('advance_mp')
class AdvanceMP(MaxPressureAgent):
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.world = world
        self.rank = rank
        self.model = None

        # get generator for each MaxPressure
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = AdvanceStateGenerator(self.world, self.inter_obj, ["lane_waiting_count", "lane_vehicles"],
                                                  in_only=True, average=None)
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
        self.ob_generator = AdvanceStateGenerator(self.world, self.inter_obj, ["lane_waiting_count", "lane_vehicles"],
                                                  in_only=True, average=None)
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
        eff_running = self.ob_generator.generate()[-12:]
        eff_running_mapping = {k: v for k, v in zip(lane_ids, eff_running)}

        pressures = []
        for phase_id in range(len(self.inter_obj.phases)):
            # grouped_lanes = group_by_first(
            #     [(k, v) for k, v in self.inter_obj.phase_available_lanelinks[phase_id] if not k.endswith('2')])
            # pressure = sum([
            #     lvw[fro] - np.sum([lvw[to] for to in tos]) for fro, tos in grouped_lanes.items()
            # ])

            pressure = sum([lvw[start] - lvw[end]
                            for start, end in self.inter_obj.phase_available_lanelinks[phase_id]
                            if not start.endswith('2')])

            pressures.append(pressure)

        cur_phase = self.inter_obj.current_phase

        grouped_lanes = group_by_first(
            [(k, v) for k, v in self.inter_obj.phase_available_lanelinks[cur_phase] if not k.endswith('2')])

        adv = sum([eff_running_mapping[fro_lane] for fro_lane in grouped_lanes.keys()])

        if adv * 1 >= np.max(pressures):
            action = cur_phase
        else:
            action = np.argmax(pressures)

        return action
