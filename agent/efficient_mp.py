from . import MaxPressureAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import numpy as np
import gym
from .utils import group_by_first


@Registry.register_model('efficient_mp')
class EfficientMP(MaxPressureAgent):
    def get_action(self, ob, phase, test=True):
        lvw = self.world.get_info("lane_waiting_count")
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        max_pressure = None
        action = -1
        for phase_id in range(len(self.inter_obj.phases)):

            grouped_lanes = group_by_first(
                [(k, v) for k, v in self.inter_obj.phase_available_lanelinks[phase_id] if not k.endswith('2')])

            # pressure = sum([lvc[start] - lvc[end]
            #                 for start, end in self.inter_obj.phase_available_lanelinks[phase_id]
            #                 if not start.endswith('2')])

            pressure = sum([
                lvw[fro] - np.mean([lvw[to] for to in tos]) for fro, tos in grouped_lanes.items()
            ])

            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action
