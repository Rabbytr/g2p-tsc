from world.world_cityflow import Intersection, World
import numpy as np


def upstream_features(inter: Intersection, world: World, feature_mapping):
    up_features = {}
    for in_road in inter.in_roads:
        up_features[in_road['id']] = get_commu_term(in_road['id'],
                                                    world.id2intersection.get(in_road['startIntersection']),
                                                    feature_mapping)
    up_lane_features = {}
    for key, value in up_features.items():
        lane_id = f'{key}_0'
        up_lane_features[lane_id] = np.mean([value, feature_mapping[lane_id]])
        lane_id = f'{key}_1'
        up_lane_features[lane_id] = np.mean([value, feature_mapping[lane_id]])
    return up_lane_features


def get_commu_term(to_road, inter: Intersection, feature_mapping):
    if inter is None:
        return np.mean([feature_mapping[f'{to_road}_{i}'] for i in range(3)])

    features = []
    vis = set()
    for fr_lane, to_lane in inter.lanelinks:
        if to_lane.startswith(to_road):
            if fr_lane in vis: continue
            vis.add(fr_lane)
            features.append(feature_mapping[fr_lane])

    return np.mean(features)
