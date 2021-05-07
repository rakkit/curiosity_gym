import json
import os

default_config = {
    "world_size": [100, 50],
    "num_vehicles": 1,
    "cell_size": [10, 10],
    "vehicle_size": [10],
    "ca_cave_open_prob": 0.85,
    "perlin_res": 40,
    "perlin_prob": 0.5,
    "vehicle_receptive_radius": [2],
    "seed": 123,
    "vehicle_communication_dis": [5]

}


class Config:
    @staticmethod
    def load_config(path):
        configurations = json.load(open(path, 'r'))
        for key, value in default_config.items():
            if key in configurations:
                if isinstance(value, list) and len(value) == len(configurations[key]):
                    default_config[key] = configurations[key]
                else:
                    default_config[key] = configurations[key]

        if default_config['cell_size'][0] != default_config['cell_size'][1]:
            default_config['cell_size'][0] = default_config['cell_size'][1]
            default_config['vehicle_size'] = default_config['cell_width'][0]

        if default_config['num_vehicles'] > 1:
            for key in ['vehicle_size', 'vehicle_receptive_radius', 'vehicle_communication_dis']:
                if len(default_config[key]) != default_config['num_vehicles']:
                    default_config[key] += default_config[key]*(default_config['num_vehicles'] - len(default_config[key]))

        return default_config
