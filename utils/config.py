import json
import os


class config:
    world_size = [1, 1]


def load_config(path):
    configurations = json.load(open(path, 'r'))
    return configurations
