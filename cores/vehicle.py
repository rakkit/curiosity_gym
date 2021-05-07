
class vehicleStatus:
    def __init__(self):
        self.position = [0, 0]
        self.direction = 1
        self.receptive_radius = 1
        self.communication_dis = 5


ACTION_MAP = {
    0: [0, 0],
    1: [-1, 0],
    2: [0, 1],
    3: [1, 0],
    4: [0, -1],
}

ACTION_SPACE = ACTION_MAP.__len__()


ACTION_MAP_INV = dict([(tuple(value), key) for key, value in ACTION_MAP.items()])

INV_ACTION = {
    1: 3,
    3: 1,
    2: 4,
    4: 2,
    0: 0
}

INV_ACTION_MAP = {
    1: [1, 0],
    3: [-1, 0],
    2: [0, -1],
    4: [0, 1],
    0: [0, 0]
}
