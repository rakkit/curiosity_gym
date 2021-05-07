import numpy as np
from utils.utils import metrics_np
import copy
import matplotlib.pyplot as plt



class scoreRecorder:
    gt_map = None
    discovered = None
    width = None
    height = None
    scale_x=None
    scale_y=None
    receptive_radius = None

    def __init__(self, world_map, receptive_radius):
        self.receptive_radius = receptive_radius
        self.reset(world_map)

    def push_data(self, observation, pos_x, pos_y):
        # print(observation)
        idx = np.where((observation > 1) & (observation < 101))
        if idx[0].__len__() > 0:
            global_x = idx[0] + pos_x - self.receptive_radius
            global_y = idx[1] + pos_y - self.receptive_radius
            _idx = (global_x>0) & (global_x<self.width) & (global_y>0) & (global_y<self.height)

            # print(self.gt_map[global_x, global_y], self.gt_map[global_y, global_x])
            self.discovered[global_x[_idx], global_y[_idx]] = observation[idx[0][_idx], idx[1][_idx]] - 1

    def get_score(self):
        metric = metrics_np(np.unique(self.gt_map).__len__())
        metric.update(self.gt_map.astype('int'), self.discovered.astype('int'))
        metric = metric.get_all()
        return metric['miou']

    def reset(self, world_map):
        self.gt_map = copy.deepcopy(world_map)
        self.gt_map[self.gt_map > 100] = 0
        self.gt_map[self.gt_map < 2] = 0
        self.gt_map[self.gt_map != 0] -= 1
        self.discovered = np.zeros(self.gt_map.shape)
        self.width, self.height = world_map.shape



