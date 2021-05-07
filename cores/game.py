import numpy as np
import os
from utils.config import load_config
import tkinter as tk
import tkinter.messagebox
from PIL import Image, ImageTk
from cores.color_board import COLOR_BOARD
import matplotlib.pyplot as plt
from utils.perlin_noise import PerlinNoiseFactory
from utils.ca_cave import CA_CaveFactory
from utils import utils
import copy
from threading import Thread
import utils.score


class vehicleStatus:
    position = [0, 0]
    direction = 1
    receptive_radius = 2


class Game:
    config = None
    map = None
    trajectory_map = None
    discovered_map = None
    vis_map = None
    world_width = 1
    world_height = 1
    n_vehicles = 1
    window = None
    canvas = None
    cell_width = 1
    cell_height = 1
    # vehicle_position = {}
    vehicle_status = {0: vehicleStatus()}
    vehicle_size = 1
    has_new_frame_to_render = False
    manual_control = False
    step = 0
    score = 0
    scoreRecorder = None

    ACTION_MAP = {
        0: [0, 0],
        1: [-1, 0],
        2: [0, 1],
        3: [1, 0],
        4: [0, -1],
    }
    ACTION_SPACE = ACTION_MAP.__len__()

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join('resources', 'configs', '2d.json')

        self.config = load_config(config_path)

        self.world_width, self.world_height = self.config['world_size']
        self.n_vehicles = self.config['num_vehicles']
        self.cell_width, self.cell_height = self.config['cell_size']
        self.vehicle_size = self.config['vehicle_size']
        self.has_new_frame_to_render = True

        self.setup()

    def setup(self):
        self.reset()
        x = Thread(target=self.vis_thread)
        x.start()
        self.render_new_frame()

    def reset(self):
        self.step = 0
        caf = CA_CaveFactory(self.world_height, self.world_width, self.config['ca_cave_open_prob'])
        self.map = caf.get_map()
        self.trajectory_map = np.zeros(self.map.shape)
        self.discovered_map = np.zeros(self.map.shape)
        idx_x, idx_y = np.where(self.map == 1)

        perlin_res = self.config['perlin_res']
        pnf = PerlinNoiseFactory(3, octaves=4, tile=(self.world_height / perlin_res,
                                                     self.world_width / perlin_res, 1))
        #
        for co_idx in range(len(idx_x)):
            n = (pnf(idx_x[co_idx] / perlin_res, idx_y[co_idx] / perlin_res, 1) + 1) / 2
            if n > self.config['perlin_prob']:
                self.map[idx_x[co_idx]][idx_y[co_idx]] = 3
        self.map[self.map > 0] -= 1
        '''
        0 -> wall
        1 -> Road
        2~100 -> curiosity
        101~ -> vehicle
        '''
        self.vis_map = np.zeros((self.world_height * self.cell_height, self.world_width * self.cell_width, 3))
        self.render_world()
        self.setup_vehicles()
        self.scoreRecorder = utils.score.scoreRecorder(self.map, self.vehicle_status[0].receptive_radius)

    def setup_vehicles(self):
        for vehicle_id in range(self.n_vehicles):
            _loc = utils.utils.generate_next_vehicle_random_pose(self.map)
            self.vehicle_status[vehicle_id].position = _loc
            self.vehicle_status[vehicle_id].direction = np.random.randint(1, 5)
            self.map[_loc[0], _loc[1]] = 101 + vehicle_id
            self.vehicle_status[vehicle_id].receptive_radius = self.config['vehicle_receptive_radius'][vehicle_id]

    def render_world(self):
        for c in np.unique(self.map):
            if c > 100:
                continue
            idx_x, idx_y = np.where(self.map == c)
            for _width in range(self.cell_width):
                for _height in range(self.cell_height):
                    self.vis_map[idx_x * self.cell_height + _height, idx_y * self.cell_width + _width, :] = \
                        COLOR_BOARD[int(c)]

    def map_to_vis_img(self):
        vis_map = copy.deepcopy(self.vis_map)
        flag = False
        for vehicle_id in range(self.n_vehicles):
            vehicle_x, vehicle_y = self.vehicle_status[vehicle_id].position
            xx, yy = utils.utils.generate_vehicle_coverage_idx(vehicle_x,
                                                               vehicle_y,
                                                               self.cell_width,
                                                               self.cell_height,
                                                               self.vehicle_size)

            vis_map[xx, yy, :] = COLOR_BOARD[101 + vehicle_id]

        vis_x, vis_y = np.where((self.map == 1) & (self.trajectory_map == 1))
        if len(vis_x) > 0:
            for _width in range(self.cell_width):
                for _height in range(self.cell_height):
                    vis_map[vis_x * self.cell_height + _height, vis_y * self.cell_width + _width, :] = \
                        COLOR_BOARD[131]

        return vis_map.astype('uint8')

    def render_new_frame(self, ):
        vis_map = self.map_to_vis_img()
        self.window.img = ImageTk.PhotoImage(image=Image.fromarray(vis_map, 'RGB'))
        self.canvas.create_image(0, 0, anchor="nw", image=self.window.img)
        self.window.step_label.config(text='Step: ' + str(self.step))
        self.window.score_label.config(text='Score : ' + str(self.score))

    def vis_thread(self):
        self.window = tk.Tk()
        h, w, _ = self.vis_map.shape

        # score_label = tk.Label(self.window, text=str(self.score))
        # self.window.score_label = score_label
        step_label = tk.Label(self.window, text='Step: ' + str(self.step))
        step_label.place(x=0, y=0, )
        step_label.pack()
        # step_label.config(width=100)
        self.window.step_label = step_label

        score_label = tk.Label(self.window, text='Score: ' + str(self.score))
        score_label.place(x=0, y=100)
        score_label.pack()
        # score_label.config(width=100)
        self.window.score_label = score_label

        self.canvas = tk.Canvas(self.window, width=w, height=h)
        self.canvas.pack()
        self.window.bind("<Key>", self.key_pressed)
        self.window.mainloop()

    def key_pressed(self, event):
        if self.manual_control:
            keyboard_map = {
                87: 1, 38: 1,
                83: 3, 40: 3,
                65: 4, 37: 4,
                68: 2, 39: 2,
            }
            if event.keycode in keyboard_map:
                self.move_by_action(keyboard_map[event.keycode])

    def move_by_action(self, action, vehicle_id=0):
        self.step += 1
        done = False
        dx, dy = self.ACTION_MAP[action]
        x, y = self.vehicle_status[vehicle_id].position
        done = self.map[x + dx][y + dy] != 1
        # done = utils.check_collision(self.map, dx, dy)

        if done:
            # print('COLLISION DETECTED, RESETTING NOW...')
            if self.manual_control:
                tk.messagebox.showinfo(title='hi',
                                       message='You score is ' + str(self.score) + ' After ' + str(
                                           self.step) + ' steps')  #
                self.reset()
        else:
            self.vehicle_status[vehicle_id].position = [x + dx, y + dy]
            self.map[x + dx][y + dy] = 101 + vehicle_id
            self.map[x][y] = 1
            self.trajectory_map[x][y] = 1
            # self.trajectory_map[x + dx][y + dy] = 1
            self.vehicle_status[vehicle_id].direction = action
        self.render_new_frame()
        obs = self.get_observation()
        self.score = self.scoreRecorder.get_score()
        return done, obs, [x + dx, y + dy]

    def enable_manual_control(self, enable=True):
        self.manual_control = enable

    def get_observation(self, vehicle_id=0):
        action = self.vehicle_status[vehicle_id].direction
        x, y = self.vehicle_status[vehicle_id].position
        dx, dy = self.ACTION_MAP[action]

        # xx = np.arange(max(x-self.vehicle_status[vehicle_id].receptive_radius, 0),
        #                min(x+self.vehicle_status[vehicle_id].receptive_radius+1, self.world_height), 1)
        #
        # yy = np.arange(max(y-self.vehicle_status[vehicle_id].receptive_radius, 0),
        #                min(y+self.vehicle_status[vehicle_id].receptive_radius+1, self.world_width), 1)

        xx = np.arange(x - self.vehicle_status[vehicle_id].receptive_radius,
                       x + self.vehicle_status[vehicle_id].receptive_radius + 1, 1)

        yy = np.arange(y - self.vehicle_status[vehicle_id].receptive_radius,
                       y + self.vehicle_status[vehicle_id].receptive_radius + 1, 1)
        xx[(xx < 0) | (xx >= self.world_height)] = 0
        yy[(yy < 0) | (yy >= self.world_width)] = 0

        xx, yy = np.meshgrid(xx, yy, sparse=True)
        # obs = self.map[xx, yy]
        obs = np.transpose(self.map[xx, yy])
        copy_obs = np.transpose(self.map[xx, yy])
        discovered = np.transpose(self.map[xx, yy])
        central = self.vehicle_status[vehicle_id].receptive_radius + 1
        # if action == 1:
        #     obs[central:, :] = -1
        #     # obs[:, :central-1] = -1
        # elif action == 3:
        #     obs[:central - 1, :] = -1
        # elif action == 2:
        #     obs[:, :central - 1] = -1
        # elif action == 4:
        #     obs[:, central:] = -1
        # obs[discovered != 0] = copy_obs[discovered != 0]
        obs[central - 1][central - 1] = -2
        self.scoreRecorder.push_data(obs, x, y)
        radius = self.vehicle_status[vehicle_id].receptive_radius
        dis_s = radius - radius // 2
        dis_e = radius + radius // 2 + 1

        obs_idx_x, obs_idx_y = np.where(obs[dis_s:dis_e, dis_s:dis_e] != -1)
        obs_idx_x = obs_idx_x + x - radius + dis_e
        obs_idx_y = obs_idx_y + y - radius + dis_e
        obs_idx_x[(obs_idx_x < 0) | (obs_idx_x >= self.world_height)] = 0
        obs_idx_y[(obs_idx_y < 0) | (obs_idx_y >= self.world_width)] = 0
        self.discovered_map[obs_idx_x, obs_idx_y] = 1
        discovered = np.transpose(self.discovered_map[xx, yy])
        trajectory = np.transpose(self.trajectory_map[xx, yy])
        return np.concatenate([np.expand_dims(obs, 0),
                               np.expand_dims(discovered, 0),
                               np.expand_dims(trajectory, 0)])

    def get_world_info(self):
        return self.map.shape

    def get_vehicle_status(self, vehicle_id=0):
        if vehicle_id not in self.vehicle_status:
            return None
        else:
            return self.vehicle_status[vehicle_id].__dict__

    def get_obs_info(self):
        info = {}
        for key in np.unique(self.map):
            if key == -1:
                info[key] = 'Unknown'
            elif key == -2:
                info[key] = 'Itself'
            elif key == 0:
                info[key] = 'Obstacle'
            elif key == 1:
                info[key] = 'Road'
            elif key < 101:
                info[key] = "Curiosity_{:d}".format(key - 1)
            else:
                info[key] = "Vehicle_{:d}".format(key - 100)
        info['obs_shape'] = self.get_observation()[0].shape
        return info

    def get_score(self):
        self.score = self.scoreRecorder.get_score()
        return self.score
