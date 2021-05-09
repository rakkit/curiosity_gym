import numpy as np
import os
from utils.config import Config
import tkinter as tk
import tkinter.messagebox
from PIL import Image, ImageTk
from cores.color_board import COLOR_BOARD
import matplotlib.pyplot as plt
from utils.perlin_noise import PerlinNoiseFactory
from utils.ca_cave import CA_CaveFactory
from utils import utils, score
import copy
from threading import Thread
import time
from cores.vehicle import vehicleStatus, ACTION_MAP
import random
import pygame

class App:
    _img = None
    _score = '0'
    _step = '0'

    def __init__(self, game):
        pygame.init()

        self.game = game
        self.width, self.height = game.get_resolution()

        display_size = (self.width, self.height + 100)
        self.display = pygame.display.set_mode(display_size)
        self.running = True
        self.lazy = True
        # self.start()

    def update_score(self, score=0):
        self._score = str(score)
        self.lazy = False

    def update_step(self, step=0):
        self._step = str(step)
        self.lazy = False

    def update_img(self, img=None):
        self._img = np.transpose(img, (1, 0, 2))
        self.lazy = False

    def close(self):
        self.running = False

    def run(self):
        running = True
        while running and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.game.move_by_action(1)
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        self.game.move_by_action(3)
                    elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.game.move_by_action(4)
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.game.move_by_action(2)

            if not self.lazy:
                self.display.fill((0, 0, 0))

                if self._img is not None:
                    surf = pygame.surfarray.make_surface(self._img)
                    self.display.blit(surf, (0, 100))

                font = pygame.font.Font(None, 30)

                score_label = font.render("Score: {:s}".format(self._score), True, (255, 255, 255))
                self.display.blit(score_label, (self.width // 3, 10))

                step_label = font.render("Step: {:s}".format(self._step), True, (255, 255, 255))
                self.display.blit(step_label, (self.width // 3, 50))
                pygame.display.update()
            self.lazy = True

        pygame.quit()


class Game:
    _config = None
    _world_width = 1
    _world_height = 1
    _window = None
    _canvas = None
    _cell_width = 1
    _cell_height = 1

    manual_control = False

    world = None
    trajectory_map = None
    discovered_map = None
    bg_world = None

    n_vehicles = 1
    vehicle_status = {}
    vehicle_distance = None

    step = 0
    score = 0
    scoreRecorder = None

    action_map = ACTION_MAP
    action_space = ACTION_MAP.__len__()

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join('resources', 'configs', '2d.json')

        self.config = Config.load_config(config_path)

        self._world_width, self._world_height = self.config['world_size']
        self.n_vehicles = self.config['num_vehicles']
        self._cell_width, self._cell_height = self.config['cell_size']

        self.seed(self.config['seed'])
        self.setup()

    @staticmethod
    def seed(_seed=123):
        random.seed(_seed)
        np.random.seed(_seed)

    def get_resolution(self):
        return (self._world_width * self._cell_width,
                self._world_height * self._cell_height)

    def setup(self):
        self.reset()
        # self._window = App(self)
        Thread(target=self.init_window).start()
        time.sleep(1)
        self.render_new_frame()

    def init_window(self, ):
        self._window = App(self)
        self._window.run()

    def destroy_window(self):
        self._window.close()
        self._window = None

    def reset(self):
        self.step = 0
        self.score = 0
        # Generate Obstacle
        caf = CA_CaveFactory(self._world_height, self._world_width, self.config['ca_cave_open_prob'])
        self.world = caf.get_map()
        idx_x, idx_y = np.where(self.world == 1)

        # Generate Curiosity  && Replace WALL TO Curiosity By perlin_prob
        perlin_res = self.config['perlin_res']
        pnf = PerlinNoiseFactory(3, octaves=4, tile=(self._world_height / perlin_res,
                                                     self._world_width / perlin_res, 1))

        for co_idx in range(len(idx_x)):
            n = (pnf(idx_x[co_idx] / perlin_res, idx_y[co_idx] / perlin_res, 1) + 1) / 2
            if n > self.config['perlin_prob']:
                self.world[idx_x[co_idx]][idx_y[co_idx]] = 3

        # Rescale map value
        # 0 -> wall
        # 1 -> Road
        # 2~100 -> curiosity
        # 101~150 -> vehicle
        self.world[self.world > 0] -= 1

        # Init the visualization window
        self.bg_world = np.zeros((self._world_height * self._cell_height, self._world_width * self._cell_width, 3))
        self.setup_vehicles()
        self.render_background()
        self.scoreRecorder = score.scoreRecorder(self.world, self.vehicle_status[0].receptive_radius)

        if self._window:
            self.render_new_frame()

    def setup_vehicles(self):
        """
            Init the vehicle settings
        """
        self.trajectory_map = {}
        self.discovered_map = {}
        for vehicle_id in range(self.n_vehicles):
            self.vehicle_status[vehicle_id] = vehicleStatus()
            _loc = utils.generate_next_vehicle_random_pose(self.world)
            self.vehicle_status[vehicle_id].position = _loc
            self.vehicle_status[vehicle_id].direction = np.random.randint(1, 5)
            self.vehicle_status[vehicle_id].receptive_radius = self.config['vehicle_receptive_radius'][vehicle_id]
            self.vehicle_status[vehicle_id].communication_dis = self.config['vehicle_communication_dis'][vehicle_id]
            self.world[_loc[0], _loc[1]] = 101 + vehicle_id
            self.trajectory_map[vehicle_id] = np.zeros(self.world.shape)
            self.discovered_map[vehicle_id] = np.zeros(self.world.shape)

        self.vehicle_distance = np.zeros((self.n_vehicles, self.n_vehicles))

    def render_background(self):
        for c in np.unique(self.world):
            if c > 100:
                color = 1
            else:
                color = int(c)
            idx_x, idx_y = np.where(self.world == c)

            for _width in range(self._cell_width):
                for _height in range(self._cell_height):
                    self.bg_world[idx_x * self._cell_height + _height, idx_y * self._cell_width + _width, :] = \
                        COLOR_BOARD[color]

    def render_world(self):
        vis_map = copy.deepcopy(self.bg_world)
        for vehicle_id in range(self.n_vehicles):
            vehicle_x, vehicle_y = self.vehicle_status[vehicle_id].position
            xx, yy = utils.generate_vehicle_coverage_idx(vehicle_x,
                                                         vehicle_y,
                                                         self._cell_width,
                                                         self._cell_height,
                                                         self._cell_width)

            vis_x, vis_y = np.where(self.trajectory_map[vehicle_id] == 1)
            if len(vis_x) > 0:
                for _width in range(self._cell_width):
                    for _height in range(self._cell_height):
                        vis_map[vis_x * self._cell_height + _height, vis_y * self._cell_width + _width, :] = \
                            COLOR_BOARD[151 + vehicle_id]

            vis_map[xx, yy, :] = COLOR_BOARD[101 + vehicle_id]

        return vis_map.astype('uint8')

    def render_new_frame(self, ):
        vis_map = self.render_world()
        if self._window:
            self._window.update_img(vis_map)
            self._window.update_score(self.score)
            self._window.update_step(self.step)

    def move_by_action(self, action, vehicle_id=0):
        self.step += 1
        dx, dy = ACTION_MAP[action]
        x, y = self.vehicle_status[vehicle_id].position
        done = self.world[x + dx][y + dy] != 1

        if done:
            if self.manual_control:
                root = tk.Tk()
                root.withdraw()
                tkinter.messagebox.showinfo(title='Failed !',
                                            message='You score is ' + str(self.score) + ' After ' + str(
                                                self.step) + ' steps')  #
                self.reset()
        else:
            self.vehicle_status[vehicle_id].position = [x + dx, y + dy]
            self.world[x + dx][y + dy] = 101 + vehicle_id
            self.world[x][y] = 1
            self.trajectory_map[vehicle_id][x][y] = 1
            self.vehicle_status[vehicle_id].direction = action
            self.update_distance_of_vehicles(vehicle_id, x + dx, y + dy)

        self.render_new_frame()
        obs = self.get_observation()
        self.score = self.scoreRecorder.get_score()
        return done, obs, [x + dx, y + dy]

    def enable_manual_control(self, enable=True):
        self.manual_control = enable

    def get_observation(self, vehicle_id=0):
        action = self.vehicle_status[vehicle_id].direction
        x, y = self.vehicle_status[vehicle_id].position
        dx, dy = ACTION_MAP[action]

        xx = np.arange(x - self.vehicle_status[vehicle_id].receptive_radius,
                       x + self.vehicle_status[vehicle_id].receptive_radius + 1, 1)

        yy = np.arange(y - self.vehicle_status[vehicle_id].receptive_radius,
                       y + self.vehicle_status[vehicle_id].receptive_radius + 1, 1)
        xx[(xx < 0) | (xx >= self._world_height)] = 0
        yy[(yy < 0) | (yy >= self._world_width)] = 0

        xx, yy = np.meshgrid(xx, yy, sparse=True)
        obs = np.transpose(self.world[xx, yy])
        # discovered = np.transpose(self.world[xx, yy])

        central = self.vehicle_status[vehicle_id].receptive_radius + 1

        # the central of observation is the drone itself
        obs[central - 1][central - 1] = - 2

        self.scoreRecorder.push_data(obs, x, y)
        radius = self.vehicle_status[vehicle_id].receptive_radius
        dis_s = radius - radius // 2
        dis_e = radius + radius // 2 + 1

        obs_idx_x, obs_idx_y = np.where(obs[dis_s:dis_e, dis_s:dis_e] != -1)
        obs_idx_x = obs_idx_x + x - radius + dis_e
        obs_idx_y = obs_idx_y + y - radius + dis_e
        obs_idx_x[(obs_idx_x < 0) | (obs_idx_x >= self._world_height)] = 0
        obs_idx_y[(obs_idx_y < 0) | (obs_idx_y >= self._world_width)] = 0
        self.discovered_map[vehicle_id][obs_idx_x, obs_idx_y] = 1
        discovered = np.transpose(self.discovered_map[vehicle_id][xx, yy])
        trajectory = np.transpose(self.trajectory_map[vehicle_id][xx, yy])
        return np.concatenate([np.expand_dims(obs, 0),
                               np.expand_dims(discovered, 0),
                               np.expand_dims(trajectory, 0)])

    def get_world_info(self):
        return self.world.shape

    def update_distance_of_vehicles(self, vehicle_id, x, y):
        for i in range(self.n_vehicles):
            if i == vehicle_id:
                dis = 0
            else:
                loc1 = self.vehicle_status[i].position
                dis = np.linalg.norm(np.array(loc1) - np.array([x, y]))
            self.vehicle_distance[i][vehicle_id] = dis
            self.vehicle_distance[vehicle_id][i] = dis

    def swap_vehicle_infomation(self, ):
        for i in range(self.n_vehicles):
            for j in range(1, self.n_vehicles):
                dis = self.vehicle_distance[i][j]
                if dis < self.vehicle_status[i].communication_dis and dis < self.vehicle_status[j].communication_dis:
                    self.trajectory_map[i] = self.trajectory_map[j] = \
                        (self.trajectory_map[i] | self.trajectory_map[j]).astype('uint8')
                    self.discovered_map[i] = self.discovered_map[j] = \
                        (self.discovered_map[i] | self.discovered_map[j]).astype('uint8')

    def get_vehicle_status(self, vehicle_id=0):
        if vehicle_id not in self.vehicle_status:
            return None
        else:
            return self.vehicle_status[vehicle_id].__dict__

    def get_obs_info(self):
        info = {}
        for key in np.unique(self.world):
            if key == -1:
                info[key] = 'Unknown'
            elif key < -1:
                info[key] = 'some vehicle'
            elif key == 0:
                info[key] = 'Obstacle'
            elif key == 1:
                info[key] = 'Road'
            elif key < 101:
                info[key] = "Curiosity_{:d}".format(key - 1)
            elif key < 151:
                info[key] = "Vehicle_{:d}".format(key - 100)
        info['obs_shape'] = self.get_observation()[0].shape
        return info

    def get_score(self):
        self.score = self.scoreRecorder.get_score()
        return self.score
