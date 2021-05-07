import numpy as np
from cores.game import Game
import time


class curiosityGym:
    def __init__(self, config_path=None):
        self.env = Game(config_path)
        self.score = 0
        self.total_step = 0
        self.done = 0
        self.loop_detect = []
        self.loop_detect_dict = {}
        self.loop_detect_queen_length = 30
        self.loop_detect_threshold = 20
        self.last_score = 0

    def reset(self):
        self.env.reset()
        self.loop_detect = []
        self.loop_detect_dict = {}
        self.last_score = 0
        return self.env.get_observation()

    @staticmethod
    def reward_func(next_obs, total_step, score, last_score, done):
        # if score > 99:
        #     return 100
        #
        # idx = (next_obs[0] > 1) & (next_obs[1] < 101)
        # # reward = 0.1 + np.mean(idx)
        # # reward = 0.1 if total_step < 50 else 0
        #
        # # reward = - np.exp(total_step/1000)
        # central = next_obs[2].shape[0] // 2
        #
        # reward = 0.5 if next_obs[2][central][central] != 1 else 0
        #
        # # DISCOVERED
        # reward += 0.4*(np.exp(-next_obs[1].mean())-np.exp(-1))/(1-np.exp(-1))
        #
        # # TRAJECTORY
        # reward += 0.6*(np.exp(-next_obs[2].mean())-np.exp(-1))/(1-np.exp(-1))
        #
        # reward = 1.5*reward + 0.5*(np.exp(np.mean(idx)-1)-np.exp(-1))/(1-np.exp(-1))
        #
        # reward -= np.exp(total_step/10000)-0.8

        reward = score - last_score

        reward -= 0.01

        if score > 96:
            return 999

        if done:
            reward -= 1000
        return reward

    @staticmethod
    def get_action_info():
        """
        World coordinate:
        o--y
        |
        x
        :return: action is [dx, dy] in the world
        """

        return {
            'action_space': Game.ACTION_SPACE,
            'action_map': Game.ACTION_MAP,
        }

    def get_world_info(self):
        """
        :return: the size of the world
        """
        return self.env.get_world_info()

    def get_observation_info(self):
        """
        obs[0] what you can see
        obs[1] where you have discovered
        :return:
        """
        return self.env.get_obs_info()

    def get_vehicle_status(self, vehicle_id=0):
        """

        :param vehicle_id:
        :return: the status of vehicle: direction and location
        """
        return self.env.get_vehicle_status(vehicle_id)

    def get_score(self):
        return self.env.get_score()

    def step(self, action):
        self.done, next_obs, loc = self.env.move_by_action(action)
        self.score = self.env.score
        self.total_step = self.env.step
        reward = self.reward_func(next_obs, self.total_step, self.score, self.last_score, self.done)
        info = {'loop_detected': False}

        self.last_score = self.score

        self.loop_detect += [tuple(loc)]
        if self.loop_detect[-1] not in self.loop_detect_dict:
            self.loop_detect_dict[self.loop_detect[-1]] = 1
        else:
            self.loop_detect_dict[self.loop_detect[-1]] += 1
        if not self.done:
            loop_detected = self.loop_detect_dict[self.loop_detect[-1]] > self.loop_detect_threshold
            info['loop_detected'] = loop_detected
            # if loop_detected:
            #     reward -= 1

        return next_obs, reward, self.done, info


if __name__ == '__main__':
    env = curiosityGym()

    for _ in range(5):
        action = np.random.randint(1, 5)
        observation, reward, done, info = env.step(action)
        if done:
            print(env.score, env.total_step)
            env.reset()
        time.sleep(0.5)