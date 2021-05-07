from cores.game import Game
import numpy as np
import os
from utils.config import load_config
from PIL import Image, ImageTk
from cores.color_board import COLOR_BOARD
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Game()
    env.get_observation()
    env.enable_manual_control()
