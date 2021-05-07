from cores.game import Game
import numpy as np
import os
from PIL import Image, ImageTk
from cores.color_board import COLOR_BOARD
import matplotlib.pyplot as plt
import time
from cores.game import App

if __name__ == '__main__':
    game = Game()
    game.enable_manual_control()
    # app = App(game)
