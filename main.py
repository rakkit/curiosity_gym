from cores.game import Game
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
from threading import Thread
from cores.game import App
import tkinter as tk
from multiprocessing import Process, Queue



if __name__ == '__main__':
    game = Game()
    game.enable_manual_control()
    app = App(game)


