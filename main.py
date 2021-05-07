from cores.game import Game
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
import multiprocessing

from cores.game import App
import tkinter as tk


def train_your_model(*args, **kwargs):
    print('here is the processing for training your env.')


'''
    Notes : To make sure it works on both Win/Mac, it is important
    to call the training in main func.
'''
game = Game()
game.enable_manual_control()
app = App(tk.Tk(), game)
game.set_window(app)

if __name__ == '__main__':
    q = multiprocessing.Queue()
    multiprocessing.Process(target=train_your_model, args=(game,)).start()

app.window.mainloop()


