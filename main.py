from cores.game import Game
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
from threading import Thread
from cores.game import App
import tkinter as tk
import pygame

#
# class Viewer():
#     _img = None
#     _score = 0
#     _step = 0
#
#     def __init__(self, update_func):
#         self.update_func = update_func
#         pygame.init()
#
#         self.width = 600
#         self.height = 600
#
#         display_size = (self.width, self.height+100)
#         self.display = pygame.display.set_mode(display_size)
#
#         # self.start()
#
#     def update_score(self, score=0):
#         self._score = score
#
#     def update_step(self, step=0):
#         self._step = step
#
#     def update_img(self, img=None):
#         self._img = img
#
#     def run(self):
#         running = True
#         while running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_w or event.key == pygame.K_UP:
#                         print('up')
#                     elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
#                         print('down')
#             Z = self.update_func()
#             surf = pygame.surfarray.make_surface(Z)
#             self.display.blit(surf, (0, 100))
#
#             font = pygame.font.Font(None, 30)
#
#             score_label = font.render("Score: {:d}".format(self._score), True, (255, 255, 255))
#             self.display.blit(score_label, (self.width//3, 10))
#
#             step_label = font.render("Step: {:d}".format(self._step), True, (255, 255, 255))
#             self.display.blit(step_label, (self.width//3, 50))
#             pygame.display.update()
#
#         pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.enable_manual_control()
#
#
# def update():
#     image = np.random.random((600, 600, 3)) * 255.0
#     image[:, :200, 0] = 255.0
#     image[:, 200:400, 1] = 255.0
#     image[:, 400:, 2] = 255.0
#     return image.astype('uint8')
#
#
# def vis_thread():
#     viewer = Viewer(update)
#     viewer.run()
#
#
# if __name__ == '__main__':
#     x = Thread(target=vis_thread)
#     x.start()
#
#     print('11111')
