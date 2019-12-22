import os
import cv2
import time
import pyautogui
import numpy as np
from PIL import ImageGrab

class GameAgent:
    def screen_grab(self):
        # x,y = 428,172
        x,y = 0,0
        s5_width=267
        s5_height=477
        capture_height=360
        box = (x, y, x+s5_width, y+capture_height)
        im = np.array(ImageGrab.grab(box))
        return im

    def move_left(self):
        pyautogui.keyDown('left')
        time.sleep(0.2)
        pyautogui.keyUp('left')

    def move_right(self):
        pyautogui.keyDown('right')
        time.sleep(0.2)
        pyautogui.keyUp('right')

    def click_game(self):
        pyautogui.click(x=100, y=365)