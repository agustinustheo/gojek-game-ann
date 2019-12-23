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
        capture_height=460
        box = (x, y, x+s5_width, y+capture_height)
        img = np.array(ImageGrab.grab(box))
        return img

    def process_img(self, image):
        # Resize image dimensions
        image = cv2.resize(image, (0,0), fx = 0.075, fy = 0.0652) 
        # Apply the canny edge detection
        image = cv2.Canny(image, threshold1 = 100, threshold2 = 200)
        return  image   

    def is_crashed(self, img):
        if np.any(img[11, 11] != 0):
            return False
        return True

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

    def get_state(self):
        img = self.screen_grab()
        return self.process_img(img), self.is_crashed(img)
