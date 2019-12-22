import os
import time
import pyautogui
from PIL import ImageGrab

def screen_grab():
    # x,y = 428,172
    x,y = 0,0
    s5_width=267
    s5_height=477
    capture_height=360
    box = (x, y, x+s5_width, y+capture_height)
    im = ImageGrab.grab(box)
    im.save(os.getcwd() + '\\game-snapshots\\full_snap__' + str(int(time.time())) + '.png', 'PNG')

if __name__ == '__main__':
    screen_grab()