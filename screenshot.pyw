import os
import time
import pyautogui
from PIL import ImageGrab

def screenGrab():
    x=428
    y=172
    s5_width=267
    s5_height=477
    box = (x, y, x+s5_width, y+s5_height)
    im = ImageGrab.grab(box)
    im.save(os.getcwd() + '\\game-snapshots\\full_snap__' + str(int(time.time())) + '.png', 'PNG')

def main():
    screenGrab()

if __name__ == '__main__':
    main()