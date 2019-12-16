from PIL import ImageGrab
import os
import pyautogui
import time
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys

def screenGrab():
    x=428
    y=172
    s5_width=267
    s5_height=477
    box = (x, y, x+s5_width, y+s5_height)
    im = ImageGrab.grab(box)
    im.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) + '.png', 'PNG')

def moveLeft():
    pyautogui.keyDown('left')
    time.sleep(0.2)
    pyautogui.keyUp('left')

def moveRight():
    pyautogui.keyDown('right')
    time.sleep(0.2)
    pyautogui.keyUp('right')

def refreshPage():
    pyautogui.keyDown('ctrl')
    pyautogui.keyDown('r')
    pyautogui.keyUp('ctrl')
    pyautogui.keyUp('r')
    time.sleep(0.5)

def main():
    # screenGrab()
    driver = webdriver.Chrome()
    driver.get("https://www.google.com")

if __name__ == '__main__':
    main()