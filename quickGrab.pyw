from PIL import ImageGrab
import os
import time

def screenGrab():
    box = (373,137,680,681)
    im = ImageGrab.grab(box)
    im.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) + '.png', 'PNG')

def main():
    screenGrab()

if __name__ == '__main__':
    main()
