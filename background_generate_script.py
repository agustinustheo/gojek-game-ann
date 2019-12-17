import os
import sys
import cv2
import random 
import pathlib
import numpy as np

if __name__ == '__main__':
    # loop through objects
    obstacle_arr=[]
    dirs = os.listdir('objects')
    for x in dirs:
        if str(os.path.splitext(x)[1]) == ".jpg":
            obstacle_arr.append(x)

    for obstacle_src in obstacle_arr:
        # open url with opencv
        bg = cv2.imread('game-snapshots/full_snap__bg.jpg')
        obstacle = cv2.imread('objects/'+obstacle_src)
        bg_width, bg_height, _ = bg.shape
        obstacle_width, obstacle_height, _ = obstacle.shape

        for i in range(1,1000):
            # random coordinates
            x=random.randint(0, bg_width-obstacle_width-50)
            y=random.randint(0, bg_height-obstacle_height-50)
            while x+obstacle_width+50>bg_width or y+obstacle_height+50>bg_height:
                x=random.randint(0, bg_width-obstacle_width-50)

            try:
                # img size 76x103
                img_crop=bg[x:x+obstacle_width+50, y:y+obstacle_height+50]

                obstacle_name=str(os.path.splitext(obstacle_src)[0])
                savepath='neg-'+obstacle_name + "/bg-" + obstacle_name + "-" + str(i) + ".jpg"

                # check if folder exists
                obstacle_directory = pathlib.Path('objects/neg-'+obstacle_name)
                if obstacle_directory.exists()==False:
                    os.mkdir('objects/neg-'+obstacle_name)

                # save background
                cv2.imwrite('objects/'+savepath, img_crop)
                with open('objects/'+obstacle_name+'.txt', 'a') as f:
                    f.write(savepath+'\n')
                print(savepath)
            except Exception as e:
                print(str(e))