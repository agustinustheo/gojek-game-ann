import os
import pygame
import random


# Define colors
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
gray = (141,139,160)

# Define size
display_width=280
display_height=449
padding=41

# Initialize game
pygame.init()
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Gojek - Pasti Ada Jalan')
clock = pygame.time.Clock()

playerImg = pygame.image.load('objects/player.png')
player_img_size = playerImg.get_rect().size

# Get background
background = pygame.image.load('game-snapshots/full_snap__bg.png')

array_of_obstacles = []
dirs = os.listdir('objects')
for x in dirs:
    if str(os.path.splitext(x)[1]) == ".png" and str(os.path.splitext(x)[0]) != "warning" and str(os.path.splitext(x)[0]) != "player":
        array_of_obstacles.append('objects/'+x)

def player(x, y):
    # Spawn player image
    gameDisplay.blit(playerImg, (x, y))
    # Spawn player replacement
    pygame.draw.rect(gameDisplay, red, [x, y, player_img_size[0], player_img_size[1]])

def add_obstacles():
    rand_index=random.randrange(0,len(array_of_obstacles))
    obstacle = pygame.image.load(array_of_obstacles[rand_index])
    return obstacle

def game_loop():
    x_change=0
    spawn_timer=0
    obstacle_speed=10

    playerPos=0
    position_coordinates=[65, 157]
    
    x = position_coordinates[playerPos]
    y = display_height-86

    generated_obstacles=[]
    crashed = False
    while not crashed:
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change=-92
                    playerPos = 0
                elif event.key == pygame.K_RIGHT:
                    x_change=92
                    playerPos = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    x_change=0

        # Create boundary
        if x+x_change <= display_width-padding-player_img_size[0] and x+x_change >= padding:
            x+=x_change

        # Use background
        gameDisplay.blit(background,(0,0))
        # Fill with white
        gameDisplay.fill(white)

        # Spawn player
        player(x, y)

        if len(generated_obstacles) <= 1 and random.randrange(0,5) == 0 and spawn_timer<=0:
            # To delay obstacle spawn
            spawn_timer=27
            # Randomize position
            pos=random.randrange(0,2)
            # Generate obstacle and append to obstacle array
            generated_obstacles.append([add_obstacles(), pos, -100])

        for i, obstacle in enumerate(generated_obstacles):
            obstacle[2]+=obstacle_speed
            obstacle_img_size = obstacle[0].get_rect().size
            # Spawn obstacle
            gameDisplay.blit(obstacle[0], (position_coordinates[obstacle[1]], obstacle[2]))
            # Spawn obstacle replacement
            pygame.draw.rect(gameDisplay, black, [position_coordinates[obstacle[1]], obstacle[2], obstacle_img_size[0], obstacle_img_size[1]])
            if obstacle[2] > display_height:
                del generated_obstacles[i]

        spawn_timer-=1
        pygame.display.update()

game_loop()
pygame.quit()
quit()