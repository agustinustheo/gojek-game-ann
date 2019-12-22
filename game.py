import os
import time
import pygame
import random
import pygame.locals

class Game:
    def __init__(self):
        # Define colors
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.red = (255,0,0)
        self.green = (0,255,0)
        self.blue = (0,0,255)
        self.gray = (141,139,160)

        # Define size
        self.display_width=280
        self.display_height=449
        self.padding=41

        self.player_img = pygame.image.load('objects/player.png')
        self.player_img_size = self.player_img.get_rect().size

        # Get background
        # background = pygame.image.load('game-snapshots/full_snap__bg.png')

        self.array_of_obstacles = []
        self.dirs = os.listdir('objects')
        for x in self.dirs:
            if str(os.path.splitext(x)[1]) == ".png" and str(os.path.splitext(x)[0]) != "warning" and str(os.path.splitext(x)[0]) != "player":
                self.array_of_obstacles.append('objects/'+x)

    def player(self, x, y, game_display):
        # Spawn player image
        # game_display.blit(playerImg, (x, y))
        # Spawn player replacement
        pygame.draw.rect(game_display, self.red, [x, y, self.player_img_size[0], self.player_img_size[1]])

    def add_obstacles(self, pygame):
        rand_index=random.randrange(0,len(self.array_of_obstacles))
        obstacle = pygame.image.load(self.array_of_obstacles[rand_index])
        return obstacle

    def start_game(self):
        # Set window spawn position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

        # Initialize game
        pygame.init()
        pygame.display.set_caption('Gojek - Pasti Ada Jalan')
        game_display = pygame.display.set_mode((self.display_width, self.display_height))
        clock = pygame.time.Clock()

        x_change=0
        spawn_timer=0
        obstacle_speed=10

        playerPos=0
        position_coordinates=[65, 157]
        
        x = position_coordinates[playerPos]
        y = self.display_height-86

        generated_obstacles=[]
        crashed = False
        while not crashed:
            clock.tick(15)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_game()
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
            if x+x_change <= self.display_width-self.padding-self.player_img_size[0] and x+x_change >= self.padding:
                x+=x_change

            # Use background
            # game_display.blit(background,(0,0))
            # Fill with white
            game_display.fill(self.white)

            # Spawn player
            self.player(x, y, game_display)

            if len(generated_obstacles) <= 1 and random.randrange(0,5) == 0 and spawn_timer<=0:
                # To delay obstacle spawn
                spawn_timer=27
                # Randomize position
                pos=random.randrange(0,2)
                # Generate obstacle and append to obstacle array
                generated_obstacles.append([self.add_obstacles(pygame), pos, -100])

            for i, obstacle in enumerate(generated_obstacles):
                obstacle[2]+=obstacle_speed
                obstacle_img_size = obstacle[0].get_rect().size

                # Check if player is in the same lane as obstacle
                if playerPos==obstacle[1]:
                    # Check if obstacle is in the player OR if the player is in the obstacle
                    if ((y+self.player_img_size[1] >= obstacle[2]+obstacle_img_size[1] and y <= obstacle[2]+obstacle_img_size[1]) or (obstacle[2]+obstacle_img_size[1] >= y+self.player_img_size[1] and obstacle[2] <= y+self.player_img_size[1])):
                        crashed=True

                # Spawn obstacle
                # game_display.blit(obstacle[0], (position_coordinates[obstacle[1]], obstacle[2]))
                # Spawn obstacle replacement
                pygame.draw.rect(game_display, self.black, [position_coordinates[obstacle[1]], obstacle[2], obstacle_img_size[0], obstacle_img_size[1]])

                if obstacle[2] > self.display_height:
                    del generated_obstacles[i]

            spawn_timer-=1
            pygame.display.update()

        # Game Over
        pygame.font.SysFont('Arial', 20).render('Game Over', False, (0, 0, 0)) 
        pygame.display.update()
        
        while True:
            for event in pygame.event.get():
                if hasattr(event, 'key'):
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.start_game()
                        else:
                            self.quit_game()
                elif event.type == pygame.QUIT:
                    self.quit_game()

    def quit_game(self):
        pygame.quit()
        quit()