import os
import time
import pygame
import random
import numpy as np
import pygame.locals
from agent import GameAgent
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

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

        # Model hyper parameters
        self.LEARNING_RATE = 1e-4

        # Set size of 20x30 frame and stack 4 frames
        self.img_rows , self.img_cols = 20,30
        self.img_channels = 4

        self.ACTIONS = 3

        self.agent = GameAgent()
        
        # Set timestep
        self.timesteps = 0

        # Set epsilon
        self.epsilon = 0.1

        # Move
        self.move_memory = deque()

        self.save_model = 1000

        if os.path.isfile("gamemodel.h5"):
            self.model = tf.keras.models.load_model("gamemodel.h5")
        else:
            self.model = self.build_model()

    def build_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(self.img_cols, self.img_rows, self.img_channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.ACTIONS))
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print("We finish building the model")
        return model

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
                        # Game Over
                        pygame.draw.rect(game_display, self.black, [10, 10, 10, 10])
                        crashed=True

                # Spawn obstacle
                # game_display.blit(obstacle[0], (position_coordinates[obstacle[1]], obstacle[2]))
                # Spawn obstacle replacement
                pygame.draw.rect(game_display, self.black, [position_coordinates[obstacle[1]], obstacle[2], obstacle_img_size[0], obstacle_img_size[1]])

                if obstacle[2] > self.display_height:
                    del generated_obstacles[i]

            spawn_timer-=1
            pygame.display.update()
        
        while True:
            for event in pygame.event.get():
                if hasattr(event, 'key'):
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.start_game()
                        elif event.key == pygame.K_DOWN:
                            self.quit_game()
                elif event.type == pygame.QUIT:
                    self.quit_game()

    def train_ai(self, train):
        # Set window spawn position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        reward = 0

        # Timesteps to observe before training and to remember
        observation = 50
        replay_memory = 50
        batch_to_train = 16

        print(self.timesteps)
        if self.timesteps > self.save_model:
            self.model.save("gamemodel.h5")
            self.save_model += 1000

        # Set epsilon and exploring to reduce epsilon
        explore = 10000
        initial_epsilon = 0.1
        final_epsilon = 0.0001

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

            # Get next step after performing the action
            image = self.agent.screen_grab()
            image = self.agent.process_img(image)
            # Stack 4 images to create placeholder input reshaped 1*30*20*4 
            stacked_image = np.stack((image, image, image, image), axis=2).reshape(1, self.img_cols, self.img_rows, self.img_channels)

            # Randomly explore an action
            if random.random() <= self.epsilon and train: 
                print("----------Random Action----------")
                action_index = random.randrange(self.ACTIONS)
            else: 
                # Predict the output
                print("----------Predict Action----------")
                q_values = self.model.predict(stacked_image)
                max_Q = np.argmax(q_values)
                action_index = max_Q 

            # Reduce epsilon gradually
            if self.epsilon > final_epsilon and self.timesteps > observation:
                self.epsilon -= (initial_epsilon - final_epsilon) / explore

            if action_index == 1:
                x_change=-92
                playerPos = 0
            elif action_index == 2:
                x_change=92
                playerPos = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_game()
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
                        # Game Over
                        pygame.draw.rect(game_display, self.black, [10, 10, 10, 10])
                        crashed=True

                        # Crashed then reward = -1
                        reward = -1

                # Spawn obstacle
                # game_display.blit(obstacle[0], (position_coordinates[obstacle[1]], obstacle[2]))
                # Spawn obstacle replacement
                pygame.draw.rect(game_display, self.black, [position_coordinates[obstacle[1]], obstacle[2], obstacle_img_size[0], obstacle_img_size[1]])

                if obstacle[2] > self.display_height:
                    del generated_obstacles[i]

            spawn_timer-=1
            pygame.display.update()

            self.timesteps += 1

            # Get next step after performing the action
            image, is_crashed = self.agent.get_state()
            # Stack 4 images to create placeholder input reshaped 1*30*20*4 
            stacked_image_1 = np.stack((image, image, image, image), axis=2).reshape(1, self.img_cols, self.img_rows, self.img_channels)

            # Save move result to memory
            self.move_memory.append((stacked_image, action_index, reward, stacked_image_1, is_crashed))
            # Remove memory when it hits the limit
            if len(self.move_memory) > replay_memory:
                self.move_memory.popleft()

            # When timesteps is more than observation, train.
            if self.timesteps > observation:
                minibatch = random.sample(self.move_memory, batch_to_train)
                inputs = np.zeros((batch_to_train, stacked_image.shape[1], stacked_image.shape[2], stacked_image.shape[3]))   #32, 20, 40, 4
                targets = np.zeros((inputs.shape[0], self.ACTIONS))                         #32, 2
                loss = 0
                
                for i in range(0, len(minibatch)):                
                    state_t = minibatch[i][0]    # 4D stack of images
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]   #reward at state_t due to action_t
                    state_t1 = minibatch[i][3]   #next state
                    terminal = minibatch[i][4]   #wheather the agent died or survided due the action
                    inputs[i:i + 1] = state_t    
                    targets[i] = self.model.predict(state_t)  # predicted q values
                    Q_sa = self.model.predict(state_t1)      #predict q values for next step
                    if terminal:
                        targets[i, action_t] = reward_t # if terminated, only equals reward
                    else:
                        targets[i, action_t] = reward_t + 0.9 * np.max(Q_sa)

                loss += self.model.train_on_batch(inputs, targets)
        
        self.train_ai()

    def quit_game(self):
        pygame.quit()
        quit()