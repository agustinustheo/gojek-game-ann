import numpy as np
from PIL import Image
from PIL import ImageGrab
from agent import GameAgent
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.layers.core import Dense, Dropout, Activation, Flatten
    
#game parameters
GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 50000. # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def process_img(image):
    # Resize image dimensions
    image = cv2.resize(image, (0,0), fx = 0.15, fy = 0.10) 
    # Apply the canny edge detection
    image = cv2.Canny(image, threshold1 = 100, threshold2 = 200)
    return  image   

# Model hyper parameters
LEARNING_RATE = 1e-4

# Set size of 20x30 frame and stack 4 frames
img_rows , img_cols = 20,30
img_channels = 4

ACTIONS = 3
def build_model():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_cols,img_rows,img_channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def trainBatch(minibatch):
    inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 20, 40, 4
    targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2
    loss = 0
    
    for i in range(0, len(minibatch)):                
        state_t = minibatch[i][0]    # 4D stack of images
        action_t = minibatch[i][1]   #This is action index
        reward_t = minibatch[i][2]   #reward at state_t due to action_t
        state_t1 = minibatch[i][3]   #next state
        terminal = minibatch[i][4]   #wheather the agent died or survided due the action
        inputs[i:i + 1] = state_t    
        targets[i] = model.predict(state_t)  # predicted q values
        Q_sa = model.predict(state_t1)      #predict q values for next step
        if terminal:
            targets[i, action_t] = reward_t # if terminated, only equals reward
        else:
            targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

    loss += model.train_on_batch(inputs, targets)

''' 
Parameters:
* model => Keras Model to be trained
'''
def trainNetwork(model, agent):
    # Store the previous observations in replay memory
    D = deque()
    
    # Get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1 #0 => Do nothing
    do_nothing[1] = 2 #1 => Go left
                      #2 => Go right
    
    image, is_crashed = agent.get_state(do_nothing) # get next step after performing the action
    stacked_image = np.stack((image, image, image, image), axis=2).reshape(1, img_cols, img_rows, img_channels) # stack 4 images to create placeholder input reshaped 1*20*40*4 
    
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    timestep = 0
    reward = 0
    while (True): #endless running
        
        loss = 0
        Q_sa = 0
        action_index = 0
        reward += 1
        action = np.zeros([ACTIONS]) # action at t
        
        # Choose an action epsilon greedy
        if  random.random() <= epsilon: 
            # Randomly explore an action
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        else: 
            # Predict the output
            q = model.predict(stacked_image)
            max_Q = np.argmax(q)         # chosing index with maximum q value
            action_index = max_Q 
            action[action_index] = 1        # o=> do nothing, 1=> jump
                
        #We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and timestep > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE 

        #run the selected action and observed next state and reward
        image_2, is_crashed = agent.get_state(action)
        last_time = time.time()
        image_2 = image_2.reshape(1, image_2.shape[0], image_2.shape[1], 1) #1x20x40x1
        stacked_image_2 = np.append(image_2, stacked_image[:, :, :, :3], axis=3) # append the new image to input stack and remove the first one
        
        # store the transition in D
        D.append((stacked_image, action_index, reward, stacked_image_2, is_crashed))
        if len(D) > REPLAY_MEMORY:
            D.popleft() 
        
        #only train if done observing; sample a minibatch to train on
        if timestep > OBSERVE:
            trainBatch(random.sample(D, BATCH)) 
        stacked_image = stacked_image_2 
        timestep += 1
        print("TIMESTEP", timestep, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward,"/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)


agent = GameAgent()
model = buildmodel()
trainNetwork(model, agent)