import collections
import gym
import numpy as np
import cv2
from PIL import Image
import torch

#Responsible for setting up the environment, pre-processing state(images) and
#overriding the default step and reset methods to be specific for this game
#good to have this class separate as a lot of code can be reused and just change this class for a different environment
#contains the environment within itself, but is a Wrapper so has more stuff
class DQNBreakout(gym.Wrapper):
    #rgb_array gives pixel info of game for us to work with
    # human mode actually allows to see
    def __init__(self, render_mode='rgb_array',repeat=4,device='cpu'):
        env = gym.make("BreakoutNoFrameskip-v4",render_mode=render_mode)

        super(DQNBreakout,self).__init__(env)#parent class initialiser, gym wrapper

        self.repeat = repeat
        self.lives = env.ale.lives()#need to train our agent to learn that losing a life in game is bad
        self.frame_buffer = [] #store seen frames of the game
        self.device = device
        self.image_shape = (84,84) #downsize images for processing

    #take action on an environment ->returns a state 
    def step(self,action):
        total_reward = 0
        done = False

        #dont want agent to think about step every frame, dont need to react every single frame
        # take same action 4 frames in a row, what the frame means basically
        # take max of the last 2 frames
        for i in range(self.repeat):
            observation,reward,done, info = self.env.step(action) # the reward function e.g. for breakout, is defined within that 
            #environment. In breakout, breaking a brick gives a reward

            total_reward += reward #add to total_reward from cycle. necessary because of our repeating action
            #caption the number of lives
            #print(info,total_reward)

            current_lives = info['lives']

            #start of with 5 lives
            if current_lives < self.lives:
                #can experiment with this number
                total_reward = total_reward - 1 # can be any number. We want to have losing live to have same impact though. 
                #positive impact for scoring, same amount of negative impact if agent lost a live compared to getting a reward
                self.lives = current_lives

            #print(f"lives: {self.lives}, Total reward: {total_reward}")
            
            self.frame_buffer.append(observation) #need to store frames

            if done:
                break
        
        #take the frame with most pixels as some images can get blurry and faded, so want better frames basically
        max_frame = np.max(self.frame_buffer[-2:],axis=0) # grab last 2 frames, take max. No need to process more than 2, 
        #can store this stuff in a buffer

        #now process max_frame
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device) #same device as the model uses

        #converting total_reward and done into tensors now. (1,-1) tensor is a single row tensor
        #why? -> neural networks operate on Tensors, not scalars. Standardising the reward shape etc is useful for batching.
        #a single tensor may have size (1,1), in a batch these tensors can then have shape (batch_size,1)
        #total_reward is float
        total_reward = torch.tensor(total_reward).view(1,-1).float() # making sure data is standardasised, can be then used in batches
        total_reward = total_reward.to(self.device) # send this to cpu/gpu for processing

        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)

        #why return these?
        #max_frame: essential to train on, it is basically the next state after taking this action
        #total_reward: feedback for learning
        #done: indicate if game finished so episode can end
        return max_frame,total_reward,done,info
    
    #observation is image
    # take images of various types and sizes -> standardise
    # also reduce Complexity of what the network needs to learn on. Shrink it
    # render it grayscale too, and divide by 255(get a range from 0 to 1), kind of like normalising values
    def process_observation(self,observation):

        img = Image.fromarray(observation)#represent an image from this array of observation, convert into Image from numpy for manipulation
        img = img.resize(self.image_shape) #fixed size input image for models. Better for computation and makes size standardised
        #dont care much about resolution in simple games
        img = img.convert("L") #grayscale - collapse 3 color channels into 1 for simplicity, color here is not crucial
        img = np.array(img)# back to numpy for more manipulation
        img = torch.from_numpy(img) #tensors and numpy arrays are similar but tensors more efficient for ML on cpus/gpus for
        #backtracking as they track gradients in gradient descent
        
        #unsqueeze the image, later we pass a batch size number, so we need something in the
        #batch size column and image channel column
        #note: here the batch size is 1 as when observing an observation, only one can be observed at a time
        #later we use e.g. 32 batch size, when we sample from memory the batch size is 32 automatically
        img = img.unsqueeze(0)
        #dimensions expected of input: Batch size,Channels,height,weight. So we add 1 for both batch size and channel(grayscale)
        img = img.unsqueeze(0)

        #divide by 255, so range is 0-1 for normalisation for training
        img = img / 255.0

        img = img.to(self.device)
        return img #return processed tensor

    #2 functions we use the most -> reset and step
    #reset is to bring it back to setup state
    #overriding it here compared to default
    def reset(self):
        self.frame_buffer = []#clear the buffer

        observation = self.env.reset()

        self.lives = self.env.ale.lives()
        #initial state
        observation = self.process_observation(observation)

        return observation