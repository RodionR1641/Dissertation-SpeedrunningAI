import collections
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import numpy as np
import cv2
from PIL import Image
import torch
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

#handles the environment, pre-processing using wrappers and
#overriding the default step and reset methods
class DQN_Mario(gym.Wrapper):
    #rgb_array gives pixel info of game for us to work with
    # human mode actually allows to see
    def __init__(self,repeat=4,device='cpu'):
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        env = JoypadSpace(env, RIGHT_ONLY)

        #apply wrappers for preprocessing of images
        env = ResizeObservation(env,shape=84)
        env = GrayScaleObservation(env)
        env = FrameStack(env,num_stack=4,lz4_compress=True)

        super(DQN_Mario,self).__init__(env)#parent class initialiser, gym wrapper

        self.action_num = len(RIGHT_ONLY)
        self.env = env
        self.repeat = repeat
        self.lives = 3
        self.device = device

    #take action on an environment ->returns a state 
    def step(self,action):
        total_reward = 0.0
        done = False
        
        for _ in range(self.repeat):
            state,reward,done, info = self.env.step(action) # the reward function e.g. for breakout, is defined within that 
            #environment. In breakout, breaking a brick gives a reward
            total_reward += reward 
            current_lives = info['life']

            if current_lives < self.lives:
                #can experiment with this number
                total_reward = total_reward - 1 # can be any number. We want to have losing live to have same impact though. 
                self.lives = current_lives

            #print(f"lives: {self.lives}, Total reward: {total_reward}")
            if done:
                break
        
        #converting total_reward and done into tensors now. (1,-1) tensor is a single row tensor
        #why? -> neural networks operate on Tensors, not scalars. Standardising the reward shape etc is useful for batching.
        #a single tensor may have size (1,1), in a batch these tensors can then have shape (batch_size,1)
        #total_reward is float
        
        """
        total_reward = torch.tensor(total_reward).view(1,-1).float() # making sure data is standardasised, can be then used in batches
        total_reward = total_reward.to(self.device) # send this to cpu/gpu for processing

        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)
        """

        #why return these?
        #total_reward: feedback for learnng
        #done: indicate if game finished so episode can end
        return state,total_reward,done,info
    
    # apply wrappers to the environment
    def apply_wrappers(self):

        env = ResizeObservation(self.env, shape=84) #resize image for efficiency
        env = GrayScaleObservation(env) #grayscale so only one color channel
        env = FrameStack(env,num_stack=4,lz4_compress=True) #capture motion, make each state as 4 consequtive frames
        return env

    #2 functions we use the most -> reset and step
    #reset is to bring it back to setup state
    #overriding it here compared to default
    def reset(self):
        state = self.env.reset()

        self.lives = 3
        #initial state

        return state