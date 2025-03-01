import collections
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import numpy as np
import cv2
from PIL import Image
import torch
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordEpisodeStatistics, RecordVideo

#handles the environment, pre-processing using wrappers and
#overriding the default step and reset methods
class Mario(gym.Wrapper):
    #rgb_array gives pixel info of game for us to work with
    # human mode actually allows to see
    def __init__(self,repeat=4,device='cpu',env_id="SuperMarioBros-1-1-v0"):
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, RIGHT_ONLY)

        #apply wrappers for preprocessing of images
        env = ResizeObservation(env,shape=84)
        env = GrayScaleObservation(env)
        env = FrameStack(env,num_stack=4,lz4_compress=True)
        env = RecordEpisodeStatistics(env) #record statistics of episode return
        #env = RecordVideo(env,"videos", record_video_trigger=lambda t: t % 100 == 0) # record video of agent playing

        super(Mario,self).__init__(env)#parent class initialiser, gym wrapper

        self.action_num = len(RIGHT_ONLY)
        self.env = env
        self.repeat = repeat
        self.lives = 3
        self.device = device

    #take action on an environment ->returns a state 
    def step(self,action):
        total_reward = 0.0
        done = False
        
        #repeat same action for self.repeat steps as have high frame rate
        for _ in range(self.repeat):
            state,reward,done, info = self.env.step(action) # the reward function e.g. for mario, is defined within that 
            #environment. In breakout, breaking a brick gives a reward
            total_reward += reward 
            if done:
                break

        #why return these?
        #total_reward: feedback for learnng
        #done: indicate if game finished so episode can end
        return state,total_reward,done,info

    #2 functions we use the most -> reset and step
    #reset is to bring it back to setup state
    #overriding it here compared to default
    def reset(self):
        state = self.env.reset()

        self.lives = 3
        #initial state

        return state