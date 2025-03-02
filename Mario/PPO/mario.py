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
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv

#handles the environment, pre-processing using wrappers and
#overriding the default step and reset methods
class Mario(gym.Wrapper):
    #rgb_array gives pixel info of game for us to work with
    # human mode actually allows to see
    def __init__(self,device='cpu',env_id="SuperMarioBros-1-1-v0"):
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, RIGHT_ONLY)

        #aaply wrappers for statistics
        env = RecordEpisodeStatistics(env) #record statistics of episode return
        #env = RecordVideo(env,"videos", record_video_trigger=lambda t: t % 100 == 0) # record video of agent playing
        
        #take random number of NOOPs on reset. overwrite reset function, sample random number of noops between 0 and 30, execute the noop and then return
        # add stochasticity to environment
        env = NoopResetEnv(env=env,noop_max=30)
        # skip 4 frames by default, repeat agents actions for those frames. Done for efficiency. Take the max pixel values over last 2 frames
        env = MaxAndSkipEnv(env=env,skip=4)
        # wrapper treats every end of life as end of that episode. So, if any life is lost episode ends. But reset is called only if lives are exhausted
        env = EpisodicLifeEnv(env=env) # this one might not work as it expects ale, and others also expect terminated as part of "step" function. Can rewrite the wrapper though
        #apply wrappers for preprocessing of images
        env = ResizeObservation(env,(84,84)) # for efficiency
        env = GrayScaleObservation(env)
        env = FrameStack(env,num_stack=4,lz4_compress=True) #stack 4 past observations together as a single observation. Helps agent identify velocities of objects

        super(Mario,self).__init__(env)#parent class initialiser, gym wrapper

        self.action_num = len(RIGHT_ONLY)
        self.env = env
        self.device = device

    #2 functions we use the most -> reset and step
    #reset is to bring it back to setup state
    #overriding it here compared to default
    def reset(self):
        state = self.env.reset()
        #initial state

        return state