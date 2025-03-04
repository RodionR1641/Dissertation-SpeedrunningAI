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
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv
import random


from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn
from typing import Dict, SupportsFloat

#handles the environment, pre-processing using wrappers and
#overriding the default step and reset methods
class Mario(gym.Wrapper):
    #rgb_array gives pixel info of game for us to work with
    # human mode actually allows to see
    def __init__(self,device='cpu',env_id="SuperMarioBros-1-1-v0",seed=None):
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, RIGHT_ONLY)

        #aaply wrappers for statistics
        env = RecordEpisodeStatistics(env) #record statistics of episode return
        
        #take random number of NOOPs on reset. overwrite reset function, sample random number of noops between 0 and 30, execute the noop and then return
        # add stochasticity to environment
        

        self.seed = seed
        self.random_gen = random.Random(self.seed)

        env = NoopResetEnv(env=env,noop_max=30,rng_gen=self.random_gen)
        # skip 4 frames by default, repeat agents actions for those frames. Done for efficiency. Take the max pixel values over last 2 frames
        
        #env = MaxAndSkipEnv(env=env,skip=4)
        # wrapper treats every end of life as end of that episode. So, if any life is lost episode ends. But reset is called only if lives are exhausted
        
        #env = EpisodicLifeEnv(env=env) # this one might not work as it expects ale, and others also expect terminated as part of "step" function. Can rewrite the wrapper though
        #apply wrappers for preprocessing of images
        env = ResizeObservation(env,(84,84)) # for efficiency
        env = GrayScaleObservation(env)
        env = FrameStack(env,num_stack=4,lz4_compress=True) #stack 4 past observations together as a single observation. Helps agent identify velocities of objects

        super(Mario,self).__init__(env)#parent class initialiser, gym wrapper

        self.action_num = len(RIGHT_ONLY)
        self.env = env
        self.device = device
        self.repeat = 4


    def step(self,action):
        total_reward = 0.0
        done = False
        
        for _ in range(self.repeat):
            state,reward,done, info = self.env.step(action) # the reward function e.g. for breakout, is defined within that 
            #environment. In breakout, breaking a brick gives a reward
            total_reward += reward 

            if done:
                break
        
        return state,total_reward,done,info
    


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, rng_gen, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        self.rng_gen = rng_gen
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.rng_gen.randint(1,self.noop_max+1)
        assert noops > 0
        obs = np.zeros(0)
        info: Dict = {}
        for _ in range(noops):
            obs, _, done, info = self.env.step(self.noop_action)
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs
    
