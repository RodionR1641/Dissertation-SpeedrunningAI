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
    def __init__(self,use_vit=False,repeat=4,device='cpu'):
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        env = JoypadSpace(env, RIGHT_ONLY)

        #apply wrappers for preprocessing of images
        if(use_vit):
            env = ResizeObservation(env,shape=256)
        else:
            env = ResizeObservation(env,shape=84)
        env = GrayScaleObservation(env)
        env = FrameStack(env,num_stack=4,lz4_compress=True)

        super(DQN_Mario,self).__init__(env)#parent class initialiser, gym wrapper

        self.action_num = len(RIGHT_ONLY)
        self.env = env
        self.repeat = repeat
        self.lives = 3

    #take action on an environment ->returns a state 
    def step(self,action):
        total_reward = 0.0
        done = False
        
        for _ in range(self.repeat):
            state,reward,done, info = self.env.step(action) # the reward function e.g. for breakout, is defined within that 
            #environment. In breakout, breaking a brick gives a reward
            total_reward += reward 

            #already handles losing life by -15 reward
            """
            if current_lives < self.lives:
                #can experiment with this number
                total_reward = total_reward - 1 # can be any number. We want to have losing live to have same impact though. 
                self.lives = current_lives
            """

            #print(f"lives: {self.lives}, Total reward: {total_reward}")
            if done:
                break
        
        # since the same action is repeated for x steps, can delete by self.repeat to "normalise" it
        #converting total_reward and done into tensors now. (1,-1) tensor is a single row tensor
        #why? -> neural networks operate on Tensors, not scalars. Standardising the reward shape etc is useful for batching.
        #a single tensor may have size (1,1), in a batch these tensors can then have shape (batch_size,1)
        #total_reward is float

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
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    

class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info #state,total reward, done ,info
    

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal so done becomes true,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, done, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if done:
                obs = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped._life  # type: ignore[attr-defined]
        return obs