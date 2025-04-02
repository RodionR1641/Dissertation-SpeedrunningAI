import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import random
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordEpisodeStatistics

#handles the environment, pre-processing using wrappers and
#overriding the default step and reset methods
class DQN_Mario(gym.Wrapper):
    #rgb_array gives pixel info of game for us to work with
    # human mode actually allows to see
    def __init__(self,use_vit=False,seed=None,rnd=False):
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        SIMPLE_MOVEMENT.append(['down']) #there is a skip on some levels mario can make with a down action
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        self.random_gen = random.Random()
        #aply wrappers for statistics
        env = RecordEpisodeStatistics(env) #record statistics of episode return

        #sticky actions were useful for RND in papers shown for PPO, maybe here too
        if rnd:
            env = StickyActionEnv(env)
        #take random number of NOOPs on reset. overwrite reset function, sample random number of noops between 0 and 30, execute the noop and then return
        # add stochasticity to environment
        env = NoopResetEnv(env=env,noop_max=30,rng_gen=self.random_gen)

        env = CustomReward(env=env)
        # skip 4 frames by default, repeat agents actions for those frames.
        # Done for efficiency. Take the max pixel values over last 2 frames
        
        env = MaxAndSkipEnv(env=env,skip=4)

        # wrapper treats every end of life as end of that episode. 
        # So, if any life is lost episode ends. But reset is called only if lives are exhausted
        
        env = EpisodicLifeEnv(env=env)

        #apply wrappers for preprocessing of images
        if(use_vit):
            env = ResizeObservation(env,shape=256)
        else:
            env = ResizeObservation(env,shape=84)
        env = GrayScaleObservation(env)
        env = FrameStack(env,num_stack=4,lz4_compress=True)

        super(DQN_Mario,self).__init__(env)#parent class initialiser, gym wrapper

        self.action_num = len(SIMPLE_MOVEMENT)
        self.env = env


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, rng_gen, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max #maximum number of no ops we can do at the start
        self.override_num_noops = None
        self.noop_action = 0 #this is just the action number
        self.rng_gen = rng_gen
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]
    # everytime reset the env - perform these NOOPs
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
        max_frame = self._obs_buffer.max(axis=0) #get the max frame, some frames may have a glitch so choose ones that are 
        # more reliable

        return max_frame, total_reward, done, info #state,total reward, done ,info
    

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    helps value estimation.
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
            # sometimes we stay in lives == 0 condition for a few frames
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
            obs, _, done, _ = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if done:
                obs = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped._life  # type: ignore[attr-defined]
        return obs
    
# a small probability to repeat the same action as before
class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()

# add a custom reward for getting some score and also for getting the flag(completing the level)
class CustomReward(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.curr_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info["score"] - self.curr_score) / 40.0 #score can help the agent move towards right directions and 
        #explore, also finishing the level gives a score so more incentive to finish level and in shorter time
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50#this is on top of the death penalty

        #scale the reward to be not too big
        return state, reward / 10.0, done, info

    def reset(self,**kwargs):
        self.curr_score = 0

        obs = self.env.reset(**kwargs)
        return obs