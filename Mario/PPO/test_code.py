import gym

class WrapperA(gym.Wrapper):
    def step(self, action):
        # Do something before calling the original step
        print("Wrapper A: Before step")
        obs, reward, done, info = self.env.step(action)
        # Do something after calling the original step
        print("Wrapper A: After step")
        return obs, reward, done, info

class WrapperB(gym.Wrapper):
    def step(self, action):
        # Do something before calling the original step
        print("Wrapper B: Before step")
        obs, reward, done, info = self.env.step(action)
        # Do something after calling the original step
        print("Wrapper B: After step")
        return obs, reward, done, info

# Create an environment and wrap it
env = gym.make('CartPole-v1')
env = env.reset()
env = WrapperA(env)
env = WrapperB(env)
# Use the environment
obs, reward, done, info = env.step(0)