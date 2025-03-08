import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        #had 2 separate inputs for our a3c, but in a2c dont need this
        self.inner = nn.Linear(*input_dims, 128)

        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def store_mem(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_mem(self):
        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, state):
        inner = F.relu(self.pi1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_return(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
        _, v = self.forward(states)

        return_R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            return_R = reward + self.gamma * return_R
            batch_return.append(return_R)

        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)
        return batch_return

    def calc_loss(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions)

        returns = self.calc_return(done)

        pi, values = self.forward(states)
        values = values.squeeze()

        critic_loss = (returns - values) ** 2

        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)

        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action

# Main training loop for A2C
def train_a2c(env_id, input_dims, n_actions, lr, gamma, num_episodes, t_max):
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, RIGHT_ONLY)

    actor_critic = ActorCritic(input_dims, n_actions, gamma)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        score = 0
        actor_critic.clear_mem()

        while not done:
            action = actor_critic.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            actor_critic.store_mem(observation, action, reward)

            if len(actor_critic.states) % t_max == 0 or done:
                loss = actor_critic.calc_loss(done)
                optim.zero_grad()
                loss.backward()
                optim.step()
                actor_critic.clear_mem()

            observation = new_observation

        print(f"Episode {episode + 1}, Reward: {score}")

    env.close()

if __name__ == "__main__":
    env_id = "SuperMarioBros-v0"
    input_dims = [1, 84, 84]  # Example input dimensions (adjust based on your environment)
    n_actions = len(RIGHT_ONLY)
    lr = 1e-4
    gamma = 0.99
    num_episodes = 1000
    t_max = 5

    train_a2c(env_id, input_dims, n_actions, lr, gamma, num_episodes, t_max)