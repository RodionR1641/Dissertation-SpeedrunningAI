import torch.nn as nn
import torch
from actor_critic import ActorCritic
from torch.distributions import Categorical # taking probability from network and map it to distribution for us
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import torch.optim as optim

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=5):
        self.gamma=gamma
        self.n_actions = n_actions
        self.action = None #keep track of the last action took, used for loss function

        self.actor_critic = ActorCritic(n_actions=n_actions)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    
    def choose_action(self,observation):

        state = torch.tensor([observation],dtype=torch.float) #add a batch dimension to it for neural network to work on it
        pi, v= self.actor_critic(state) #dont need value, just actor actions
        probs = torch.softmax(pi,dim=1) #get the softmax activation, for probability distribution

        dist = Categorical(probs) #feed into categorical distribution
        action = dist.sample() #sample the distribution
        self.action = action

        return action.numpy()[0]#return a numpy version of the action as action is a tensor, but openai gym needs numpy arrays. Also add a batch dimension

    def save_models(self):
        print("... saving models ...")
        self.actor_critic #TODO: save here

    def load_models(self):
        print("... loading models ...")
        ###/////

    #functionality to learn

    def learn(self,state,reward, next_state, done):
        state = torch.tensor([state], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        #calculate gradients here

        state_value, probs = self.actor_critic(state)
        state_value_next, _ = self.actor_critic(next_state)

        #squeeze the 2 params to get rid of batch dimension for calculation of loss, 1 dimensional quantity
        state_value = torch.squeeze(state_value)
        state_value_next = torch.squeeze(state_value_next)

        probs = torch.softmax(probs, dim=1)
        action_probs = Categorical(probs)

        log_probs = action_probs.log_prob(self.action) #do it on the most recent action

        #TD loss
        delta = reward + self.gamma * state_value_next *(1-int(done)) - state_value #if its a terminal state, no returns follow after

        actor_loss = -log_probs*delta

        critic_loss = delta**2

        total_loss = actor_loss + critic_loss

        #then calculate gradient here