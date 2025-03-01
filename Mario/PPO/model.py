import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

class MarioNet(nn.Module):
    def __init__(self,envs):
        super(MarioNet,self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)), #input shape to first layer is product of obesrvation space
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1) , std=1.), #output linear layer uses 1 as a standard deviation
        )
        # actor here . std=0.01 so that layer parameters have similar values so probabilty of taking each action is similar
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space.n) , std=0.01),
        )
    
    def get_value(self,x):
        return self.critic(x)

    def get_action_plus_value(self,x,action=None):
        logits = self.actor(x) #unnormalised action probabilities
        probabilities = Categorical(logits=logits) #softmax operation to get the action probability distribution we need
        if action is None:
            action = probabilities.sample
        #return actions, log probabilities, entropies and values from critic
        return action,probabilities.log_prob(action), probabilities.entropy(),self.critic(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0): #use sqrt 2 as standard deviation
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer