import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

# Note: PPO is meant to be run primarily on the CPU, so dont put this on gpu
class MarioNet(nn.Module):
    def __init__(self,envs,input_shape):
        super(MarioNet,self).__init__()
        
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0],32,8,stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32,64,4,stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64,64,3,stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            #linear layer that takes input of the flattened features
            layer_init(nn.Linear(64*7*7, 512)), # get reduced into a 7x7 image with 64 channels 
            # TODO: rewrite this to be better and programatical e.g. get_flat_size() function 
            nn.ReLU(),
        )
        # TODO: may need more layers here, and more than 512
        self.actor = layer_init(nn.Linear(512, envs.single_action.space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512,1), std=1)
        """
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
        """
    
    def get_value(self,x):
        #divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1
        return self.critic(self.cnn(x / 255.0)) #go through cnn first then critic

    def get_action_plus_value(self,x,action=None):  
        #divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1
        hidden = self.cnn(x / 255.0) #get the hidden layer output, after CNN input
        logits = self.actor(hidden) #unnormalised action probabilities
        probabilities = Categorical(logits=logits) #softmax operation to get the action probability distribution we need
        if action is None:
            action = probabilities.sample
        #return actions, log probabilities, entropies and values from critic
        return action,probabilities.log_prob(action), probabilities.entropy(),self.critic(hidden)

# TODO: go over this 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0): #use sqrt 2 as standard deviation
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer