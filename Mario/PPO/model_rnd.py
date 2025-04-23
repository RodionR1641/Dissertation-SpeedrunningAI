import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

# Note: PPO is meant to be run primarily on the CPU, so dont put this on gpu
class MarioNet(nn.Module):
    def __init__(self,envs,input_shape,device="cpu"):
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
            nn.ReLU(),
        )
        
        
        self.critic = layer_init(nn.Linear(512,1),std=1.0)
        self.actor = layer_init(nn.Linear(512,envs.single_action_space.n),std=0.01)
        
        self.device = device
        self.to(device)
    
    def get_value(self,x):
        if x.device != self.device:
            x = x.to(self.device)
        #divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1
        return self.critic(self.cnn(x / 255.0)) #go through cnn first then critic

    def get_action_plus_value(self,x,action=None):  
        if x.device != self.device:
            x.to(self.device)
        #divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1
        hidden = self.cnn(x / 255.0) #get the hidden layer output, after CNN input
        logits = self.actor(hidden) #unnormalised action probabilities
        probabilities = Categorical(logits=logits) #softmax operation to get the action probability distribution we need
        if action is None:
            action = probabilities.sample()
        #return actions, log probabilities, entropies and values from critic
        return action,probabilities.log_prob(action), probabilities.entropy(),self.critic(hidden)
    

# Layer initialisation
# PPO uses orthogonal initialisation on layers weight and constant initialisation on bias
# std is sqrt 2 for most layers except the output layers where critic uses 1 as std and actor uses 0.01 to make sure the 
# layer parameters have similar scalar values and probability of takin each action is similar
# orthogonal - makes sure that the weight matrix of a layer is orthogonal(column vectors are orthogonal - dot product of them is 0)
# this is to preserve the norm of the input, helps with stability and prevents vanishing gradients
# constant - makes sure bias is fixed value. Initialised to 0 to ensure initial outputs of layer are
# not biased towards anything, neutral state of network 

# why sqrt(2) -> ensure the variance of activations remains approximately the same across layers, stability
# critic output -> std = 1 ,not too large or too small
# actor -> smaller std = 0.01, nearly uniform probabilities. Start exploring all actions
def layer_init(layer, std=np.sqrt(2), bias_const=0.0): #use sqrt 2 as standard deviation
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer


#calculates the intrinsic reward: make a prediction on a given state and see 
# how well the learned model matches a random target
class RND_model(nn.Module):
    def __init__(self,
                input_shape
                ,device="cpu"
                ):
        super(RND_model,self).__init__()

        self.relu = nn.ReLU()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=(8,8),stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()        
        flat_size = get_flat_size(input_shape,self.feature_layer)
        
        self.fc1 = nn.Linear(flat_size,512)
        self.out_layer = nn.Linear(512,512)
        self.device = device
        self.to(device)

    def forward(self,x):
        if x.device != self.device:
            x.to(self.device)

        x = x/255.0 #normalise to be between 0 and 1
        x.to(self.device)
        x = self.feature_layer(x)
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.out_layer(x)
        return x
    
def get_flat_size(input_shape,feature_layer):
#pass dummy input through conv layers to get flatten size dynamically

    with torch.no_grad():#no gradient computation, just a dummy pass
        dummy_input = torch.zeros(1,*input_shape)
        x = feature_layer(dummy_input)
        flatten = nn.Flatten() #need instance of this to calculate shape
        flattened_x = flatten(x)
        return flattened_x.shape[1]