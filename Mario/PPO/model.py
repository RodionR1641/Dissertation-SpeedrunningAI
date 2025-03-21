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
            layer_init(nn.Linear(64*7*7, 1024)), # get reduced into a 7x7 image with 64 channels 
            nn.ReLU(),
        )
        #self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        #self.critic = layer_init(nn.Linear(512,1), std=1)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(1024,1024)), #input shape to first layer is product of obesrvation space
            nn.ReLU(),
            layer_init(nn.Linear(1024,1) , std=1.0), #output linear layer uses 1 as a standard deviation
        )
        # actor here . std=0.01 so that layer parameters have similar values so probabilty of taking each action is similar
        self.actor = nn.Sequential(
            layer_init(nn.Linear(1024,1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024,envs.single_action_space.n) , std=0.01),
        )
        self.device = device
        self.to(device)
        
    
    def get_value(self,x):
        if x.device != self.device:
            x.to(self.device)
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
    
    #these models take a while to train, want to save it and reload on start
    def save_model(self, weights_filename="models/latest_ppo.pt"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done
        print("...saving checkpoint...")
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(self.state_dict(),weights_filename)
    
    def load_model(self, weights_filename="models/latest_ppo.pt",device="cpu"):
        try:
            self.load_state_dict(torch.load(weights_filename,map_location=device,weights_only=True))
            print(f"Loaded weights filename: {weights_filename}")            
        except Exception as e:
            print(f"No weights filename: {weights_filename}")
            print(f"Error: {e}")

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