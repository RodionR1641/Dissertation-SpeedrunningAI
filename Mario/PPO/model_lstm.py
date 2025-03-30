import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

# Note: PPO is meant to be run primarily on the CPU, so dont put this on gpu
class MarioNet(nn.Module):
    def __init__(self,envs,input_shape,device="cpu"):
        super(MarioNet,self).__init__()
        
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0],32,8,stride=4)), #TODO: check that input_shape[0] is 1 for LSTM
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

        self.lstm = nn.LSTM(1024,128)
        for name,param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param,0)
            elif "weight" in name:
                nn.init.orthogonal_(param,1.0)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(128,128)), #input shape to first layer is product of obesrvation space
            nn.ReLU(),
            layer_init(nn.Linear(128,1) , std=1.0), #output linear layer uses 1 as a standard deviation
        )
        # actor here . std=0.01 so that layer parameters have similar values so probabilty of taking each action is similar
        self.actor = nn.Sequential(
            layer_init(nn.Linear(128,128)),
            nn.ReLU(),
            layer_init(nn.Linear(128,envs.single_action_space.n) , std=0.01),
        )
        
        self.device = device
        self.to(device)
    
    def get_states(self,x,lstm_state,done):
        if x.device != self.device:
            x.to(self.device)

        hidden = self.cnn(x / 255.0)#divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1

        # LSTM logic TODO: go over this
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    #just return the critic value
    def get_value(self,x,lstm_state,done):
        if x.device != self.device:
            x.to(self.device)
        #divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1
        hidden, _ = self.get_states(x, lstm_state,done)
        return self.critic(hidden) #go through cnn first then critic

    #returns action, action probability distribution, entropy, critic value and the lstm state
    def get_action_plus_value(self,x,lstm_state,done,action=None):  
        if x.device != self.device:
            x.to(self.device)

        hidden, lstm_state = self.get_states(x,lstm_state,done) #get the hidden layer output, after CNN input
        logits = self.actor(hidden) #unnormalised action probabilities
        probabilities = Categorical(logits=logits) #softmax operation to get the action probability distribution we need
        if action is None:
            action = probabilities.sample()
        #return actions, log probabilities, entropies and values from critic
        return action,probabilities.log_prob(action), probabilities.entropy(),self.critic(hidden), lstm_state

def layer_init(layer, std=np.sqrt(2), bias_const=0.0): #use sqrt 2 as standard deviation
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer