import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

#agent class handles multiprocessing of our program
#choose action is put in network class because of this
class ActorCritic(nn.Module):
    def __init__(self, input_dims,n_actions, gamma=0.99,name="actor_critic",chkt_dir="models/actor_critic"):
        super(ActorCritic,self).__init__()

        self.model_name = name #cant use self.name as that is reserved by nn.Module
        self.checkpoint_dir = chkt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_ac") #make sure we know what model it is

        self.gamma = gamma
        self.n_actions = n_actions

        #combined input network, but output has 2 values
        self.fc1 = nn.Linear(*input_dims,128)

        #output layers 
        self.pi = nn.Linear(128,n_actions) #actor part -> probabilities for each action
        self.v = nn.Linear(128,1) #value part

        #basic memory of network
        self.rewards = []
        self.actions = []
        self.states = []
    
    def store_mem(self,state,action,reward):
        self.states.append(state)
        self.actions.append(action)        
        self.rewards.append(reward)
    
    def clear_mem(self):
        self.rewards = []
        self.actions = []
        self.states = []
    
    def forward(self,state):
        f1 = F.relu(self.f1(state))

        pi = self.pi(f1)
        v = self.v(f1)

        return pi,v

    #calculate returns from sequence of steps
    # calculation is like: R = V(t3) ..... ->  R = r3 + gamma * r2 + gamma^2 * r1
    def calc_return(self,done):
        states = torch.tensor(self.state, dtype=torch.float)
        _,v = self.forward(states)#dont care about policy output is, just value evaluation of critic

        return_R = v[-1] * (1-int(done))#last element of that list, if episode is done get 0

        batch_return = [] #handle returns at all the other time steps
        for reward in self.rewards[::-1]: #going reverse through memory
            return_R = reward + self.gamma * return_R # sequence of steps of rewards in actor critic calculation
            batch_return.append(return_R)

        batch_return.reverse()#same order as the starting list
        batch_return = torch.tensor(batch_return, dtype=torch.float)
        return batch_return
    
    #calculate the loss function
    def calc_loss(self,done):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions)

        returns = self.calc_return(done)

        #perform the update, pass the states through actor critic to get new values and distribution
        #use the distribution to get the log probs of the actions the agent actually took, and use those qualities for loss

        pi, values = self.forward(states)
        values = values.squeeze() #convert into shape we need, e.g. 5x5 into 5 elements as output of neural network

        critic_loss = (returns-values)**2 #simple loss

        probs = torch.softmax(pi, dim=1) # get the softmax activation of the output, guarantee every action has finite value and the probabilities add up to 1

        dist = Categorical(probs) # get a categorical distribution TODO: go over this
        log_probs = dist.log_prob(actions) # calculate the log probabilities of the actions actually taken

        actor_loss = -log_probs * (returns - values)

        #note: can also add an entropy here for the total loss
        total_loss = (critic_loss + actor_loss).mean() #sum together for back propagation
        return total_loss
    
    def choose_action(self,observation):

        state = torch.tensor([observation],dtype=torch.float) #add a batch dimension to it for neural network to work on it
        pi, v= self.forward(state)
        probs = torch.softmax(pi,dim=1)

        dist = Categorical(probs)
        action = dist.sample().numpy()[0] #numpy quantity, take the 0th element

        return action