import torch.nn as nn
import torch
from a2c_model import ActorCritic
from torch.distributions import Categorical # taking probability from network and map it to distribution for us
import torch.optim as optim
import time
import numpy as np

class Agent:
    def __init__(self,input_shape,lr_rate=1e-5,device="cpu", gamma=0.99, n_actions=5):
        self.gamma=gamma
        self.n_actions = n_actions
        self.actions = None #keep track of the last action took, used for loss function
        self.lr_rate = lr_rate

        self.actor_critic = ActorCritic(input_shape=input_shape,n_actions=n_actions,device=device)
        self.optimiser = optim.Adam(self.actor_critic.parameters(), lr=self.lr_rate)

    
    def choose_action(self,states):
        
        #state = torch.tensor(observation,dtype=torch.float).unsqueeze(0) #add a batch dimension to it for neural network to work on it
        
        with torch.no_grad():
            pi, v= self.actor_critic(states) #dont need value, just actor actions
        
        probs = torch.softmax(pi,dim=1) #get the softmax activation, for probability distribution

        dist = Categorical(probs) #feed into categorical distribution
        actions = dist.sample() #sample the distribution
        self.actions = actions

        return actions.numpy()#return a numpy version of the action as action is a tensor, but openai gym needs numpy arrays. Also add a batch dimension

    def save_models(self,weights_filename="models/a2c_latest.pt"):
        print("... saving models ...")
        self.actor_critic.save_model(weights_filename=weights_filename)

    def load_models(self,weigts_filename="models/a2c_latest.pt",device="cpu"):
        print("... loading models ...")
        self.actor_critic.load_model(weights_filename=weigts_filename,device=device)

    #functionality to learn

    def learn(self,states,rewards, next_states, dones):
        start_time_tensors = time.time()
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32) #this one isnt fed into NN so dont need it to be [] batch dimension
        dones = torch.as_tensor(dones,dtype=torch.float32)#will do maths on them so convert
        end_time_tensors = time.time() - start_time_tensors
        #print(f"tensors took {end_time_tensors}")
        
        #calculate gradients here

        start_time_model = time.time()
        state_values, probs = self.actor_critic(states) #state_value is what critic returns and probs is what actor returns
        state_value_nexts, _ = self.actor_critic(next_states)
        end_time_model =  time.time() - start_time_model
        #print(f"model took {end_time_model}")

        #squeeze the 2 params to get rid of batch dimension for calculation of loss, 1 dimensional quantity
        
        #state_value = torch.squeeze(state_value)
        #state_value_next = torch.squeeze(state_value_next)

        start_time_optim = time.time()
        probs = torch.softmax(probs, dim=1)
        action_probs = Categorical(probs)

        log_probs = action_probs.log_prob(self.actions) #do it on the most recent action

        #TD loss
        delta = rewards + self.gamma * state_value_nexts *(1-dones) - state_values #if its a terminal state, no returns follow after
        actor_loss = -log_probs*delta
        critic_loss = delta**2
        total_loss = (actor_loss + critic_loss).mean()
        #loss_val = total_loss.item()
        #then calculate gradient here

        self.optimiser.zero_grad()
        total_loss.backward()
        self.optimiser.step()
        end_time_optim =  time.time() - start_time_optim
        #print(f"optim took {end_time_optim}")
        return total_loss.item()