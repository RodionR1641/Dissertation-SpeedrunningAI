import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

#agent class handles multiprocessing of our program
#choose action is put in network class because of this
class ActorCritic(nn.Module):
    def __init__(self, input_shape,n_actions,device="cpu"):
        super(ActorCritic,self).__init__()

        self.n_actions = n_actions

        self.relu = nn.ReLU()

        #combined input network, but output has 2 values

        self.conv1 = nn.Conv2d(input_shape[0],32,kernel_size=(8,8),stride=(4,4))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1))

        self.flatten = nn.Flatten()
        flat_size = self.get_flat_size(input_shape)

        #self.actor_pi1 = nn.Linear(flat_size,n_actions)
        #self.actor_pi2 = nn.Linear(1024,1024)
        self.actor_pi3 = nn.Linear(flat_size,n_actions)

        #self.critic_value1 = nn.Linear(flat_size,1024)
        #self.critic_value2 = nn.Linear(1024,1024)
        self.critic_value3 = nn.Linear(flat_size,1)
        
        self.device = device
        self.to(self.device)
    
    
    def forward(self,state):

        state.to(self.device)

        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)

        #actor_pi = self.relu(self.actor_pi1(x))
        #actor_pi = self.relu(self.actor_pi2(actor_pi))
        actor_pi = self.actor_pi3(x)

        #critic_value = self.relu(self.critic_value1(x))
        #critic_value = self.relu(self.critic_value2(critic_value))
        critic_value = self.critic_value3(x)

        return critic_value, actor_pi #return state value and probabilities

    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass
            dummy_input = torch.zeros(1,*input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x).shape[1] #get number of features after flattening
        

    def save_model(self,weights_filename="models/a2c_latest.pt"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done
        print("...saving checkpoint...")
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(self.state_dict(),weights_filename)
    
    def load_model(self, weights_filename="models/a2c_latest.pt",device="cpu"):
        try:
            self.load_state_dict(torch.load(weights_filename,map_location=device,weights_only=True))
            print(f"Loaded weights filename: {weights_filename}")            
        except Exception as e:
            print(f"No weights filename: {weights_filename}")
            print(f"Error: {e}")
    


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