import gym
import torch 
import torch.multiprocessing as multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical # taking probability from network and map it to distribution for us
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

#shared adam class -> share single optimiser among all agents
#iterate over parameters in our parameter groups, then share those parameters amongst the different pools

class SharedAdam(torch.optim.Adam):
    def __init__(self,params, lr=1e-3, betas=(0.9,0.99), eps=1e-8,weight_decay= 0):
        super(SharedAdam,self).__init__(params,lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                #share memory for gradient descent
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

#agent class handles multiprocessing of our program
#choose action is put in network class because of this
class ActorCritic(nn.Module):
    def __init__(self, input_dims,n_actions, gamma=0.99):
        super(ActorCritic,self).__init__()

        self.gamma = gamma

        #have 2 separate inputs
        #2 distinct networks as multiprocessing doesnt work when shared for neural networks
        self.pi1 = nn.Linear(*input_dims,128)
        self.v1 = nn.Linear(*input_dims,128)

        #output layers
        self.pi = nn.Linear(128,n_actions)
        self.v = nn.Linear(128,1)

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
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

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
    

# handle multiprocessing
class Worker(multiprocessing.Process):
    # global actor critic handles functionality to keep track of all the learning we do
    # name keeps track of each of the workers
    # global_ep_index tracks the global number of episodes done by agents
    def __init__(self, global_actor_critic, optimiser, input_dims,n_actions,lr,name
                 ,global_ep_index, env_id, gamma=0.99):
        
        super(Worker,self).__init__()
        self.local_actor_critic = ActorCritic(input_dims=input_dims,n_actions=n_actions,gamma=gamma)
        self.global_actor_critic = global_actor_critic #can update its parameters

        self.name = 'Worker%02i' % name
        self.ep_index = global_ep_index
        self.env = gym_super_mario_bros.make(env_id)
        self.optimiser = optimiser

    #multiprocessing function - handles the main loop functionality
    #local agent chooses actions, perform gradient descent optimiser update using the gradients calculated by local agent
    #upload those to the global network. So every agent uploads to the global network, and then download that from the global
    #network each time to make the local agent "up to date" and better over time too
    #so locals update the global, then the global updates the local
    def run(self):
        t_step = 1
        while self.ep_index.value < NUM_EPISODES: # .value is a dereference
            done = False
            observation = self.env.reset()
            score = 0 #reward received for episode
            self.local_actor_critic.clear_mem() # clear agent memory at top of every episode
            #play sequence of steps within episode
            while not done:
                #each worker gets their own actor critic, synchronise them to the global network, but we never actually
                #use the global network e.g. to choose action
                action = self.local_actor_critic.choose_action(observation=observation)
                new_observation, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.store_mem(state=new_observation,action=action,reward=reward)

                # if the number of steps % max_num_steps is 0 -> do learning. Or if we finished
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimiser.zero_grad()
                    loss.backward()

                    for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad # set the parameter of the global gradient to the local agents gradient
                    
                    self.optimiser.step()

                    #synchronise networks now
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict()) #load the state dictionary of the global to local

                    self.local_actor_critic.clear_mem() #clear memory after each learning step
                t_step += 1

                observation = new_observation
            
            #end of an episode - may have finished an episode from another agent while one thread was running
            # make sure no other thread is trying to access that variable
            with self.ep_index.get_lock():
                self.ep_index.value += 1 # this is a shared value, so lock its modification
            
            print(self.name,"episode ", self.ep_index.value, "reward %.1f" % score) #reward over total score of episode


if __name__ == "__main__":
    lr = 1e-4
    env_id = ""
    n_actions = 3
    input_dims = []
    N_GAMES = 10000
    T_MAX = 5 # comes from the paper
    global_actor_critic = ActorCritic(input_dims=input_dims, n_actions=n_actions)
    global_actor_critic.share_memory() #share its memory
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92,0.999))

    global_episode = multiprocessing.Value('i',0) #signed integer type, can be negative or positive

    # create a list of workers

    workers = [Worker(global_actor_critic,
                      optim,
                      input_dims,
                      n_actions,
                      lr,
                      env_id,gamma=0.99,
                      name=i,
                      global_ep_index=global_episode,
                      env_id=env_id) for i in range(multiprocessing.cpu_count())] #handle starting and running of threads

    [worker.start() for worker in workers]

    [worker.join() for worker in workers] # ///


