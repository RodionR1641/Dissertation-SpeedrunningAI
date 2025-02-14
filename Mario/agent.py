import random
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
from plot import LivePlot
import numpy as np
import time
import os
import logging
import datetime

log_dir = "/cs/home/psyrr4/Code/Code/logs"
os.makedirs(log_dir, exist_ok=True)

# Define log file name (per process)
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
log_file = os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_rank{rank}.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

print("starting logging")
logging.info(f"Process {rank} started training on GPUs")

if torch.cuda.is_available():
    try:
        logging.info(torch.cuda.current_device())
        logging.info("GPU Name: " + torch.cuda.get_device_name(0))
        logging.info("PyTorch Version: " + torch.__version__)
        logging.info("CUDA Available: " + str(torch.cuda.is_available()))
        logging.info("CUDA Version: " + str(torch.version.cuda))
        logging.info("Number of GPUs: " + str(torch.cuda.device_count()))
    except RuntimeError as e:
        logging.info(f"{e}")
else:
	logging.info("cuda not available")

#agent's memory
# The way deep q learning works: 
# build a buffer over time of all of the state-action pairs it played
# pick up the state,the action, the next state and the reward of every play
# then use that to Train the model to approximate the Q value

# so a way for the agent to remember X number of games and then replay it back for training
class ReplayMemory:
    
    def __init__(self, capacity, device="cpu"):
        self.capacity = capacity #memory capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    #TODO: can implement a Prioritised Sampling -> sample more important experiences to learn from. Given a probability score
    # transitions where agent made a lot of errors are important, e.g. high temporal difference. These experiences indicate stuff
    # the model struggles with, so we can learn faster and better


    #need to make sure memory doesnt go over a certain size
    #transition is the data unit at play, tuple of state,action,reward,next state,done. It is an experience of the game at certain time
    def insert(self, transition):
        transition = [item.to("cpu") for item in transition] #this replay memory can get large, can run out of GPU memory quickly
        #so store everything about replay memory in CPU,pushes it into computers main RAM
        #when we use "sample" method, we can push back to device(e.g. gpu)

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            #keep everything under capacity(e.g. a million). Ensure we keep most recent experiences
            self.memory.remove(self.memory[0]) # like a queue
            self.memory.append(transition)
    
    # sample the memory now, using a batch size 
    def sample(self,batch_size=32):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory,batch_size) # take a random sample of transitions
        batch = zip(*batch) #basically make columns out of rows. We get transitions as tuples of (state,action .....)
        #so we then make lists of columns of state,action ...
        return [torch.cat(items).to(self.device) for items in batch] # convert each list of components
       #into a tensor using torch.cat(), and back to device now
       #when batching, these will have size 32 etc

    # see if we can sample from memory given a batch size, so have enough memory to sample
    def can_sample(self,batch_size):
        #need enough varied data to sample, as we sample random data
        return len(self.memory) >= batch_size * 10

    #why do we need a len object? -> we need to make sure we get the number of items in memory rather than some object output
    #other objects can call the len function of this class, but get what we actually want which is len(self.memory) rather than e.g. len(self)
    def __len__(self):
        return len(self.memory) #get number of transitions stored in buffer

#plays the game
#covers a lot of training
class Agent:
    #model we train on passed
    #epsilon -> frequency with which we select a random action. When we start training, want to select certain number of random actions to start off with.
    #min_epsilon -> threshold for how low epsilon can be, 10% of the time random so model tries new things i.e. exploration
    #nb_warmup -> number of warmup steps, period where epsilon decays
    #nb_actions -> number of actions
    #memory_capacity -> for ReplayMemory
    #batch_size
    #learning_rate -> how big of a step we want the agent to take at a time, how quickly we want it to learn. If its too high, jump erradically from solution to solution
    #rather than slowly building to a right solution. Want it to be high enough to pick up changes though, but too high it wont learn well
    def __init__(self,model,device="cpu",epsilon=1.0,min_epsilon=0.1,nb_warmup=10000,nb_actions=None,memory_capacity=10000,
                 batch_size=32,learning_rate=0.00025):
        
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model #policy network to predict actions
        #this is our anchor to work towards, self.model will work towards this target_model. Helps to stabilise
        # 
        # this is a copy of initial self.model, calculate Q values using it
        # why eval() -> set the model to Evaluation Mode to disable dropout as its used for predictions, not training. 
        # Makes it deterministic and consistent for validation 
        #TODO: read through how dueling deep q learning works
        self.target_model = copy.deepcopy(model).eval() #when we work with q learning, want a model and another model we can evaluate of off. Part of Dueling deep Q
        
        self.epsilon = epsilon #this is probability of selecting random action basically. Expoloration vs exploitation
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1- (((epsilon - min_epsilon) / nb_warmup) *2) # linear decay rate, close to the nb_warmup steps count
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        
        self.gamma = 0.99 #how much we discount future rewards compared to immediate rewards. 0.99 heavily considers long term rewards
        #if it was 0.5 it would consider short term rewards more. High one helps with games as delayed rewards are more important,but
        #can lead to instability
        self.nb_actions = nb_actions
        #this updates the parameters of model during training. Combines adaptive learning rates with weight decay regularisation for 
        #better generalisation
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)# TODO: go over Adam

        logging.info(f"starting, epsilon={self.epsilon},epsilon_decay={self.epsilon_decay}")

    #state is image of our environment
    def get_action(self,state,test=False):

        #only use random action if its training
        if (torch.rand(1) < self.epsilon) and not test: #if a random number between 0 and 1 is smaller than epsilon, do random move
            #randint returns a tensor
            return torch.randint(self.nb_actions, (1,1)) #random action. adding 1,1 for tensors is for e.g. batch size etc
        else:
            action_value = self.model(state).detach() #get all the action probabilities
            # model returns a list of probabilities e.g. [0.11,0.22,0.45,0.3]
            return torch.argmax(action_value,dim=1,keepdim=True) # argmax grabs highest value. This will return the action of index 2

    #epochs = how many iterations to train for
    def train(self,env, epochs):
        #see how the model is doing over time
        stats = {"Returns": [], "AvgReturns": [], "Epsilon": []} #store as dictinary of lists

        plotter = LivePlot()

        for epoch in range(1,epochs+1):
            state = env.reset() #reset the environment for each iteration
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state,reward,done,info = env.step(action)

                self.memory.insert([state, action, reward, done, next_state])#record both the previous and next_state

                #actual training part
                #can take out of memory only if sufficient size
                if self.memory.can_sample(self.batch_size):
                    #all batches data, series of batches of all of the states,actions,rewards etc
                    state_b,action_b,reward_b,done_b,next_state_b = self.memory.sample(self.batch_size) #series of batches, unpack them
                    #can pass a batch of e.g. size32, speeds up training to do it in batches

                    #now get into q states
                    qsa_b = self.model(state_b).gather(1,action_b)#gather the action chosen in state_b by our network
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0] #getting the predictions of the target model on what the appropriate Action is for the next state
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b#~negates the value of done. If done is true, there is no next state value
                    #self.gamma=how much we discount the next step
                    loss = F.mse_loss(qsa_b, target_b) #difference between prediction(qsa_b) and the value based on the actual rewards above
                    self.model.zero_grad() #zeroes out the gradients, can go through and update them
                    loss.backward() # backpropagation
                    self.optimizer.step() # optimiser

                state = next_state #did the training, now move on with next state
                ep_return += reward.item()

            stats["Returns"].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay #if greater than min_epsilon, multiply by epsilon decay. Update the epsilon in each Epoch

            #gatherin stats
            if epoch % 10 == 0:
                self.model.save_model() #save model every 10th epoch

                average_returns = np.mean(stats["Returns"][-100:]) #average of the last 100 returns

                #graph can turn too big if we try to plot everything through. Only update a graph data point for every 10 epochs

                stats["AvgReturns"].append(average_returns)
                stats["Epsilon"].append(self.epsilon) #see where the epsilon was at. Do we see higher returns with high epsilon, or only when it dropped etc

                if(len(stats["Returns"]) > 100):
                    logging.info(f"Epoch: {epoch} - Average return: {np.mean(stats['Returns'][-100:])}  - Epsilon: {self.epsilon} ")
                else:
                    #for the first 100 iterations, just return the episode return,otherwise return the average like above
                    logging.info(f"Epoch: {epoch} - Episode return: {np.mean(stats['Returns'][-1:])}  - Epsilon: {self.epsilon} ")

            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict()) #keep the target_model lined up with main model, its learning in hops
                plotter.update_plot(stats)
            
            if epoch % 1000 == 0:
                self.model.save_model(f"models/model_iter_{epoch}.pt") #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed
            
        return stats
    

    #run something on the machine, and see how we perform
    def test(self, env):
        
        #just see how game performs for 3 trials
        for epoch in range(1,3):
            state = env.reset()

            done = False

            #1000 steps
            for _ in range(1000):
                time.sleep(0.01) #by default it runs very quickly, so slow down
                action = self.get_action(state,test=True)
                state,reward,done,info = env.step(action) #make the environment step through the game
                if done:
                    break
    
