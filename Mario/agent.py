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
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from model import MarioNet
from model_mobile_vit import MarioNet_ViT
from collections import deque

log_dir = "/cs/home/psyrr4/Code/Code/Mario/logs"
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

def print_info():
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

#a similar buffer memory, but it prioritises experiences that have a higher error
class PrioritisedMemory:
    def __init__(self,capacity,device="cpu",alpha=0.6,beta=0.4,beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0 
        self.memory = []
        self.priorities = np.zeros((capacity,) , dtype=np.float32)
        self.max_priorities = 1.0

        self.sum_tree = PrioritisedMemory.SumTree()

    
    class SumTree:
        def __init__(self,capacity):
            self.capacity = capacity
            self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)  # Binary tree structure
            self.data_pointer = 0
        
        #adding priority to a tree
        def add(self,priority):
            tree_idx = self.data_pointer + self.capacity - 1
            self.update(tree_idx,priority)
            self.data_pointer += 1
            if self.data_pointer >= self.capacity:
                self.data_pointer = 0
        
        #update a specific node and propagate the change 
        def update(self, tree_idx, priority):
            change = priority - self.tree[tree_idx]

#using a new buffer because deque is better anyway, can make a non priority version of this
class ReplayBufferPriority:
    def __init__(self,capacity,device="cpu"):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.device = device

    def add(self,transition):
        transition = self.get_to_cpu(transition)#make sure the tensors are stored on cpu
        self.buffer.append(transition)
        self.priorities.append(max(self.priorities,default=1)) #give new experience a maximum prioritity in the buffer so high chance of selection
    
    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale #scale the priorities by ^ priority scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities) #divide each scaled priority by sum of priorities 
        return sample_probabilities

    #apply the equation 1/buf_len * 1/prob as was done in paper
    def get_importance(self,probabilites):
        importance = 1/len(self.buffer) * 1/probabilites
        importance_norm = importance / max(importance) # normalise between 0 and 1 to keep the update step size bounded
        return importance_norm

    def sample(self,batch_size,priority_scale=1.0):
        assert self.can_sample(batch_size)

        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=batch_size, weights=sample_probs) #get the indices of memory based on priority 
        samples = np.array(self.memory)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices]) #get the importance of our probabiliteis that we sampled

        #make sure they are back on device for training

        #get a list of lists which is tranisition = map(list, zip(*samples)). then make sure that every tensor there is on device
        #verify this
        transitions = [[tensor.to(self.device) for tensor in transition] for transition in map(list, zip(*samples))]

        return transitions, importance, sample_indices# get a list of each column and the importance

    # see if we can sample from memory given a batch size, so have enough memory to sample
    def can_sample(self,batch_size):
        #need enough varied data to sample, as we sample random data
        return len(self.memory) >= batch_size * 10
    
    #make sure every item of transition is a tensor and is stored on the cpu as it has more ram
    def get_to_cpu(self,transition):
        #divide by 255 to normalise the image values between 0 and 1
        transition[0] = torch.tensor(np.array(transition[0]), dtype=torch.float32,device="cpu") / 255.0 #state
        transition[1] = torch.tensor(transition[1],device="cpu") #action
        transition[2] = torch.tensor(np.array(transition[3]), dtype=torch.float32,device="cpu") / 255.0#next_state
        transition[3] = torch.tensor(transition[2],device="cpu") #reward
        transition[4] = torch.tensor(transition[4],device="cpu") #done
        return transition

    def set_priorities(self,indices,errors, offset=0.1):
        for i,e in zip(indices,errors): #loop through each index and error
            self.priorities[i] = abs(e) + offset #set the priority to the absolute value of the error. Offset to make sure
            # that priority is never 0
        
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
        self.priority_mem = PrioritisedMemory(self.capacity)

    #need to make sure memory doesnt go over a certain size
    #transition is the data unit at play, tuple of state,action,reward,next state,done. It is an experience of the game at certain time
    def insert(self, transition):
        #this replay memory can get large, can run out of GPU memory quickly so put on cpu
        #divide state and next_state by 255 to normalise around 0 and 1
        transition[0] = torch.tensor(np.array(transition[0]), dtype=torch.float32,device="cpu") / 255.0 #state.
        transition[1] = torch.tensor(transition[1],device="cpu") #action
        transition[2] = torch.tensor(transition[2],device="cpu") #reward
        transition[3] = torch.tensor(np.array(transition[3]), dtype=torch.float32,device="cpu") / 255.0#next_state
        transition[4] = torch.tensor(transition[4],device="cpu") #done
            
        #so store everything about replay memory in CPU,pushes it into computers main RAM
        #when we use "sample" method, we can push back to device(e.g. gpu)

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            #keep everything under capacity(e.g. a million). Ensure we keep most recent experiences
            self.memory.remove(self.memory[0]) # like a queue
            self.memory.append(transition)
    
    # sample the memory now, using a batch size 
    def sample(self,batch_size=32,priority_scale=1.0):
        assert self.can_sample(batch_size)
        
        
        indices = torch.randperm(len(self.memory))[:batch_size] #make sure we have unique indices
        #make sure they are back on device
        batch = [self.memory[i] for i in indices]
        batch_states = torch.stack([b[0] for b in batch]).to(self.device)  # Shape: [batch_size, 4, 84, 84]
        batch_actions = torch.stack([b[1] for b in batch]).to(self.device) # Shape: [batch_size]
        batch_rewards = torch.stack([b[2] for b in batch]).to(self.device)  # Shape: [batch_size]
        batch_next_states = torch.stack([b[3] for b in batch]).to(self.device)  # Shape: [batch_size, 4, 84, 84]
        batch_dones = torch.stack([b[4] for b in batch]).to(self.device)  # Shape: [batch_size]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
        

        sample_probs = self.priority_mem.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.memory)),k=batch_size,weights=sample_probs) #get the indicices of our probablities by priority
        samples = np.array(self.memory)[sample_indices]

        #batch = random.sample(self.memory,batch_size) # take a random sample of transitions
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
    def __init__(self,input_dims,device="cpu",epsilon=1.0,min_epsilon=0.1,nb_warmup=250_000,nb_actions=5,memory_capacity=100_000,
                 batch_size=32,learning_rate=0.00020,gamma=0.95,sync_network_rate=10_000,use_vit=False):
        
        if(use_vit):
            self.model = MarioNet_ViT(nb_actions=nb_actions,device=device) #5 actions for agent can do in this game
        else:
            self.model = MarioNet(input_dims,nb_actions=nb_actions,device=device)
        self.target_model = copy.deepcopy(self.model).eval() #when we work with q learning, want a model and another model we can evaluate of off. Part of Dueling deep Q
        
        """
        if os.path.exists("models"):
            self.model.load_model(device=device)
            self.target_model.load_model(device=device)
        """
        self.device = device
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.nb_actions = nb_actions

        #hyperparameters
        self.learning_rate = learning_rate
        self.nb_warmup = nb_warmup
        self.gamma = gamma #how much we discount future rewards compared to immediate rewards
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon

        #self.epsilon_lambda = np.log(1/min_epsilon) / nb_warmup # this is the epsilon decay rate

        #update epsilon at every time step instead now
        self.epsilon_decay = 0.99999975#1- (((epsilon - min_epsilon) / nb_warmup) *2) # linear decay rate, close to the nb_warmup steps count
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate
        self.game_steps = 0 #track how many steps taken over entire training
        
        
        #Combines adaptive learning rates with weight decay regularisation for better generalisation
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()
        
        self.memory = ReplayBufferPriority(memory_capacity,self.device)
        
        logging.info(f"starting, device={device}")
        print_info()

    #state is image of our environment
    def get_action(self,state,test=False):

        #only use random action if its training
        if (torch.rand(1) < self.epsilon) and not test: #if a random number between 0 and 1 is smaller than epsilon, do random move
            #randint returns a tensor
            return np.random.randint(self.nb_actions) #random action
        else:

            #convert state into np_array for calculations, then make a tensor, then unsqueese to add batch dimension
            state = torch.tensor(np.array(state), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.model.device)
            #use advantage function to calculate max action
            return self.model(state).argmax().item()

    def store_memory(self,state,action,reward,next_state,done):
        """
        if td_error is None:
            #give a high td_err as at start probably bad, also making it uniform until fill up buffer
            td_error = torch.tensor(1.0, dtype=torch.float32)
        """

        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32,device="cpu"), 
                                            "action": torch.tensor(action,device="cpu"),
                                            "reward": torch.tensor(reward,device="cpu"), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32,device="cpu"), 
                                            "done": torch.tensor(done,device="cpu")
                                            #"td_error":torch.tensor(td_error,dtype=torch.float32)
                                          }, batch_size=[]))

    def decay_epsilon(self):
        # TODO: can do exponential, but good enough
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def sync_networks(self):
        if self.game_steps % self.sync_network_rate == 0 and self.game_steps > 0:
            #TODO: consider tau here instead rather than quick changes
            self.target_model.load_state_dict(self.model.state_dict()) #keep the target_model lined up with main model, its learning in hops


    #epochs = how many iterations to train for
    def train(self,env, epochs):
        #see how the model is doing over time
        stats = {"Returns":[],"Loss": [],"AverageLoss": [], "Epsilon": []} #store as dictinary of lists

        plotter = LivePlot()

        for epoch in range(1,epochs+1):
            state = env.reset() #reset the environment for each iteration
            done = False
            ep_return = 0
            ep_loss = 0

            while not done:
                action = self.get_action(state)

                self.game_steps += 1
                next_state,reward,done,info = env.step(action)
                #order of list matters
                self.memory.add((state,action,next_state,reward,done))

                #actual training part
                #can take out of memory only if sufficient size
                #if len(self.replay_buffer) >= (self.batch_size * 10):
                if self.memory.can_sample(self.batch_size):    
                    self.optimizer.zero_grad()

                    self.sync_networks()

                    (states,actions,next_states,rewards,dones), importance, sample_indices = self.memory.sample(self.batch_size)                   

                    qsa_b = self.model(states)  # Shape: (batch_size, n_actions) as network estimates q value for all actions. so have rows of q values for each action
                    qsa_b = qsa_b[np.arange(self.batch_size), actions.squeeze()] #action contains the actual actions taken, remove extra batch dimension via squeeze
                    # then generate an array of batch indices. So select q value of each action taken

                    # DDQN - Compute target Q-values from the online network, then use the 
                    # target network to evaluate
                    best_next_actions = self.model(next_states).argmax(dim=1) #get the best action using max of dim=1(which are the actions). argmax return indices
                    #this feeds the next_states into target_model and then selects its own values of the actions that online model chose
                    next_qsa_b = self.target_model(next_states)[np.arange(self.batch_size),best_next_actions]
                    
                    # dqn = r + gamma * max Q(s,a)
                    # ddqn = r + gamma * online_network(s',argmax target_network_Q(s',a'))
                    #detach -> important as we dont want to back propagate on target network
                    target_b = (rewards + self.gamma * next_qsa_b * (1 - dones.float()) ).detach() #1-dones.float() -> stop propagating when finished episode
                    
                    #indices = samples["index"]
                    #use epsilon value for exponent b -> as epsilon decreases the b value gets larger
                    loss = self.compute_loss(qsa_b,target_b,importance_weights=importance**(1-self.epsilon))

                    with torch.no_grad():
                        td_errors = (qsa_b - target_b).abs().numpy()

                    self.memory.set_priorities(sample_indices,td_errors)

                    loss.backward()
                    ep_loss += loss.item()
                    self.optimizer.step()
                    self.decay_epsilon() #decay epsilon at each step in environment

                state = next_state #did the training, now move on with next state
                ep_return += reward
                #print(f"Got here, episode return={ep_return}, time step = {self.game_steps}")

            stats["Returns"].append(ep_return)
            stats["Loss"].append(ep_loss)

            #print("Total reward = "+str(ep_return))
            print("Total loss = "+str(ep_loss))
            print("Time Steps = "+str(self.game_steps))

            #gatherin stats
            if epoch % 10 == 0:
                self.model.save_model() #save model every 10th epoch

                #average_returns = np.mean(stats["Returns"][-100:]) #average of the last 100 returns
                average_loss = np.mean(stats["Loss"][-100:])
                #graph can turn too big if we try to plot everything through. Only update a graph data point for every 10 epochs

                stats["AverageLoss"].append(average_loss)
                stats["Epsilon"].append(self.epsilon) #see where the epsilon was at. Do we see higher returns with high epsilon, or only when it dropped etc

                if(len(stats["Loss"]) > 100):
                    logging.info(f"Epoch: {epoch} - Average loss: {np.mean(stats['Loss'][-100:])}  - Epsilon: {self.epsilon} ")
                else:
                    #for the first 100 iterations, just return the episode return,otherwise return the average like above
                    logging.info(f"Epoch: {epoch} - Episode loss: {np.mean(stats['Loss'][-1:])}  - Epsilon: {self.epsilon} ")

            if epoch % 100 == 0:
                plotter.update_plot(stats)
            
            if epoch % 1000 == 0:
                self.model.save_model(f"models/model_iter_{epoch}.pt") #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed
            
        return stats
    
    def compute_loss(self,q_values, q_targets, importance_weights):
        
        # Compute the loss with importance sampling weights
        loss = self.loss(q_values, q_targets)
        weighted_loss = importance_weights * loss
        
        return weighted_loss.mean()

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
                env.render()
    
