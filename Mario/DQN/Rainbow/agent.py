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
from Rainbow.rainbow_model import MarioNet
from model_mobile_vit import MarioNet_ViT
from collections import deque
from segment_tree import MinSegmentTree, SumSegmentTree

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

class ReplayBuffer():

    def __init__(self,input_dim, size, batch_size=32,device="cpu",n_steps=3,gamma=0.99):
        frames = input_dim[0]
        height = input_dim[1]
        width = input_dim[2]

        self.state_storage = np.zeros([size,frames,height,width],dtype=np.float32) #store by size of image as that includes the image shape
        self.next_state_storage = np.zeros([size,frames,height,width], dtype=np.float32)
        self.actions_storage = np.zeros([size], dtype=np.float32)
        self.rewards_storage = np.zeros([size], dtype=np.float32)
        self.done_storage = np.zeros(size,dtype=np.float32)
        self.max_size, self.batch_size = size,batch_size
        self.ptr, self.size, = 0,0
        self.device = device
        
        # N-step Learning storage
        self.step_storage = deque(maxlen=n_steps) #stores the n_step transitions
        self.n_steps = n_steps
        self.gamma = gamma
    
    def insert(self,state,action,reward,next_state,done):

        transition = (state,action,reward,next_state,done)
        self.step_storage.append(transition)

        #if not enough steps then it is not ready yet
        if len(self.step_storage) < self.n_steps:
            return ()
        
        #n_step transition
        reward, next_state,done = self.get_n_step_info(self.step_storage,self.gamma)

        state, action = self.step_storage[0][:2]

        self.state_storage[self.ptr] = state #TODO: normalise values between 0 and 1 for training 
        self.next_state_storage[self.ptr] = next_state
        self.actions_storage[self.ptr] = action
        self.rewards_storage[self.ptr] = reward
        self.done_storage[self.ptr] = done
        #make sure it loops back around once reach size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size +1 ,self.max_size)

        return self.step_storage[0] #return the first transition in n_steps
    
    def sample_batch(self):
        indxs = np.random.choice(self.size,size=self.batch_size,replace=False)
        return dict(states=self.state_storage[indxs],
                    next_states=self.next_state_storage[indxs],
                    actions=self.actions_storage[indxs],
                    rewards=self.rewards_storage[indxs],
                    dones=self.done_storage[indxs],
                    indxs=indxs,)
    
    def sample_batch_idxs(self,idxs):
        return dict(
            states=self.state_storage[idxs],
            next_states=self.next_state_storage[idxs],
            actions=self.actions_storage[idxs],
            rewards=self.rewards_storage[idxs],
            dones=self.done_storage[idxs],
        )

    #return n_step reward,state and done
    def get_n_step_info(self,step_storage,gamma):
        #info of last transition
        reward, next_state, done = step_storage[-1][-3:]

        for transition in reversed(list(step_storage)[:-1]):
            rwd, n_s,d = transition[-3:]

            reward = rwd + gamma * reward * (1-d) #just a normal reward calculation
            next_state,done = (n_s,d) if d else (next_state,done)
        
        return reward,next_state,done

    #return number of elements stored
    def __len__(self):
        return self.size

    def can_sample(self):
        #need enough varied data to sample, as we sample random data
        print(f"current size is {self.size}")
        return self.size >= (self.batch_size * 10)

class PrioritisedMemory(ReplayBuffer):
    
    """Attributes:
    max_priority (float): max priority
    tree_ptr (int): next index of tree
    alpha (float): alpha parameter for prioritized replay buffer
    sum_tree (SumSegmentTree): sum tree for prior
    min_tree (MinSegmentTree): min tree for min prior to get max weight
    """
    def __init__(self, input_dim,capacity,batch_size=32,n_steps=1,alpha=0.6,gamma=0.99,device="cpu"):
        assert alpha >= 0
        super(PrioritisedMemory,self).__init__(input_dim,capacity,batch_size,device,gamma=gamma,n_steps=n_steps)

        self.max_priority, self.tree_ptr = 1.0 ,0
        self.alpha = alpha #controls how much to consider priorities

        tree_capacity = 1
        #capacity is a power of 2
        while tree_capacity < self.max_size:
            tree_capacity *= 2 

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def insert(self,state,action,reward,next_state,done):
        transition = super().insert(state,action,reward,next_state,done)

        if transition:
            #newly inserted get the max_priority as an initialised
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition
    
    def sample_batch(self,beta=0.4):
        
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self.sample_proportional()

        states = self.state_storage[indices]
        next_states = self.next_state_storage[indices]
        actions = self.actions_storage[indices]
        rewards = self.rewards_storage[indices]
        dones = self.done_storage[indices]
        #get the weight of each experience at this index
        weights = np.array([self.calculate_weight(index,beta) for index in indices])

        #now convert into tensors for training
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions.reshape(-1,1), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards.reshape(-1,1), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones.reshape(-1,1), dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights.reshape(-1,1), dtype=torch.float32, device=self.device) #TODO: make sure this is correct

        return dict(states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,)
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)#make sure it never goes above max priority

    #sample indices, proportions 
    def sample_proportional(self):

        indices= []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def calculate_weight(self,index,beta):
        #the pi / sum pk formula
        p_min = self.min_tree.min() / self.sum_tree.sum()
        #get max weight we have
        max_weight = (p_min * len(self)) ** (-beta) # /// go over this

        #calculate weights
        p_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

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
    def __init__(self,input_dims,device="cpu",
                 beta=0.6,
                 prior_eps=1e-6,
                 nb_warmup=250_000,
                 nb_actions=5,
                 memory_capacity=100_000,
                 n_step=3,
                 batch_size=32,
                 learning_rate=0.00020,
                 gamma=0.95,
                 sync_network_rate=10_000,
                 use_vit=False,
                 v_min=0.0,
                 v_max=200.0,
                 atom_size=51,):
        
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

        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate
        self.game_steps = 0 #track how many steps taken over entire training
        
        #Combines adaptive learning rates with weight decay regularisation for better generalisation
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        #categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min,self.v_max,self.atom_size).to(self.device)

        if(use_vit):
            self.model = MarioNet_ViT(nb_actions=nb_actions,device=device) #5 actions for agent can do in this game
        else:
            self.model = MarioNet(input_dims,nb_actions=nb_actions,device=device)
        self.target_model = copy.deepcopy(self.model).eval() #when we work with q learning, want a model and another model we can evaluate of off. Part of Dueling deep Q

        #memory for 1 step transitions  and n_step transitions
        # guarantees any paired 1 step and n_step transitions have the same indices    
        # therefore can sample pairs of transitions from 2 buffers once we have indices for samples
        self.memory = PrioritisedMemory(input_dims,memory_capacity,self.batch_size,n_steps=1,device=self.device)
        
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.n_memory = PrioritisedMemory(input_dims,memory_capacity,self.batch_size,n_steps=n_step,device=self.device,gamma=gamma)
        
        # transition to store in memory
        self.transition = list()

        self.beta = beta # ///TODO: go over what beta does for prioritised buffer
        self.prior_eps = prior_eps
        
        logging.info(f"starting, device={device}")
        print_info()

    #Noisy net way and not epsilon greedy, so just pick the action
    def get_action(self,state,test=False):
        
        #convert state into np_array for calculations, then make a tensor, then unsqueese to add batch dimension
        state = torch.tensor(np.array(state), dtype=torch.float32) \
                    .unsqueeze(0) \
                    .to(self.model.device)
        #use advantage function to calculate max action
        
        action = self.model(state).argmax().item()

        self.transition = [state,action]#store the transition

        return action

    def sync_networks(self):
        if self.game_steps % self.sync_network_rate == 0 and self.game_steps > 0:
            #TODO: consider tau here instead rather than quick changes
            self.target_model.load_state_dict(self.model.state_dict()) #keep the target_model lined up with main model, its learning in hops


    #epochs = how many iterations to train for
    def train(self,env, epochs):
        #see how the model is doing over time
        stats = {"Returns":[],"Loss": [],"AverageLoss": [], "TimeStep": []} #store as dictinary of lists

        plotter = LivePlot()

        for epoch in range(1,epochs+1):
            state = env.reset() #reset the environment for each iteration
            done = False
            ep_return = 0
            ep_loss = 0

            while not done:
                start_whole = time.time()
                action = self.get_action(state)

                self.game_steps += 1
                start_step = time.time()
                next_state,reward,done,info = env.step(action)
                end_step = time.time() - start_step
                #print(f"step took {end_step}")
                #order of list matters

                self.transition += [reward,next_state,done]
                if self.use_n_step:
                    one_step_transition = self.n_memory.insert(*self.transition)
                else:
                    one_step_transition = self.transition

                #add a single step transition
                if one_step_transition:
                    self.memory.insert(*one_step_transition)

                # update beta, make it go closer to 1 as towards the end of training as we want more unbiased updates to 
                fraction = min(epoch/epochs,1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)
                #actual training part
                #can take out of memory only if sufficient size
                #if len(self.replay_buffer) >= (self.batch_size * 10):
                if self.memory.can_sample():    
                    print("can sample")
                    self.sync_networks()

                    samples = self.memory.sample_batch(self.beta)
                    #states = samples["states"]
                    #next_states = samples["next_states"]
                    #actions = samples["actions"]
                    #rewards = samples["rewards"]
                    #dones = samples["dones"]
                    #weights = samples["weights"]
                    indices = samples["indices"]
                    
                    element_loss,weight_loss = self.compute_dqn_loss(samples,self.gamma)

                    if(self.use_n_step):
                        samples = self.n_memory.sample_batch_idxs(indices)
                        gamma = self.gamma ** self.n_step
                        element_loss_n, weight_loss_n = self.compute_dqn_loss(samples,gamma)
                        weight_loss += weight_loss_n         

                    """
                    qsa_b = self.model(states)  # Shape: (batch_size, n_actions) as network estimates q value for all actions. so have rows of q values for each action
                    #qsa_b = qsa_b[np.arange(self.batch_size), actions.squeeze()] #action contains the actual actions taken, remove extra batch dimension via squeeze
                    
                    qsa_b = qsa_b[torch.arange(32), actions.squeeze().long()]
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
                    """

                    self.optimizer.zero_grad()
                    weight_loss.backward()
                    ep_loss += weight_loss.item()
                    self.optimizer.step()

                    loss_for_prior = element_loss.detach().cpu().numpy()
                    new_probabilities = loss_for_prior + self.prior_eps # a small number added at the end to make sure its never 0
                    self.memory.update_priorities(indices,new_probabilities)

                    # NoisyNet: reset noise
                    self.model.reset_noise()
                    self.target_model.reset_noise()

                state = next_state #did the training, now move on with next state
                ep_return += reward
                end_whole = time.time() - start_whole
                #print(f"whole took {end_whole}")
                #print(f"Got here, episode return={ep_return}, time step = {self.game_steps}")

            stats["Returns"].append(ep_return)
            stats["Loss"].append(ep_loss)
            stats["TimeStep"].append(self.game_steps)

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

                if(len(stats["Loss"]) > 100):
                    logging.info(f"Epoch: {epoch} - Average loss: {np.mean(stats['Loss'][-100:])}  - TimeStep: {self.game_steps} ")
                else:
                    #for the first 100 iterations, just return the episode return,otherwise return the average like above
                    logging.info(f"Epoch: {epoch} - Episode loss: {np.mean(stats['Loss'][-1:])}  - TimeStep: {self.game_steps} ")

            if epoch % 100 == 0:
                plotter.update_plot(stats)
            
            if epoch % 1000 == 0:
                self.model.save_model(f"models/model_iter_{epoch}.pt") #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed
            
        return stats


    def compute_dqn_loss(self,samples,gamma):
        states = samples["states"]
        next_states = samples["next_states"]
        actions = samples["actions"]
        rewards = samples["rewards"]
        dones = samples["dones"]
        weights = samples["weights"]

        qsa_b = self.model(states)[torch.arange(32), actions.squeeze().long()]
        qsa_b = qsa_b.unsqueeze(dim=-1)
        next_qsa_b = self.target_model(next_states).max(dim=1,keepdim=True)[0].detach()
        mask = 1 - dones
        target_b = (rewards + self.gamma * next_qsa_b * mask).to(self.device)
        #indices = samples["index"]
        #use epsilon value for exponent b -> as epsilon decreases the b value gets larger

        #element_loss = self.loss(qsa_b, target_b)
        element_loss = torch.nn.functional.smooth_l1_loss(qsa_b, target_b, reduction="none")
        weighted_loss = torch.mean(element_loss * weights)

        return element_loss,weighted_loss

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
    
