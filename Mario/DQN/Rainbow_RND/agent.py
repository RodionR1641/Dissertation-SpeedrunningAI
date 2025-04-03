import random
import torch
import os
import torch.optim as optim
import numpy as np
import time
from Rainbow_RND.rainbow_model import MarioNet, RND_model, NoisyLinear
from model_mobile_vit import MarioNet_ViT
from collections import deque
from segment_tree import MinSegmentTree, SumSegmentTree
from torch.nn.utils import clip_grad_norm_
from gym.wrappers import RecordVideo
import wandb    

# Define log file name (per process)
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
load_models_flag = True #specify if to load existing models or start from new ones


def print_info():
    print(f"Process {rank} started training on GPUs")

    if torch.cuda.is_available():
        try:
            print(torch.cuda.current_device())
            print("GPU Name: " + torch.cuda.get_device_name(0))
            print("PyTorch Version: " + torch.__version__)
            print("CUDA Available: " + str(torch.cuda.is_available()))
            print("CUDA Version: " + str(torch.version.cuda))
            print("Number of GPUs: " + str(torch.cuda.device_count()))
        except RuntimeError as e:
            print(f"{e}")
    else:
        print("cuda not available")

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
        # in standard 1-step learning, agent learns from individual transitions. But here we learn from N-transitions, not just 
        # the immediate reward -> a more informed estimate of return, consider long term consequences better
        # agent associates actions with long term rewards, less bias estimate too
        self.step_storage = deque(maxlen=n_steps) #stores the n_step transitions
        self.n_steps = n_steps
        self.gamma = gamma
    
    def insert(self,state,action,reward,next_state,done):

        transition = (state,action,reward,next_state,done)
        self.step_storage.append(transition)#keep memory of the last n_step transitions

        #if not enough steps then it is not ready yet
        if len(self.step_storage) < self.n_steps:
            return ()
        
        #n_step transition
        reward, next_state,done = self.get_n_step_info(self.step_storage,self.gamma)

        state, action = self.step_storage[0][:2]

        self.state_storage[self.ptr] = state 
        self.next_state_storage[self.ptr] = next_state
        self.actions_storage[self.ptr] = action
        self.rewards_storage[self.ptr] = reward
        self.done_storage[self.ptr] = done
        #make sure it loops back around once reach size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size +1 ,self.max_size)

        return self.step_storage[0] #return the first transition in n_steps
    
    #return the stored N-step transitions
    def sample_batch(self):
        indxs = np.random.choice(self.size,size=self.batch_size,replace=False)
        return dict(states=self.state_storage[indxs],
                    next_states=self.next_state_storage[indxs],
                    actions=self.actions_storage[indxs],
                    rewards=self.rewards_storage[indxs],
                    dones=self.done_storage[indxs],
                    #for N-step learning
                    indxs=indxs,)
    
    def sample_batch_idxs(self,indxs):
        #for N-step learning
        return dict(
            states=self.state_storage[indxs],
            next_states=self.next_state_storage[indxs],
            actions=self.actions_storage[indxs],
            rewards=self.rewards_storage[indxs],
            dones=self.done_storage[indxs],
        )

    #return n_step reward,state and done
    # compute the N-step reward, n-step next state and n-step done based on the last N transitions
    # the oldest transition is at index 0
    def get_n_step_info(self,step_storage,gamma):
        #info of the most recent transition
        reward, next_state, done = step_storage[-1][-3:]

        #move backward through the previous transitions, updating the reward using the formula
        # reward = rwd + gamma * reward * (1 - d)
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
        self.alpha = alpha #controls how much to consider priorities over stochastic sampling. A mix between pure greedy and stochastic
        # with the formula P(i) = pi^a/sum pk_a for all pk, priority of transition i
        #alpha = 0 -> uniform(p^0 = 1)

        tree_capacity = 1
        #capacity is a power of 2
        while tree_capacity < self.max_size:
            tree_capacity *= 2 

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def insert(self,state,action,reward,next_state,done):
        transition = super().insert(state,action,reward,next_state,done)

        if transition:
            #newly inserted get the max_priority as the TD error is unknown and we want all transitions to have a chance 
            # of being sampled
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition
    
    def sample_batch(self,game_steps,beta=0.4):
        
        assert len(self) >= self.batch_size
        assert beta > 0 #beta controls for bias introduce by prioritised sampling. DQN tries to remove correlations of observations
        #through uniform sampling. Bias can be corrected by importance sampling(IS) weights
        # wi = (1/N * 1/P(i))^beta. Beta=1 fully compensates for non-uniform probabilites
        # these weights can then be used in q learning update by using wi * TD_err
        # we want more unbiased updates towards the end of training, so go towards beta=1 towards end of training

        indices = self.sample_proportional()

        states = self.state_storage[indices]
        next_states = self.next_state_storage[indices]
        actions = self.actions_storage[indices]
        rewards = self.rewards_storage[indices]
        dones = self.done_storage[indices]
        #get the weight of each experience at this index
        weights = np.array([self.calculate_weight(index,beta) for index in indices])

        if game_steps % 500 == 0: #500
            batch_stats = self.get_priority_stats(full_buffer=False,indices=indices)

            wandb.log({
                "game_steps": game_steps,
                "PER_Batch/Priority_Mean": batch_stats['mean'],
                "PER_Batch/Priority_Max": batch_stats['max'],
                "PER_Batch/Priority_Min": batch_stats['min'],
                #weight stats
                "PER_Batch/IS_Weight_mean": np.mean(weights),
                "PER_Batch/IS_Weight_max": np.max(weights),
                "PER_Batch/IS_Weight_min": np.min(weights),
                "PER_Batch/IS_Weight_std": np.std(weights),
                "PER_Batch/Effective_Batch_Size": (weights > 0.1).sum(),  # How many samples have significant weight
                # Scatter plot (every 2.5K steps)
                **({
                    "PER_Batch/Priority_vs_Weight": wandb.plot.scatter(
                        wandb.Table(
                            columns=["Priority", "IS_Weight"],
                            data=list(zip(batch_stats['raw_priorities'], weights))
                        ),
                        x="Priority",
                        y="IS_Weight",
                        title="Priority vs. IS Weight"
                    )
                } if game_steps % 2500 == 0 else {}) #2500

            },commit=False)    

            wandb.log({
                "PER_Batch/IS_Weight_Hist": wandb.Histogram(
                    weights,
                    num_bins=50
                )
            },commit=False)
        
        if game_steps % 10000 == 0: #10,000
            full_stats = self.get_priority_stats(full_buffer=True)

            wandb.log({
                "game_steps": game_steps,
                "PER_Full/Priority_Mean": full_stats['mean'],
                "PER_Full/Priority_Max": full_stats['max'],
                "PER_Full/Priority_Min": full_stats['min'],
                "PER_Full/PriorityP90": full_stats['p90'],
                "PER_Full/Priority_Spread": full_stats['max'] / (full_stats['mean'] + 1e-6), #how much max and min differ
                "PER_Full/HighPriority_Ratio": np.sum(full_stats['raw_priorities'] > full_stats['p90'])/len(self), #percentage of 
                #priorities falling outside the 90% range
                "PER_Full/Global_Active_Size": len(self),  # Current buffer size

            },commit=False)  
            wandb.log({
                "PER_Full/Priority_Hist": wandb.Histogram(
                    full_stats['raw_priorities'],
                    num_bins=50
                )
            },commit=False)
        
        return dict(states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,)
    
    def get_priority_stats(self, full_buffer=False,indices=None):
        """Returns priority statistics for either current batch or full buffer"""
        if full_buffer:
            # Full buffer scan (use sparingly)
            priorities = np.array([self.sum_tree[idx] ** (1/self.alpha) 
                                for idx in range(len(self))])
        else:
            # Current batch only, selected indices
            if not indices:
                indices = self.sample_proportional()
            
            priorities = np.array([self.sum_tree[idx] ** (1/self.alpha) 
                                for idx in indices])
        
        return {
            'mean': np.mean(priorities),
            'max': np.max(priorities),
            'min': np.min(priorities),
            'p90': np.percentile(priorities, 90),
            'raw_priorities': priorities  # Raw values for histogram
        }

    
    def update_priorities(self, indices, priorities,game_steps):
        #Update priorities of sampled transitions based on td error
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)# update our max priority in case this is higher
        

    #sample indices based on proportions 
    def sample_proportional(self):

        indices= []
        p_total = self.sum_tree.sum(0, len(self) - 1) #calculate the sum of priorities
        segment = p_total / self.batch_size #get a proportion based on batch size
        
        #iterate i times to sample required amount for our batch
        for i in range(self.batch_size):
            #for each iteration, calculate the bounds which define a segment of total priority range. So e.g. total priority is 100
            # and batch size is 5 -> segment is 20
            #so go through each segment of priorities
            a = segment * i 
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)#generate a random number within segment range
            idx = self.sum_tree.retrieve(upperbound) #get the index corresponding to the generated upperbound, this is determined
            # by the priority distribution. Higher priorities are more likely to be selected as they make up a higher amount of 
            # range within the segment e.g. 5 > 2, between 0 and 20(segment size), 5 is more likely to be "fallen" to 
            indices.append(idx)
            
        return indices
    
    #calculate the weight that adjusts the learning process based on priority
    def calculate_weight(self,index,beta):
        p_min = self.min_tree.min() / self.sum_tree.sum() #minimum priority normalised by the sum of all priorities
        #get max weight we have
        max_weight = (p_min * len(self)) ** (-beta) #len(self) is number of experiences in buffer. calculate the maximum
        #weight based on the minimum priority -> minimum priority will have the max weight

        p_sample = self.sum_tree[index] / self.sum_tree.sum() #normalise the given priority by the total priority 
        weight = (p_sample * len(self)) ** (-beta) #beta calculation for this priority
        weight = weight / max_weight #make sure it is bounded by max weight. comparative scale

        return weight

#plays the game
#covers a lot of training
class Agent_Rainbow_RND:
    #nb_actions -> number of actions
    #memory_capacity -> number of previous transitions to store in replay buffers
    #batch_size -> how much is sampled at each time step for training
    #learning_rate -> how big of a step we want the agent to take at a time, how quickly we want it to learn.
    #gamma -> controls how much importance future rewards are given when calculating q values
    #sync_network_rate -> controls in how many timesteps to copy over the online network into target network
    #v_min -> min value of support
    #v_max -> max value of support
    #atom_size -> the unit number of support
    #support -> support for categorical DQN
    #alpha -> 
    #beta -> 
    #prior_eps -> small value to make sure every experience has a chance to be selected(even if loss was 0)
    #n_step -> step number to calculate n-step td error
    #memory_n -> n-step replay buffer
    def __init__(self,input_dims,
                 env,
                 device="cpu",
                 nb_actions=5,
                 memory_capacity=50_000,
                 batch_size=32,
                 learning_rate=1e-5,
                 gamma=0.99,
                 sync_network_rate=1_000,
                 #Categorical DQN parameters
                 v_min=0.0,
                 v_max=200.0,
                 atom_size=51,
                 #Prioritised Experience Replay parameters
                 alpha=0.2,
                 beta=0.6,
                 prior_eps=1e-6,
                 # N-step learning
                 n_step=3,
                 # Decide if to use a vit feature network or not
                 use_vit=False,
                 ):
        
        
        self.device = device

        self.nb_actions = nb_actions
        
        #if loading models, then continue from the epoch left off at training. otherwise start from scratch
        self.curr_epoch = 1

        #hyperparameters for learning
        self.learning_rate = learning_rate
        self.gamma = gamma

        #network hyperparameters
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate
        

        #categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min,self.v_max,self.atom_size).to(self.device)

        #PER parameters
        self.beta = beta
        self.prior_eps = prior_eps

        #memory for 1 step transitions  and n_step transitions
        # guarantees any paired 1 step and n_step transitions have the same indices    
        # therefore can sample pairs of transitions from 2 buffers once we have indices for samples
        self.memory = PrioritisedMemory(input_dims,memory_capacity,self.batch_size,alpha=alpha,gamma=gamma,device=self.device)
        #memory for N-step learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            #store n-step(multiple transitions) and 1-step separately. To compute loss, agent accesses paired transitions from
            #both buffers. paired transitions have the same indices. Storing seperately allows to avoid redundancy, as 1step buffer
            #doesnt need n-step info. The agent can learn from both 1-step and n-step, improved learning
            
            #note: this is just a normal buffer not prioritised. The 1 step buffer is used more frequently so prioritising it 
            #is more important.
            self.n_memory = ReplayBuffer(input_dims,memory_capacity,self.batch_size
                                              ,n_steps=n_step,device=self.device,gamma=gamma)
        

        #online and target DQN networks
        if(use_vit):
            self.model = MarioNet_ViT(nb_actions=nb_actions,device=device)
            self.target_model = MarioNet_ViT(nb_actions=nb_actions,device=device)
        else:
            self.model = MarioNet(input_dims,support=self.support,out_dim=self.nb_actions,
                                  atom_size=self.atom_size,device=device)
            self.target_model = MarioNet(input_dims,support=self.support,out_dim=self.nb_actions,
                                  atom_size=self.atom_size,device=device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()#evaluation mode, means it wont learn via gradient updates

        #RND networks - dont copy them over as target must be a random network we predict
        self.model_rnd = RND_model(input_dims,device=device)
        self.target_rnd = RND_model(input_dims,device=device)
        self.target_rnd.eval()

        #Combines adaptive learning rates with weight decay regularisation for better generalisation
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #rnd optimizer is different. TODO: maybe a different learning rate 
        self.optimizer_rnd = optim.Adam(self.model_rnd.parameters(), lr=self.learning_rate)

        # transition to store in memory
        self.transition = list()
        
        self.game_steps = 0 #track how many steps taken over entire training
        self.best_time_episode = 1e9 #best time of an episode in speedrunning
        self.env = env
        self.num_completed_episodes = 0#how many games have ended in getting the flag

        self.is_test = False #controls the test/train mode. Better than just having 2 files
        
        #load existing models
        if os.path.exists("models/rainbow_rnd") and load_models_flag==True:
            self.load_models()
        
        self.epoch = self.curr_epoch #track the current epoch

        self.model.to(self.device)
        self.target_model.to(self.device)
        self.model_rnd.to(self.device)
        self.target_rnd.to(self.device)
        print(f"Device for Agent = {device}")

        print_info()

    def record_video(self,run_name):
        self.env = RecordVideo(self.env,"videos/Rainbow_RND",name_prefix=f"{run_name}_{self.epoch}"
                          ,episode_trigger=lambda x: x % 500 == 0)  # Record every 500th episode

    #Noisy net way and not epsilon greedy, so just pick the action
    def get_action(self,state):
        
        #convert state into np_array for calculations, then make a tensor, then unsqueese to add batch dimension
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.model.device)
        #use advantage function to calculate max action
        
        action = self.model(state).argmax().item() #self.model() returns the q value, then get the action associated with max value

        if not self.is_test:
            self.transition = [state,action]#store the transition

        return action

    def sync_networks(self):
        if self.game_steps % self.sync_network_rate == 0 and self.game_steps > 0:
            self.target_model.load_state_dict(self.model.state_dict()) #keep the target_model lined up with main model, its learning in hops
            print(f"synced target and online networks, current game step = {self.game_steps}")


    def step_env(self,action,intrinsic_reward=None):
        next_state,reward,done, info = self.env.step(action)
        extrinsic_reward = reward
        if intrinsic_reward:
            reward += intrinsic_reward #total reward is extrinsic(from env) + intrinsic reward

        if not self.is_test:
            self.transition += [reward,next_state,done]
            if self.use_n_step:
                one_step_transition = self.n_memory.insert(*self.transition)
            else:
                one_step_transition = self.transition

            #add a single step transition
            if one_step_transition:
                self.memory.insert(*one_step_transition)
        
        #return both the total reward and true reward(extrinsic) for stats
        return next_state,reward,extrinsic_reward,done, info

    
    #update by gradient descent and calculate loss her
    def update_model(self):
        self.sync_networks()

        samples = self.memory.sample_batch(self.game_steps,self.beta)
        weights = torch.tensor(samples["weights"].reshape(-1,1), dtype=torch.float32, device=self.device)
        indices = samples["indices"]
        
        #1 step learning loss
        element_loss = self.compute_dqn_loss(samples,self.gamma)

        #PER: importance sampling before getting the average
        loss = torch.mean(element_loss * weights)

        #N-step learning loss. Combine n-loss and 1 step loss to prevent high variance, but original paper employs n-step only
        if(self.use_n_step):
            gamma = self.gamma ** self.n_step
            samples = self.n_memory.sample_batch_idxs(indices)
            element_loss_n = self.compute_dqn_loss(samples,gamma)#measure how well the agent predictions match the n step target
            #1 step is high bias, and n step is high variance. so we get a balance
            element_loss += element_loss_n

            #PER: importance sampling before getting the average
            loss = torch.mean(element_loss * weights)

        #compute the RND loss - make sure the states are tensors
        intrinsic_true = self.target_rnd(torch.tensor(samples["states"], dtype=torch.float32, device=self.device))
        intrinsic_predicted = self.model_rnd(torch.tensor(samples["states"], dtype=torch.float32, device=self.device))

        #optimizer steps
        self.optimizer_rnd.zero_grad()
        loss_rnd = torch.pow(intrinsic_predicted - intrinsic_true,2).sum()
        loss_rnd.backward()

        if self.game_steps % 500 == 0:

            self.plot_gradient_norms(self.model_rnd,rnd=True) #plot gradient norms for rnd

        self.optimizer_rnd.step()

        self.optimizer.zero_grad()
        loss.backward()
        ep_loss = loss.item()

        if self.game_steps % 1 == 0:

            with torch.no_grad():
                states = torch.tensor(samples["states"], dtype=torch.float32, device=self.device)
                actions = torch.tensor(samples["actions"], dtype=torch.long, device=self.device) #long tensor for actions
                dist = self.model.dist(states)
                probs = torch.softmax(dist[range(self.batch_size), actions ], dim=-1) #softmax so easy to compare the distribution, all add up to 1 
                expected_q = (probs * self.support).sum(-1)

                if self.use_n_step:
                    n_step_samples = self.n_memory.sample_batch_idxs(indices)
                    n_step_gamma = self.gamma ** self.n_step
                    n_step_td = self.compute_dqn_loss(n_step_samples, n_step_gamma)

                log_data = {
                    "game_steps": self.game_steps,
                    
                    # Distributional RL
                    "Rainbow/Expected_Q_Avg": expected_q.mean().item(),
                    "Rainbow/Atoms_Std": probs.std(dim=-1).mean().item(),
                    "Rainbow/Expected_Q_Std": expected_q.std().item(),

                    # N-step (if used)
                    "Rainbow/N_Step_Gamma": self.gamma ** self.n_step if self.use_n_step else 0,
                    "Rainbow/N_Step_TD_Avg": n_step_td.mean().item(),
                    "Rainbow/N_Step_TD_Ratio": n_step_td.mean().item() / element_loss.mean().item()
                }
                #noisy net data
                for name, module in self.model.named_modules():
                    if isinstance(module, NoisyLinear):
                        # Calculate current effective noise magnitude
                        weight_noise = (module.weight_sigma * module.weight_epsilon).std().item()
                        bias_noise = (module.bias_sigma * module.bias_epsilon).std().item()
                        
                        log_data[f"Noisy/{name}.weight"] = weight_noise
                        log_data[f"Noisy/{name}.bias"] = bias_noise
                        log_data[f"Noisy/{name}.weight_mu"] = module.weight_mu.abs().mean().item()
                        log_data[f"Noisy/{name}.bias_mu"] = module.bias_mu.abs().mean().item()


                # Action distribution (less frequent)
                if self.game_steps % 1 == 0:
                    action_probs = torch.softmax(self.model(states), dim=1).squeeze()
                    for action in range(self.nb_actions):
                        log_data[f"Rainbow/Action_{action}_Prob"] = action_probs[action].mean().item() 
                        #plot the softmax probability for each action for a random state
                        #could help see how confident the network is
                    #log the entropy
                    log_data["Rainbow/Atom_Entropy"] = -(action_probs * torch.log(action_probs + 1e-6)).sum(-1).mean().item(),
            wandb.log(log_data, commit=False)

            self.plot_gradient_norms(self.model) #plot gradient
        #clip gradient after the graphs as the graphs need to show the real gradient
        clip_grad_norm_(self.model.parameters(),10.0) #prevent exploding gradient

        self.optimizer.step()

        #PER: update priorities
        loss_for_prior = element_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps # a small number added at the end to make sure its never 0
        self.memory.update_priorities(indices,new_priorities,self.game_steps)

        # NoisyNet: reset noise
        self.model.reset_noise()
        self.target_model.reset_noise()

        return ep_loss
    
    #plots the gradient norms for the specified model
    def plot_gradient_norms(self,model,rnd=False):
        #Track gradient norms for monitoring stability and see exploding or vanishing gradients
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)  # L2 norm here - get the gradient tensor, calculate the L2 norm
                #which is the square root of sum of squared values
                total_norm += param_norm.item() ** 2 # square each parameters norm and adds to total_norm
        total_norm = total_norm ** 0.5  #Overall gradient norm - square root of total
        
        #calculate per-layer gradient norms. Map layer names to their norms
        layer_norms = {
            name: p.grad.detach().norm(2).item() #map name and gradient norm of that layer
            for name, p in model.named_parameters() 
            if p.grad is not None
        }
        if rnd:
            wandb.log({
                "game_steps": self.game_steps,
                "Gradient_rnd/gradient_norm_total": total_norm,
                **{f"Gradient_rnd/gradients/gradient_{name}": norm for name, norm in layer_norms.items()}
            },commit=False)
        else:
            wandb.log({
                "game_steps": self.game_steps,
                "Gradient/gradient_norm_total": total_norm,
                **{f"Gradient/gradients/gradient_{name}": norm for name, norm in layer_norms.items()}
            },commit=False)

    #return categorical dqn loss
    def compute_dqn_loss(self,samples,gamma):

        #now convert into tensors for training
        states = torch.tensor(samples["states"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(samples["next_states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(samples["actions"], dtype=torch.long, device=self.device) #long tensor for actions
        rewards = torch.tensor(samples["rewards"].reshape(-1,1), dtype=torch.float32, device=self.device)
        dones = torch.tensor(samples["dones"].reshape(-1,1), dtype=torch.float32, device=self.device)
        
        #Categorical DQN algorithm here
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        #we learn distribution of returns instead of the expected return. Capture full distribution of outcomes. which is sum(gammat * Rt)
        #agent can learn the uncertainty and risk of certain actions better.
        #use discrete support z, where z is a vector with N atoms defined as zi = Vmin+(i-1)(Vmax-Vmin)/(Natoms-1) for i in Natams
        #are the maximum/minimum possible returns Vmin and Vmax
        #Distribution of returns is a probability mass function(PMF) over these atoms

        #return of distributions satisfies a variant of bellman equation. Z(s,a) D= R + gamma Z(s',a'). D= means equal in distribution
        #Z(s',a') is the return distribution of next state and the optimal action a*= Pi*(s') wher Pi is the policy
        #so distribution of returns in optimal policy should match this target distribution, with a discount and a reward R
        # we minimise the  
        with torch.no_grad():
            #Double DQN
            next_action = self.model(next_states).argmax(1)#get the next_action that online network would choose
            next_dist = self.target_model.dist(next_states)
            next_dist = next_dist[range(self.batch_size),next_action]#see the q values the target model gives for the
            #actions online network took

            t_z = rewards + (1 - dones) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.model.dist(states)
        log = torch.log(dist[range(self.batch_size), actions])
        element_loss = -(proj_dist * log).sum(1) #different loss function compared to MSE

        return element_loss


    #epochs = how many iterations to train for
    def train(self, epochs):

        self.is_test = False
        #kept outside outer loop to not lose the data
        episodic_return = 0 #note, this is only the extrinsic reward of the environment
        episodic_len = 0

        sps = 0
        window_sec = 10 # number of seconds in which Steps Per Second(SPS) is calculated
        step_count_window = 0# number of time steps seen in the window time
        last_time = time.time()

        for epoch in range(self.curr_epoch,epochs+1):
            state = self.env.reset() #reset the environment for each iteration
            done = False
            ep_reward_intrinsic = 0
            ep_reward_extrinsic = 0
            intrinsic_reward = 0
            extrinsic_reward = 0
            ep_loss = 0
            loss_count = 0
            loss = 0
            self.epoch = epoch
            episodes = epoch

            while not done:
                action = self.get_action(state) #this will store the state and action in transition

                self.game_steps += 1

                #track how many steps taken in given time window for SPS
                step_count_window += 1
                current_time = time.time()
                if current_time - last_time >= window_sec:
                    sps = step_count_window / (current_time - last_time)
                    step_count_window = 0
                    last_time = current_time

                #RND: compute intrinsic reward on the current state
                ##make sure it is a tensor when passed into network -> convert Lazy Frames into tensor. make sure it has 
                #batch dimension
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
                true_intrinsic = self.target_rnd(state_tensor)
                predicted_intrinsic = self.model_rnd(state_tensor)

                #the RND reward is just the loss between target and predicted
                #the higher the loss, the less visited the state is likely to be so its novel -> want to encourage exploration
                intrinsic_reward = torch.pow(predicted_intrinsic - true_intrinsic,2).sum().detach()
                #clamp the reward
                intrinsic_reward = intrinsic_reward.clamp(-1.0,1.0).item() #keep the rewards clamped to not have too much effect

                next_state,reward,extrinsic_reward,done, info = self.step_env(action,intrinsic_reward=intrinsic_reward)
                ep_reward_extrinsic += extrinsic_reward
                ep_reward_intrinsic += intrinsic_reward

                #PER: update beta, make it go closer to 1 as towards the end of training as we want 
                # more unbiased updates to prevent overfitting
                fraction = min(epoch/epochs,1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)
                #actual training part
                #can take out of memory only if sufficient size
                if self.memory.can_sample():    
                    loss = self.update_model()
                    ep_loss += loss
                    loss_count += 1

                state = next_state #did the training, now move on with next state

                if "episode" in info:
                    episodic_reward = info["episode"]["r"]
                    episodic_len = info["episode"]["l"]

                    wandb.log({
                        "game_steps": self.game_steps,
                        "episodes": episodes,
                        # Log by game_steps (fine-grained steps)
                        "Charts/episodic_return": episodic_reward,
                        "Charts/episodic_length": episodic_len,
                    },commit=False)  # Default x-axis is game_steps

                    if info["flag_get"] == True:
                        self.num_completed_episodes += 1
                        #MOST IMPORTANT - time to complete game. See if we improve in speedrunning when we finish the game
                        # Log completion metrics (by game_steps)
                        wandb.log({
                            "game_steps": self.game_steps,  # Tracks the global step counter
                            "episodes": episodes,
                            "Charts/time_complete": info["time"],
                            "Charts/completion_rate": self.num_completed_episodes / episodes,
                        },commit=False)

                        if info["time"] < self.best_time_episode:
                            #find the previous file with this old best time
                            filename = f"models/rainbow_rnd/best_{self.best_time_episode}.pth"
                            new_filename = f"models/rainbow_rnd/best_{info['time']}.pth"

                            #rename so that not saving a new file for each new time
                            if os.path.exists(filename):
                                os.rename(filename,new_filename)
                            
                            #save this model that gave best time, if the model didnt exist then its just created
                            self.best_time_episode = info["time"]
                            self.save_models(epoch,new_filename)

            wandb.log({
                "game_steps": self.game_steps,
                "episodes": episodes,
                "Charts/epochs": epoch,
                "Charts/beta": self.beta,
                #total extrinsic and intrinsic rewards for the episode
                "Charts/intrinsic_reward_total": ep_reward_intrinsic,
                "Charts/extrinsic_reward_total": ep_reward_extrinsic,
                #average extrinsic and intrinsic rewards for the episode steps
                "Charts/intrinsic_reward_avg": ep_reward_intrinsic/episodic_len,
                "Charts/extrinsic_reward_avg": ep_reward_extrinsic/episodic_len,
                #last extrinsic and intrinsic rewards
                "Charts/intrinsic_reward": intrinsic_reward,
                "Charts/extrinsic_reward"  : extrinsic_reward,

                "Charts/SPS": sps
            })

            if loss > 0 or loss_count > 0:
                #average loss - more representitive
                wandb.log({
                    "game_steps": self.game_steps,
                    "episodes": episodes,
                    "losses/loss": loss,
                    "losses_avg/loss": ep_loss/loss_count,
                    "losses_total/total_loss" : ep_loss
                })

            #gatherin stats
            if epoch % 10 == 0:
                print("")
                if loss_count > 0:
                    print(f"Episode return = {episodic_return}, Episode len = {episodic_len},  \
                        Episode loss = {ep_loss}, Average loss = {ep_loss/loss_count}, Epoch = {epoch}, \
                        Time Steps = {self.game_steps}, Extrinsic Reward = {ep_reward_extrinsic}, \
                        Intrinsic Reward = {ep_reward_intrinsic}, Beta = {self.beta}, \
                        SPS = {sps}")
                print("")
            
            if epoch % 10 == 0:
                self.save_models(epoch=epoch)

            if epoch % 1000 == 0:
                self.save_models(epoch=epoch,weights_filename=f"models/rainbow_rnd/rainbow_rnd_iter_{epoch}.pth")
                 #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed
        
        self.save_models(epoch=epoch)
        self.env.close()
        wandb.finish()

    #run something on the machine, and see how we perform
    def test(self):
        self.is_test = True
        normal_env = self.env

        state = self.env.reset()
        done = False

        #1000 steps
        while not done:
            time.sleep(0.01) #by default it runs very quickly, so slow down
            action = self.get_action(state)
            next_state,reward,done, info = self.env.step(action) #make the environment step through the game
            
            state = next_state

            self.env.render()
            if "episode" in info:
                    episodic_return = info["episode"]["r"]
                    episodic_len = info["episode"]["l"]
                    print(f"episodic return = {episodic_return}, episodic len = {episodic_len}")

        self.env.close()
        #reset
        self.env = normal_env

    #these models take a while to train, want to save it and reload on start. Save both target and online for exact reproducibility
    def save_models(self,epoch,weights_filename="models/rainbow/rainbow_latest.pth"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'model_rnd_state_dict': self.model_rnd.state_dict(),
            'target_rnd_state_dict': self.target_rnd.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_rnd_state_dict': self.optimizer_rnd.state_dict(),
            'beta': self.beta,  # Save the current beta value
            'epoch': epoch,      # Save the current epoch
            'game_steps': self.game_steps,  # Save the global step
            'num_completed_episodes' : self.num_completed_episodes, # num of epochs where the game flag was received
            'best_time_episode': self.best_time_episode
        }

        print("...saving checkpoint...")
        if not os.path.exists("models/rainbow_rnd"):
            os.makedirs("models/rainbow_rnd",exist_ok=True)
        torch.save(checkpoint,weights_filename)
    
    #if model doesnt exist, we just have a random model
    def load_models(self, weights_filename="models/rainbow/rainbow_latest.pth"):
        try:

            checkpoint = torch.load(weights_filename)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            self.model_rnd.load_state_dict(checkpoint["model_rnd_state_dict"])
            self.target_rnd.load_state_dict(checkpoint["target_rnd_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.optimizer_rnd.load_state_dict(checkpoint["optimizer_rnd_state_dict"])
            self.beta = checkpoint["beta"]
            self.curr_epoch = checkpoint["epoch"]
            self.game_steps = checkpoint["game_steps"]
            self.num_completed_episodes = checkpoint["num_completed_episodes"]
            self.best_time_episode = checkpoint["best_time_episode"]

            print(f"Loaded weights filename: {weights_filename}, curr_epoch = {self.curr_epoch}, beta = {self.beta}, \
                  game steps = {self.game_steps}")    
        except Exception as e:
            print(f"No weights filename: {weights_filename}, using a random initialised model")
            print(f"Error: {e}")