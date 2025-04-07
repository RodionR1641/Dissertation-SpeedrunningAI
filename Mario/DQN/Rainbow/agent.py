import random
import torch
import os
import torch.optim as optim
import numpy as np
import time
from Rainbow.rainbow_model import MarioNet
from Rainbow.rainbow_model import NoisyLinear
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

        # Convert state if it's a PyTorch tensor
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        # Handle LazyFrames (Gym Atari wrappers)
        elif hasattr(state, '__array__'):  # Works for LazyFrames
            state = np.array(state)  # Convert LazyFrames to NumPy
        
        # Same for next_state
        if torch.is_tensor(next_state):
            next_state = next_state.cpu().numpy()
        elif hasattr(next_state, '__array__'):
            next_state = np.array(next_state)
        
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
        return self.size >= 80_000 #80k frames from the rainbow paper #(self.batch_size * 10)

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
            # of being sampled. Alpha controls the importance of priority sampling
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

            min_p, max_p = np.min(weights), np.max(weights)
            bins = np.linspace(min_p, max_p, 50)  # 50 evenly spaced bins

            wandb.log({
                "PER_Batch/IS_Weight_Hist": wandb.Histogram(
                    np_histogram=np.histogram(weights, bins=bins)
                )
            }, commit=False)
        
        if game_steps % 10000 == 0: #10,000
            full_stats = self.get_priority_stats(full_buffer=True)
            priorities = full_stats['raw_priorities']

            wandb.log({
                "game_steps": game_steps,
                "PER_Full/Priority_Mean": full_stats['mean'],
                "PER_Full/Priority_Max": full_stats['max'],
                "PER_Full/Priority_Min": full_stats['min'],
                "PER_Full/PriorityP90": full_stats['p90'],
                "PER_Full/Priority_Spread": full_stats['max'] / (full_stats['mean'] + 1e-6), #how much max and min differ
                "PER_Full/HighPriority_Ratio": np.sum(priorities > full_stats['p90'])/len(self), #percentage of 
                #priorities falling outside the 90% range
                "PER_Full/Global_Active_Size": len(self),  # Current buffer size

            },commit=False)  

            min_p, max_p = np.min(priorities), np.max(priorities)
            bins = np.linspace(min_p, max_p, 50)  # 50 evenly spaced bins

            wandb.log({
                "PER_Full/Priority_Hist": wandb.Histogram(
                    np_histogram=np.histogram(priorities, bins=bins)
                )
            }, commit=False)
        
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
class Agent_Rainbow:
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
                 memory_capacity=1_000_000,
                 batch_size=32,
                 learning_rate=0.0000625, #slightly higher rate than original paper for faster covergence with fewer samples
                 adam_epsilon=1.5e-4, #small denominator added to adam to prevent division by 0 and improve numerical stability. reduce sensitivity to tiny gradients. bigger than the default 1e-8
                 gamma=0.99,
                 sync_network_rate=32_000,
                 beta_decay_steps = 10_000_000, #the steps at which beta goes from initial value to 1.0
                 #Categorical DQN parameters
                 v_min=-35.0, #maximum negative reward that can be reasonably expected. If too low, increase to -100. Both of these values
                 #consider the discounted. Smaller values provide more granularity
                 v_max=100.0, #max positive reward that can be reasonably expected for mario
                 atom_size=51,
                 #Prioritised Experience Replay parameters
                 alpha=0.5, #controls how important prioritisation is, 0 is uniform
                 beta=0.4, #compensates for bias - Higher initial β (0.6) applies stronger correction early in training, reducing overfitting to high-priority transitions.
                 prior_eps=1e-6,
                 # N-step learning
                 n_step=3,
                 clip_grad_norm = 10.0, #max gradient update value
                 # Decide if to use a vit feature network or not
                 use_vit=False,
                 ):
        
        self.device = device

        self.nb_actions = nb_actions

        #if loading models, then continue from the epoch left off at training. otherwise start from scratch
        self.curr_epoch = 1

        self.initial_beta = beta
        self.beta_decay_steps = beta_decay_steps

        self.clip_grad_norm = clip_grad_norm
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

        #Combines adaptive learning rates with weight decay regularisation for better generalisation
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,eps=adam_epsilon)

        # transition to store in memory
        self.transition = list()
        
        self.game_steps = 0 #track how many steps taken over entire training
        self.best_time_episode = 1e9
        self.env = env
        self.num_completed_episodes = 0#how many games have ended in getting the flag

        self.is_test = False #controls the test/train mode. Better than just having 2 files
        
        #load existing models
        if os.path.exists("models/rainbow") and load_models_flag==True:
            self.load_models()

        self.epoch = self.curr_epoch #track the current epoch

        self.model.to(self.device)
        self.target_model.to(self.device)
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
        return next_state,reward,done, info


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

        #optimizer steps

        self.optimizer.zero_grad()
        loss.backward()
        ep_loss = loss.item()

        if self.game_steps % 500 == 0:
            #Track gradient norms for monitoring stability and see exploding or vanishing gradients
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)  # L2 norm here - get the gradient tensor, calculate the L2 norm
                    #which is the square root of sum of squared values
                    total_norm += param_norm.item() ** 2 # square each parameters norm and adds to total_norm
            total_norm = total_norm ** 0.5  #Overall gradient norm - square root of total
            
            #calculate per-layer gradient norms. Map layer names to their norms
            layer_norms = {
                name: p.grad.detach().norm(2).item() #map name and gradient norm of that layer
                for name, p in self.model.named_parameters() 
                if p.grad is not None
            }

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
                    # Gradient metrics
                    "Gradient/gradient_norm_total": total_norm ** 0.5,
                    **{f"Gradient/gradients/gradient_{name}": norm for name, norm in layer_norms.items()},
                    
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
                if self.game_steps % 2500 == 0:
                    action_probs = torch.softmax(self.model(states), dim=1).squeeze()
                    for action in range(self.nb_actions):
                        log_data[f"Rainbow/Action_{action}_Prob"] = action_probs[action].mean().item() 
                        #plot the softmax probability for each action for a random state
                        #could help see how confident the network is
                    #log the entropy
                    log_data["Rainbow/Atom_Entropy"] = -(action_probs * torch.log(action_probs + 1e-6)).sum(-1).mean().item(),
                
                #plot return distribution and probabilities of the entire atom size space
                if self.game_steps % 2000 == 0: #5000
                    atom_probs = probs.mean(dim=0).cpu().numpy()
                    support = self.support.to("cpu").numpy()
                    
                    log_data["Rainbow/Return_Probability_Distribution"] = wandb.plot.line(
                        wandb.Table(
                            columns=["Return_Value", "Probability"],
                            data=list(zip(support, atom_probs))
                        ),
                        x="Return_Value",
                        y="Probability",
                        title="Return Probability Distribution"
                    )
                # Log how much probability mass falls in different support regions
                if self.game_steps % 2000 == 0:
                    low = probs[:, :10].sum(dim=-1).mean().item()    # Mass in [-35, -8]
                    mid = probs[:, 10:40].sum(dim=-1).mean().item() # Mass in (-8, 50]
                    high = probs[:, 40:].sum(dim=-1).mean().item()  # Mass in (50, 100]
                    
                    log_data.update({
                        "Rainbow/Support_Low_Mass": low,
                        "Rainbow/Support_Mid_Mass": mid,
                        "Rainbow/Support_High_Mass": high,
                    })

                    atom_probs = probs.mean(dim=0).cpu().numpy()
                    top10_indices = np.argsort(atom_probs)[-10:][::-1] #Indices sorted descending
                    #top10_probs = atom_probs[top10_indices] #Corresponding probabilities

                    support_np = self.support.cpu().numpy().copy()  
                    #top10_returns = support_np[top10_indices]

                    top10_data = []
                    # Create a table

                    for idx in top10_indices:
                        top10_data.append([
                            idx,                          # Atom index (0–50)
                            support_np[idx],      # Return value (e.g., +10.0)
                            atom_probs[idx]                # Probability
                        ])
                    
                    log_data["Rainbow/Top5_Atoms_Table"] = wandb.Table(
                        columns=["Atom_Index", "Return_Value", "Probability"],
                        data=top10_data
                    )
            wandb.log(log_data, commit=False)
        
        #clip gradient after the graphs as the graphs need to show the real gradient
        clip_grad_norm_(self.model.parameters(),self.clip_grad_norm) #prevent exploding gradient

        self.optimizer.step()

        #PER: update priorities. The prioritirisation is based on Loss, not on TD error
        loss_for_prior = element_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps # a small number added at the end to make sure its never 0
        self.memory.update_priorities(indices,new_priorities,self.game_steps)

        # NoisyNet: reset noise
        self.model.reset_noise()
        self.target_model.reset_noise()

        return ep_loss

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

        dist = self.model.dist(states) #for each state, predict a prob distribution over possible returns, shape [batch_size,num_actions,num_atoms]
        log = torch.log(dist[range(self.batch_size), actions]) #select the prob distributions only for the actions actually taken, then take a log
        element_loss = -(proj_dist * log).sum(1) #multiply the target distribution with what model actually predicted. proj_dist is the
        # mathematically correct distribution after seeing the reward. Loss measures how suprised we are, so the further off the predictions
        # the worse the loss.then sum it up to get a cross entropy loss per experience. loss is small when predictions match targets

        return element_loss


    #epochs = how many iterations to train for
    def train(self, epochs):
        self.is_test = False

        episodic_return = 0
        episodic_len = 0

        sps = 0
        window_sec = 10 # number of seconds in which Steps Per Second(SPS) is calculated
        step_count_window = 0# number of time steps seen in the window time
        last_time = time.time()

        for epoch in range(self.curr_epoch,epochs+1):
            state = self.env.reset() #reset the environment for each iteration
            done = False
            ep_loss = 0
            loss_count = 0
            loss = 0
            self.epoch = epoch
            episodes = epoch

            while not done:
                action = self.get_action(state) #this will store the state and action in transition

                self.game_steps += 1

                #track how many steps taken in given time window
                step_count_window += 1
                current_time = time.time()
                if current_time - last_time >= window_sec:
                    sps = step_count_window / (current_time - last_time)
                    step_count_window = 0
                    last_time = current_time

                next_state,reward,done, info = self.step_env(action)

                #PER: update beta, make it go closer to 1 as towards the end of training as we want 
                # more unbiased updates to prevent overfitting
                self.beta = min(self.initial_beta + (1.0 - self.initial_beta) * (self.game_steps / self.beta_decay_steps), 1.0)


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
                    })  # Default x-axis is game_steps

                    if info["flag_get"] == True:
                        self.num_completed_episodes += 1
                        #MOST IMPORTANT - time to complete game. See if we improve in speedrunning when we finish the game
                        # Log completion metrics (by game_steps)
                        wandb.log({
                            "game_steps": self.game_steps,  # Tracks the global step counter
                            "episodes": episodes,
                            "Charts/time_complete": info["time"],
                            "Charts/completion_rate": self.num_completed_episodes / episodes,
                        })

                        if info["time"] < self.best_time_episode:
                            #find the previous file with this old best time
                            filename = f"models/rainbow/best_{self.best_time_episode}.pth"
                            new_filename = f"models/rainbow/best_{info['time']}.pth"

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
                        Time Steps = {self.game_steps}, Beta = {self.beta}, SPS = {sps}")
                print("")
            
            if epoch % 10 == 0:
                self.save_models(epoch=epoch) #save models every 10th epoch
            
            if epoch % 1000 == 0:
                self.save_models(epoch=epoch,weights_filename=f"models/rainbow/rainbow_iter_{epoch}.pth") 
                #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed
        
        self.env.close()
        self.save_models(epoch=epoch) #TODO: can save on the cluster here
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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'beta': self.beta,  # Save the current beta value
            'epoch': epoch,      # Save the current epoch
            'game_steps': self.game_steps,  # Save the global step
            'num_completed_episodes' : self.num_completed_episodes, # num of epochs where the game flag was received
            'best_time_episode': self.best_time_episode
        }

        print("...saving checkpoint...")
        if not os.path.exists("models/rainbow"):
            os.makedirs("models/rainbow",exist_ok=True)
        torch.save(checkpoint,weights_filename)
    
    #if model doesnt exist, we just have a random model
    def load_models(self, weights_filename="models/rainbow/rainbow_latest.pth"):
        try:

            checkpoint = torch.load(weights_filename)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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