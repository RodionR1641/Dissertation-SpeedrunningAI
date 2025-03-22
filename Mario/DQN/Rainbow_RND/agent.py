import random
import torch
import gym
import torch.optim as optim
import numpy as np
import time
from Rainbow_RND.rainbow_model import MarioNet, RND_model
from model_mobile_vit import MarioNet_ViT
from collections import deque
from segment_tree import MinSegmentTree, SumSegmentTree
from torch.nn.utils import clip_grad_norm_


# Define log file name (per process)
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
video_folder = "" #TODO: make a folder here


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
        return self.size >= (self.batch_size * 5)

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
    
    def sample_batch(self,beta=0.4):
        
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

        return dict(states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,)
    
    def update_priorities(self, indices, priorities):
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
                 sync_network_rate=10_000,
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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        #rnd optimiser is different
        self.optimizer_rnd = optim.AdamW(self.model_rnd.parameters(), lr=self.learning_rate)

        # transition to store in memory
        self.transition = list()
        
        self.game_steps = 0 #track how many steps taken over entire training
        self.env = env

        self.is_test = False #controls the test/train mode. Better than just having 2 files
        
        self.model.to(self.device)
        self.target_model.to(self.device)
        print_info()

    #Noisy net way and not epsilon greedy, so just pick the action
    def get_action(self,state):
        
        #convert state into np_array for calculations, then make a tensor, then unsqueese to add batch dimension
        state = torch.tensor(np.array(state), dtype=torch.float32) \
                    .unsqueeze(0) \
                    .to(self.model.device)
        #use advantage function to calculate max action
        
        action = self.model(state).argmax().item() #self.model() returns the q value, then get the action associated with max value

        if not self.is_test:
            self.transition = [state,action]#store the transition

        return action

    def sync_networks(self):
        if self.game_steps % self.sync_network_rate == 0 and self.game_steps > 0:
            self.target_model.load_state_dict(self.model.state_dict()) #keep the target_model lined up with main model, its learning in hops

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

    #set the tensorboard writer
    def set_writer(self,writer):
        self.writer = writer
    
    #update by gradient descent and calculate loss her
    def update_model(self):
        self.sync_networks()

        samples = self.memory.sample_batch(self.beta)
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
        self.optimizer_rnd.step()

        self.optimizer.zero_grad()
        loss.backward()
        ep_loss = loss.item()
        clip_grad_norm_(self.model.parameters(),10.0) #prevent exploding gradient
        self.optimizer.step()

        #PER: update priorities
        loss_for_prior = element_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps # a small number added at the end to make sure its never 0
        self.memory.update_priorities(indices,new_priorities)

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
        for epoch in range(1,epochs+1):
            state = self.env.reset() #reset the environment for each iteration
            done = False
            ep_reward_intrinsic = 0
            ep_reward_extrinsic = 0
            ep_loss = 0
            loss_count = 0
            loss = 0

            while not done:
                action = self.get_action(state) #this will store the state and action in transition

                self.game_steps += 1

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
                    episodic_return = info["episode"]["r"]
                    episodic_len = info["episode"]["l"]
                    self.writer.add_scalar("Charts/episodic_return", episodic_return, self.game_steps) 
                    self.writer.add_scalar("Charts/episodic_length", episodic_len, self.game_steps)

            self.writer.add_scalar("Charts/intrinsic_reward",intrinsic_reward,self.game_steps)
            self.writer.add_scalar("Charts/extrinsic_reward",extrinsic_reward,self.game_steps)
            self.writer.add_scalar("Charts/beta",self.beta,self.game_steps)
            self.writer.add_scalar("Charts/epochs",epoch,self.game_steps)

            if loss > 0 or loss_count > 0:
                #average loss - more representitive
                self.writer.add_scalar("losses/loss_episodic",ep_loss/loss_count,self.game_steps)
                #last loss - up to date changes shown
                self.writer.add_scalar("losses/loss",loss,self.game_steps)

            #gatherin stats
            if epoch % 10 == 0:
                print("")
                if loss_count > 0:
                    print(f"Episode return = {episodic_return}, Episode len = {episodic_len},  \
                        Episode loss = {ep_loss}, Average loss = {ep_loss/loss_count}, Epoch = {epoch}, \
                        Time Steps = {self.game_steps}, Extrinsic Reward = {ep_reward_extrinsic}, \
                        Intrinsic Reward = {ep_reward_intrinsic}, Beta = {self.beta}")
                print("")
            
            if epoch % 100 == 0:
                self.model.save_model()
                self.model_rnd.save_model()

            if epoch % 1000 == 0:
                self.model.save_model(f"models/rainbow_rnd_iter_{epoch}.pt")
                 #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed
        
        self.env.close()
        self.writer.close()
        self.model.save_model()
        self.model_rnd.save_model()

    #run something on the machine, and see how we perform
    def test(self):
        self.is_test = True

        #recording video
        normal_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env,video_folder=video_folder)

        state = self.env.reset()
        done = False
        score = 0

        #1000 steps
        while not done:
            time.sleep(0.01) #by default it runs very quickly, so slow down
            action = self.get_action(state,test=True)
            state,reward,done, _ = self.env.step(action) #make the environment step through the game
            score += reward
            self.env.render()
        print("score: ",score)

        self.env.close()

        #reset
        self.env = normal_env