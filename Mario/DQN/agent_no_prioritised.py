import torch
import copy
import torch.optim as optim
import numpy as np
import time
from model import MarioNet
from model_mobile_vit import MarioNet_ViT
import gym

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

#agent's memory
# The way deep q learning works: 
# build a buffer over time of all of the state-action pairs it played
# pick up the state,the action, the next state and the reward of every play
# then use that to Train the model to approximate the Q value

# so a way for the agent to remember X number of games and then replay it back for training
class ReplayBuffer():

    def __init__(self,input_dim, size, batch_size=32):
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
    
    def insert(self,state,action,reward,next_state,done):
        self.state_storage[self.ptr] = state 
        self.next_state_storage[self.ptr] = next_state
        self.actions_storage[self.ptr] = action
        self.rewards_storage[self.ptr] = reward
        self.done_storage[self.ptr] = done
        #make sure it loops back around once reach size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size +1 ,self.max_size)
    
    #return the stored N-step transitions
    def sample_batch(self):
        indxs = np.random.choice(self.size,size=self.batch_size,replace=False)#make sure same item cant be selected twice
        return dict(states=self.state_storage[indxs],
                    next_states=self.next_state_storage[indxs],
                    actions=self.actions_storage[indxs],
                    rewards=self.rewards_storage[indxs],
                    dones=self.done_storage[indxs]
                    )

    #return number of elements stored
    def __len__(self):
        return self.size

    def can_sample(self):
        #need enough varied data to sample, as we sample random data
        return self.size >= (self.batch_size * 5)

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
    def __init__(self,
                 env,
                 input_dims,
                 device="cpu",
                 epsilon=1.0,
                 min_epsilon=0.1,
                 nb_warmup=250_000,
                 nb_actions=5,
                 memory_capacity=50_000,
                 batch_size=32,
                 learning_rate=0.00020,
                 gamma=0.99,
                 sync_network_rate=10_000,
                 use_vit=False
                 ):

        self.env = env

        #initialise the models
        if(use_vit):
            self.model = MarioNet_ViT(nb_actions=nb_actions,device=device) #5 actions for agent can do in this game
        else:
            self.model = MarioNet(input_dims,nb_actions=nb_actions,device=device)
        self.target_model = copy.deepcopy(self.model).eval() #when we work with q learning, want a model and another model we can evaluate of off. Part of Dueling deep Q
        
        """
        if os.path.exists("models"):
            self.model.load_model(device=device)
            self.target_model.load_model(device=device)"
        """
        
        self.device = device
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.nb_actions = nb_actions

        #hyperparameters for DQN
        self.learning_rate = learning_rate
        self.nb_warmup = nb_warmup
        self.gamma = gamma #how much we discount future rewards compared to immediate rewards
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        #epsilon hyper parameters - update epsilon at every time step.
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.init_epsilon = 1.0
        self.epsilon_decay = 0.99999975#1- (((epsilon - min_epsilon) / nb_warmup) *2) # linear decay rate, close to the nb_warmup steps count

        self.game_steps = 0 #track how many steps taken over entire training
        
        #Combines adaptive learning rates with weight decay regularisation for better generalisation
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

        #simple uniform sampling memory        
        self.memory = ReplayBuffer(input_dims,memory_capacity,batch_size)
        
        print_info()#get device information printed

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

    def decay_epsilon(self,episode=None):
        # linear decay : 
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def sync_networks(self):
        if self.game_steps % self.sync_network_rate == 0 and self.game_steps > 0:
            self.target_model.load_state_dict(self.model.state_dict()) #keep the target_model lined up with main model, its learning in hops

    #set the tensorboard writer
    def set_writer(self,writer):
        self.writer = writer
    
    #epochs = how many iterations to train for
    def train(self, epochs):

        for epoch in range(1,epochs+1):
            state = self.env.reset() #reset the environment for each iteration
            done = False
            ep_return = 0
            ep_loss = 0
            loss = 0
            loss_count = 0

            while not done:
                action = self.get_action(state)

                self.game_steps += 1
                next_state,reward,done,info = self.env.step(action)
                #order of list matters
                self.memory.insert(state, action, reward, next_state, done)

                #actual training part
                #can take out of memory only if sufficient size
                if self.memory.can_sample():    
                    self.optimizer.zero_grad()

                    self.sync_networks()

                    samples = self.memory.sample_batch()
                    #now convert into tensors for training
                    states = torch.tensor(samples["states"], dtype=torch.float32, device=self.device)
                    next_states = torch.tensor(samples["next_states"], dtype=torch.float32, device=self.device)
                    actions = torch.tensor(samples["actions"], dtype=torch.long, device=self.device) #long tensor for actions
                    rewards = torch.tensor(samples["rewards"], dtype=torch.float32, device=self.device)
                    dones = torch.tensor(samples["dones"], dtype=torch.float32, device=self.device)


                    qsa_b = self.model(states)  # Shape: (batch_size, n_actions) as network estimates q value for all actions. so have rows of q values for each action
                    qsa_b = qsa_b[np.arange(self.batch_size), actions.squeeze()] #action contains the actual actions taken, remove extra batch dimension via squeeze
                    # then generate an array of batch indices. So select q value of each action taken

                    # DDQN - Compute target Q-values from the online network, then use the 
                    # target network to evaluate
                    best_next_actions = self.model(next_states).argmax(dim=1) #get the best action using max of dim=1(which are the actions). argmax return indices
                    #this feeds the next_states into target_model and then selects its own values of the actions that online model chose
                    next_qsa_b = self.target_model(next_states).detach()[np.arange(self.batch_size),best_next_actions]
                    
                    # dqn = r + gamma * max Q(s,a)
                    # ddqn = r + gamma * online_network(s',argmax target_network_Q(s',a'))
                    #detach -> important as we dont want to back propagate on target network
                    target_b = (rewards + self.gamma * next_qsa_b * (1 - dones.float()) ) #1-dones.float() -> stop propagating when finished episode
                    
                    #indices = samples["index"]

                    loss = self.loss(qsa_b,target_b)
                    loss.backward()
                    ep_loss += loss.item()
                    loss_count += 1
                    self.optimizer.step()
                    self.decay_epsilon() #decay epsilon at each step in environment

                if "episode" in info:
                    self.writer.add_scalar("Charts/episodic_return", info["episode"]["r"], self.game_steps) 
                    self.writer.add_scalar("Charts/episodic_length", info["episode"]["l"], self.game_steps)
                    # episodic length(number of steps)
                        
                state = next_state #did the training, now move on with next state
                ep_return += reward
                #print(f"Got here now, episode return={ep_return}, time step = {self.game_steps}")

            #gatherin stats
            if epoch % 10 == 0:
                print("")
                if loss_count > 0:
                    print(f"Episode loss = {ep_loss}, Average loss = {ep_loss/loss_count}, Epoch = {epoch}, \
                        Time Steps = {self.game_steps}, epsilon = {self.epsilon}")
                print("")
            if epoch % 100 == 0:
                self.model.save_model() #save model every 10th epoch
            
            if epoch % 1000 == 0:
                self.model.save_model(f"models/dqn_iter_{epoch}.pt") #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed

            self.writer.add_scalar("Charts/epochs",epoch,self.game_steps)
            self.writer.add_scalar("Charts/epsilon",self.epsilon,self.game_steps)

            if loss > 0 or loss_count > 0:
                #average loss - more representitive
            
                self.writer.add_scalar("losses/loss_episodic",ep_loss/loss_count,self.game_steps)

                #last loss - up to date changes shown
                self.writer.add_scalar("losses/loss",loss.item(),self.game_steps)

        self.env.close()
        self.writer.close()
        self.model.save_model()

    #run something on the machine, and see how we perform
    def test(self):
        
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
 