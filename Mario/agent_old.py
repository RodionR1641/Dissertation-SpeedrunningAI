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
        self.model.to(device)
        self.target_model.to(device)

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
        self.sync_network_rate = 10000
        self.game_steps = 0 #track how many steps taken over entire training
        
        
        #Combines adaptive learning rates with weight decay regularisation for better generalisation
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()
        
        #sampler = PrioritizedSampler(max_capacity=memory_capacity, priority_key="td_error")

        storage = LazyMemmapStorage(memory_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)
        
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

    #adjust the priorities in the buffer
    def update_priorities(self,indices, td_errors):
        priorities = (td_errors + 1e-5).pow(self.alpha)  # Ensure nonzero priority with epsilon, the alpha control how much priorities matter
        # if alpha 0 -> all experiences sampled uniformly, if alpha = 1 -> experiences sample fully by priority
        self.replay_buffer.update_priority(indices, priorities)

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

                self.store_memory(state,action,reward,next_state,done)#record both the previous and next_state

                #actual training part
                #can take out of memory only if sufficient size
                if len(self.replay_buffer) >= (self.batch_size * 10):
                    
                    self.optimizer.zero_grad()

                    self.sync_networks()

                    samples = self.replay_buffer.sample(self.batch_size).to(self.model.device)

                    keys = ("state","action","reward","next_state","done")

                    states, actions, rewards, next_states, dones = [samples[key] for key in keys]
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

                    loss = self.loss(qsa_b,target_b)
                    loss.backward()
                    ep_loss += loss.item()
                    self.optimizer.step()
                    self.decay_epsilon() #decay epsilon at each step in environment

                state = next_state #did the training, now move on with next state
                ep_return += reward

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
    