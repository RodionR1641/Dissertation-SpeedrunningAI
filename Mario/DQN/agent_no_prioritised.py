import torch
import copy
import torch.optim as optim
import numpy as np
import time
from model import MarioNet
from model_mobile_vit import MarioNet_ViT
import os
from gym.wrappers import RecordVideo
import wandb
from torch.nn.utils import clip_grad_norm_

# Define log file name (per process)
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
load_models_flag = True # flag to see if we want to load an existing model and continue training, or train a new one

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
                 sync_network_rate=1_000,
                 use_vit=False
                 ):

        self.env = env
        self.device = device

        #initialise the models
        if(use_vit):
            self.model = MarioNet_ViT(nb_actions=nb_actions,device=device) #5 actions for agent can do in this game
        else:
            self.model = MarioNet(input_dims,nb_actions=nb_actions,device=device)
        self.target_model = copy.deepcopy(self.model).eval() #when we work with q learning, want a model and another model we can evaluate of off. Part of Dueling deep Q

        #model default value - if we already had a model, then use the epochs we left off  
        self.curr_epoch = 1

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
        self.num_completed_episodes = 0#how many games have ended in getting the flag
        self.best_time_episode = 1e9 #best time of an episode in speedrunning

        self.loss = torch.nn.MSELoss()

        #simple uniform sampling memory        
        self.memory = ReplayBuffer(input_dims,memory_capacity,batch_size)

        #Combines adaptive learning rates with weight decay regularisation for better generalisation
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        #load models
        if os.path.exists("models/dqn") and load_models_flag==True:
            self.load_models()
        
        self.epoch = self.curr_epoch #track the current epoch for video naming
        
        self.model.to(self.device)
        self.target_model.to(self.device)
        print(f"Device for Agent = {device}")
        
        print_info()#get device information printed

    def record_video(self,run_name):
        self.env = RecordVideo(self.env,"videos/DQN",name_prefix=f"{run_name}_{self.epoch}"
                          ,episode_trigger=lambda x: x % 500 == 0)  # Record every 500th episode

    #state is image of our environment
    def get_action(self,state,test=False):

        #only use random action if its training
        if (torch.rand(1) < self.epsilon) and not test: #if a random number between 0 and 1 is smaller than epsilon, do random move
            #randint returns a tensor
            return np.random.randint(self.nb_actions) #random action
        else:

            #convert state into np_array for calculations, then make a tensor, then unsqueese to add batch dimension
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.model.device)
            #use advantage function to calculate max action
            return self.model(state).argmax().item()

    def decay_epsilon(self):
        # linear decay : 
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def sync_networks(self):
        if self.game_steps % self.sync_network_rate == 0 and self.game_steps > 0:
            self.target_model.load_state_dict(self.model.state_dict()) #keep the target_model lined up with main model, its learning in hops
            print(f"synced target and online networks, current game step = {self.game_steps}")
    
    #epochs = how many iterations to train for
    def train(self, epochs):
        
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
            loss = 0
            loss_count = 0
            self.epoch = epoch #make sure it is tracked for video file naming
            episodes = epoch

            while not done:
                action = self.get_action(state)

                self.game_steps += 1

                #track how many steps taken in given time window for SPS
                step_count_window += 1
                current_time = time.time()
                if current_time - last_time >= window_sec:
                    sps = step_count_window / (current_time - last_time)
                    step_count_window = 0
                    last_time = current_time

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
                    qsa_b = qsa_b[np.arange(self.batch_size), actions.squeeze()] #action contains the actual actions taken,
                    # remove extra batch dimension via squeeze then generate an array of batch indices.
                    # So select q value of each action taken

                    # DDQN - Compute target Q-values from the online network, then use the 
                    # target network to evaluate. No gradient computation here
                    
                    best_next_actions = self.model(next_states).argmax(dim=1) #get the best action using max of dim=1
                    #(which are the actions). argmax return indices this feeds the next_states into target_model 
                    # and then selects its own values of the actions that online model chose
                    next_qsa_b = self.target_model(next_states).detach()[np.arange(self.batch_size),best_next_actions]
                    
                    # dqn = r + gamma * max Q(s,a)
                    # ddqn = r + gamma * online_network(s',argmax target_network_Q(s',a'))
                    #detach -> important as we dont want to back propagate on target network
                    target_b = (rewards + self.gamma * next_qsa_b * (1 - dones.float()) )
                         #1-dones.float() -> stop propagating when finished episode
                    
                    loss = self.loss(qsa_b,target_b)
                    td_errors = (target_b - qsa_b).abs() #absolute TD error

                    loss.backward()
                    ep_loss += loss.item()
                    loss_count += 1

                    if self.game_steps % 500 == 0: #only track periodically for efficiency reasons
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
                            # Base metrics
                            log_data = {
                                "game_steps": self.game_steps,
                                "DQN/TD_Error_Avg": td_errors.mean().item(),
                                "DQN/TD_Error_Max": td_errors.max().item(),
                                "DQN/Q_Value_Avg": qsa_b.mean().item(),
                                "DQN/Q_Value_Max": qsa_b.max().item(),
                                "Gradient/gradient_norm_total": total_norm,
                                **{f"Gradient/gradients/gradient_{name}": norm for name, norm in layer_norms.items()}
                            }

                            # Action-specific Q-values (less frequent to reduce overhead)
                            if self.game_steps % 2500 == 0:  # Every 5x gradient logging
                                rand_state = states[0:1]  # Use already-loaded state
                                action_qs = self.model(rand_state)
                                for action in range(action_qs.shape[1]):
                                    log_data[f"DQN_Action/Q_Action_{action}"] = action_qs[0, action].item()

                        wandb.log(log_data, commit=False)
                    #clip gradient after the graphs as the graphs need to show the real gradient
                    clip_grad_norm_(self.model.parameters(),10.0) #prevent exploding gradient

                    self.optimizer.step()
                    self.decay_epsilon() #decay epsilon at each step in environment

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
                        },commit=False)

                        if info["time"] < self.best_time_episode:
                            #find the previous file with this old best time
                            filename = f"models/dqn/best_{self.best_time_episode}.pth"
                            new_filename = f"models/dqn/best_{info['time']}.pth"

                            #rename so that not saving a new file for each new time
                            if os.path.exists(filename):
                                os.rename(filename,new_filename)
                            
                            #save this model that gave best time, if the model didnt exist then its just created
                            self.best_time_episode = info["time"]
                            self.save_models(epoch,new_filename)
                
             
            #gatherin stats
            if epoch % 10 == 0:
                print("")
                if loss_count > 0:
                    print(f"Episode return = {episodic_return}, Episode len = {episodic_len},  \
                        Episode loss = {ep_loss}, Average loss = {ep_loss/loss_count}, Epoch = {epoch}, \
                        Time Steps = {self.game_steps}, epsilon = {self.epsilon}, \
                        SPS = {sps}")
                print("")
            if epoch % 10 == 0:
                self.save_models(epoch=epoch) #save models every 100th epoch
            
            if epoch % 1000 == 0:
                self.save_models(epoch=epoch,weights_filename=f"models/dqn/dqn_iter_{epoch}.pth") #saving the models, may see where the good performance was and then it might tank -> can copy
                #this in as the main model. Then can start retraining from this point if needed


            wandb.log({
                "game_steps": self.game_steps,
                "episodes": episodes,
                "Charts/epochs": epoch,
                "Charts/epsilon": self.epsilon,
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


        self.save_models(epoch=epoch)
        self.env.close()
        wandb.finish()

    #run something on the machine, and see how we perform
    #models already loaded when this is run
    def test(self):
        normal_env = self.env

        state = self.env.reset()
        done = False
        
        while not done:
            time.sleep(0.01) #by default it runs very quickly, so slow down
            action = self.get_action(state,test=True) #dont want epsilon
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
    def save_models(self,epoch, weights_filename="models/dqn/dqn_latest.pth"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,  # Save the current epsilon value
            'epoch': epoch,      # Save the current epoch
            'game_steps': self.game_steps,  # Save the global step
            'num_completed_episodes' : self.num_completed_episodes, # num of epochs where the game flag was received
            'best_time_episode': self.best_time_episode
        }

        print("...saving checkpoint...")
        if not os.path.exists("models/dqn"):
            os.makedirs("models/dqn",exist_ok=True)
        torch.save(checkpoint,weights_filename)
    
    #if model doesnt exist, we just have a random model
    def load_models(self, weights_filename="models/dqn/dqn_latest.pth"):
        try:

            checkpoint = torch.load(weights_filename)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.curr_epoch = checkpoint["epoch"]
            self.epsilon = checkpoint["epsilon"]
            self.game_steps = checkpoint["game_steps"]
            self.num_completed_episodes = checkpoint["num_completed_episodes"]
            self.best_time_episode = checkpoint["best_time_episode"]

            self.model.to(self.device)
            self.target_model.to(self.device)

            print(f"Loaded weights filename: {weights_filename}, curr_epoch = {self.curr_epoch}, epsilon = {self.epsilon}, \
                  game steps = {self.game_steps}")
                        
        except Exception as e:
            print(f"No weights filename: {weights_filename}, using a random initialised model")
            print(f"Error: {e}")