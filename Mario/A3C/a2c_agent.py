import torch
from a2c_model import ActorCritic
from torch.distributions import Categorical # taking probability from network and map it to distribution for us
import torch.optim as optim
import os

class Agent:
    def __init__(self,input_shape,lr_rate=1e-5,device="cpu", gamma=0.99, n_actions=5,num_envs=8):
        self.gamma=gamma #used for future rewards
        self.n_actions = n_actions
        self.action = None
        self.action = None #keep track of the last action took, used for loss function
        self.lr_rate = lr_rate

        self.actor_critic = ActorCritic(input_shape=input_shape,n_actions=n_actions,device=device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr_rate)
        self.num_envs = num_envs

        self.log_probs = None
        self.game_steps = 0
        self.num_completed_episodes = 0#how many games have ended in getting the flag
        self.total_episodes = 1 #total number of epochs/episodes of game playing that happened
        self.curr_epoch = 1 #what the current epoch is
        self.entropy = 0
        self.best_time_episode = 1e9
        self.device = device
        print(f"Device for Agent = {device}")


    def choose_action_entropy(self,states):
        probabilities, state_values = self.actor_critic(states) #dont need value, just actor actions
        
        #probabilities = F.softmax(probabilities,dim=-1) #get the softmax activation, for probability distribution
        action_probabilities  = Categorical(logits=probabilities) #feed into categorical distribution
        action = action_probabilities.sample() #sample the distribution    
        
        log_probs = action_probabilities.log_prob(action)#use logarithmic probabilities for stability as probabilities can
        entropy = action_probabilities.entropy()
        #get small. derivatives of logs are also simpler to compute anyway
        self.log_probs = log_probs
        self.entropy = entropy

        return (action.cpu().numpy(), log_probs, state_values, entropy)
        #return a numpy version of the action as action is a tensor, but openai gym needs numpy arrays.

    """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438) """

    def get_losses(self,
                   rewards, #shape of [n_steps_per_update,...]
                   action_log_probs, #tensor with log_probs of actions taking at each time step
                   value_pred, #tensor with state value predictions
                   entropy,
                   masks, #tensor with masks for each time step in episode
                   gamma, #discount factor
                   lam, #GAE hyperparameter. lam=1 is Monte Carlo sampling with high variance and no bias, lam=0 is normal TD
                   #learning that has low variance but is biased
                   ent_coef):
        
        T = len(rewards)
        advantages = torch.zeros(T, self.num_envs, device=self.device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_pred[t + 1] - value_pred[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration

        entropy_loss = entropy.mean()

        actor_loss = (
            #advantages detached as dont want to back propagate on them, treated as constants
            #entropy regularisation at the end
            #policy gradient in first part - negated as we want gradient ascent as we want to maximise the expected cumulative reward
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy_loss #entropy maximised
        )
        return (critic_loss, actor_loss, entropy_loss)

    def update_params(self,actor_loss,critic_loss):
        loss = critic_loss + actor_loss #combine the loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), critic_loss.item(), actor_loss.item()
    

    #these models take a while to train, want to save it and reload on start. Save both target and online for exact reproducibility
    def save_models(self,epoch, weights_filename="models/a2c/a2c_latest.pth"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done

        checkpoint = {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,      # Save the current epoch
            'game_steps': self.game_steps,  # Save the global step
            'completed_episodes' : self.num_completed_episodes, # num of epochs where flag is gotten
            'total_episodes': self.total_episodes, #total number of completed epochs/episodes
            'best_time_episode': self.best_time_episode
        }

        print("...saving checkpoint...")
        if not os.path.exists("models/ppo"):
            os.makedirs("models/ppo",exist_ok=True)
        torch.save(checkpoint,weights_filename)

    #if model doesnt exist, we just have a random model
    def load_models(self, weights_filename="models/a2c/a2c_latest.pth"):
        try:

            checkpoint = torch.load(weights_filename)
            self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.curr_epoch = checkpoint["epoch"]
            self.game_steps = checkpoint["game_steps"]
            self.num_completed_episodes = checkpoint["completed_episodes"]  
            self.total_episodes = checkpoint["total_episodes"]
            self.best_time_episode = checkpoint["best_time_episode"]
            
            self.actor_critic.to(self.device)

            print(f"Loaded weights filename: {weights_filename}, curr_epoch = {self.curr_epoch}, \
                  game steps = {self.game_steps}")               
        except Exception as e:
            print(f"No weights filename: {weights_filename}, using a random initialised model")
            print(f"Error: {e}")
