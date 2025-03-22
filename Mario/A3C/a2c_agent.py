import torch
from a2c_model import ActorCritic
from torch.distributions import Categorical # taking probability from network and map it to distribution for us
import torch.optim as optim
import torch.nn.functional as F

class Agent:
    def __init__(self,input_shape,lr_rate=1e-5,device="cpu", gamma=0.99, n_actions=5,num_envs=8):
        self.gamma=gamma #used for future rewards
        self.n_actions = n_actions
        self.action = None
        self.action = None #keep track of the last action took, used for loss function
        self.lr_rate = lr_rate

        self.actor_critic = ActorCritic(input_shape=input_shape,n_actions=n_actions,device=device)
        self.optimiser = optim.Adam(self.actor_critic.parameters(), lr=self.lr_rate)
        self.num_envs = num_envs

        self.log_probs = None
        self.device = device


    def choose_action_entropy(self,states):
        probabilities, state_values = self.actor_critic(states) #dont need value, just actor actions
        
        #probabilities = F.softmax(probabilities,dim=-1) #get the softmax activation, for probability distribution
        action_probabilities  = Categorical(logits=probabilities) #feed into categorical distribution
        action = action_probabilities.sample() #sample the distribution    
        
        log_probs = action_probabilities.log_prob(action)#use logarithmic probabilities for stability as probabilities can
        entropy = action_probabilities.entropy()
        #get small. derivatives of logs are also simpler to compute anyway
        self.log_probs = log_probs

        return (action.numpy(), log_probs, state_values, entropy)
        #return a numpy version of the action as action is a tensor, but openai gym needs numpy arrays.

    def save_models(self,weights_filename="models/a2c_latest.pt"):
        print("... saving models ...")
        self.actor_critic.save_model(weights_filename=weights_filename)

    def load_models(self,weigts_filename="models/a2c_latest.pt",device="cpu"):
        print("... loading models ...")
        self.actor_critic.load_model(weights_filename=weigts_filename,device=device)

    """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438) """

    def get_losses(self,
                   rewards, #shape of [n_steps_per_update,...]
                   action_probs, #tensor with log_probs of actions taking at each time step
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
        actor_loss = (
            -(advantages.detach() * action_probs).mean() - ent_coef * entropy.mean() #includes the entropy loss already
        )
        return (critic_loss, actor_loss)

    def update_params(self,actor_loss,critic_loss):
        loss = critic_loss + actor_loss #combine the loss

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()
