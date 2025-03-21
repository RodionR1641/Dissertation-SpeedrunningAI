import argparse
import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import RecordVideo
from mario import Mario
from model import MarioNet
import wandb
from wandb.integration.tensorboard import patch
import datetime

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="PPO_experiment",
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="SuperMarioBros-1-1-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=777,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, # 10 million timesteps
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, #reproduce experiments
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    #
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="RL Mario",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    
    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8, #number of sub environments in a vector environment
        help="the number of parallel game environments")
    
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout") # control the number of data to collect for EACH policy rollout
    # total data = num_steps * num_envs . This is our BATCH_SIZE
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy") # ///
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.05, #0.05 is quite lenient
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # the total batch size for learning, split into minibatches
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # minibatches used for training
    return args



def make_env(gym_id,seed,environment_num,cap_video,name):
    def one_env():
        env = Mario(device=device,env_id=gym_id,seed=seed)
        if(cap_video):
            if environment_num == 0:
                env = RecordVideo(env,f"videos/{name}")
        return env    
    return one_env

#seed setup for reproducibility
def seed_run():
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(args.seed)
        #torch.backends.cudnn.benchmark = False # could be useful for reproducibility, but setting to False affects performance
        torch.backends.cudnn.deterministic = args.torch_deterministic
    
    np.random.seed(args.seed)
    random.seed(args.seed)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.track:
        #wanbd allows to track info related to our experiment on the cloud
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=False, #monitors videos, but for old gym. Doesnt work now
            save_code=True
        )

    patch() #make sure tensorboard graphs are saved to wandb
    #visualisation toolkit to visualise training - Tensorboard, allows to see the metrics like loss and see hyperparameters
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    #seeding
    seed_run()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device PPO: ",device)
    #Vectorised environment - run N environments. Envs is a synchronous interface that outputs a batch of N observations from
    #N environments. The done flags then become a list for each env being done or not. Then have rollout and learn phase
    #rollout - sample actions from N environments and step for M steps. if env done, just set the done flag but can restart and continue going collecitng data
    #learning - learn from data in rollout phase, calculate advantages and returns. Learn from [data,advantages,returns] which
    # are the fixed length trajectory segments
    # next_done tells if next_obs is actually the first observation of a new episode. PPO can still learn even if sub env never
    # terminate or truncate
    # so at end of j-th rollout phase, next_obs can be used to estimate the value of the final state during learning phase and
    # the beginning of the j+1th rollout phase, next_obs becomes the initial observation in data
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )#vectorised environment

    assert isinstance(envs.single_action_space, gym.spaces.Discrete) #only for discrete actions here

    ac_model = MarioNet(envs,input_shape=envs.envs[0].observation_space.shape,device=device) #actor critic model.

    optimizer = optim.Adam(ac_model.parameters(), lr=args.learning_rate, eps=1e-5) #epsilon decay of 1e-5 for PPO

    """
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, done, info = envs.step(action) # once the episode finishes -> discard the current observation and return initial observation of next episode
        for item in info:
            if "episode" in item.keys():
                print(f"episodic_return {item['episode']['r']}")
                # the vector environments auto reset
    """
                
    # Storage setup - shape is to match num_steps * num_envs size
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    #track number of environment steps
    global_step = 0
    start_time = time.time() #helps to calculate fps later
    next_obs = torch.Tensor(envs.reset()).to(device) #store initial and subsequent observations
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size # e.g. 25k/512, this is how many iterations of updating training we have
    
    print(f"num_updates = {num_updates}")

    print("next_obs.shape",next_obs.shape)
    print("ac_model.get_value(next_obs)",ac_model.get_value(next_obs))
    print("ac_model.get_value(next_obs).shape",ac_model.get_value(next_obs).shape)
    print()
    print("ac_model.get_action_plus_value",ac_model.get_action_plus_value(next_obs))


    # main loop -> the learning rate is annealed with each update
    # have a rollout phase and learning phase
    # rollout - interact with the environment for a fixed number of steps to collect trajectories. Stores observation and dones
    # gets actions and values using actor critic to sample actions and compute values, step the environment
    # have a GAE - used to compute Advantages, measure how much better or worse an action is compared to average
    # learning phase - use a clipped surrogate objective to ensure policy does not change too drastically in singular update. 
    # also have a Value loss as MSE between predicted and target values, also gets clipped to prevent large updates. Also have
    # entropy loss, which encourages exploration. Total loss includes all of these together. We also stop early if the approximate
    # KL divergence exceeds a thershold, this is to prevent the policy changing too drastically
    #each update is one iteration of the training loop
    for update in range(1,num_updates+1):
        loss_total = 0
        v_loss_total = 0
        pg_loss_total = 0
        entropy_loss_total = 0 
        loss_count = 0

        #learning rate annealing - the learning rate of adam decays linearly. Papers show this annealing allows agents to obtain
        # higher episodic return

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # linearly decrease to 0 as update increases to num_updates
            curr_lr = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = curr_lr
        
        #policy rollout is itself a loop inside the training process, run for num_steps
        # at each step: observe current state,take action based on current policy, receive reward and next state
        # and store collected data for training
        for step in range(0,args.num_steps):
            global_step += 1 * args.num_envs # doing steps for all the envs, so add that many steps
            #store next observation and dones
            obs[step] = next_obs# current observation
            dones[step] = next_done #tell if episode terminated or not

            #during roll out, dont catch any gradient
            with torch.no_grad():
                action,logprob,_,value = ac_model.get_action_plus_value(next_obs) # actor critic takes current state
                # and gives back action, logprob of said action and critics estimate of value
                values[step] = value.flatten() #1d tensor for value
            #store action and log probs
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done, info = envs.step(action.cpu().numpy()) #action on cpu, converted to numpy as env expects it to be
            #store rewards and update variables, make sure tensors
            rewards[step] = torch.tensor(reward).to(device).view(-1)# to gpu
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)   #reassign variables

            #log episodic return and info, this does it on vectorised envs 
            for item in info:
                if "episode" in item.keys():#check if current env completed episode
                    episodic_reward = item["episode"]["r"]
                    episodic_len = item["episode"]["l"]

                    print(f"global_step={global_step}, episodic_return={episodic_reward}, episodic len={episodic_len}")

                    writer.add_scalar("Charts/episodic_return", episodic_reward, global_step) 
                    writer.add_scalar("Charts/episodic_length", episodic_len, global_step)
                    # episodic length(number of steps)
                    break
            
        # use General Advantage Estimation(GAE) to do advantage estimation

        #PPO bootstraps values if environments are not done. The values of next observations are estimated as the end of rollout values
        
        # compute the advantages and returns for the collected trajectories using either GAE or simpler discounted return method
        # need these values for updating policy and value networks in the PPO algorithm
        # this is part of learning now, getting advantages and returns(discounted sum of future rewards, used as target for value
        # function)

        #GAE provides a better estimate of advantage, as it balances bias and variance
        with torch.no_grad(): # part of preprocessing state, not doing loss yet
            next_value = ac_model.get_value(next_obs).reshape(1, -1) # critic value estimates the value of next observation
            #Generalised Advantage Estimation way - get a better estimate of the advantage by balancing bias and variance 
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device) #store advantages for each step
                lastgaelam = 0 #running sum of GAE terms
                #loop over steps in reverse to compute advantages using the reverse GAE formula
                # At = delta_t + (gamma * lambda) At+1. Delta_t = TD error at time t, gamma is discount factor and lambda
                # is the GAE paramater that controls the bias variance trade off
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done #binary flag indicating if next state is non-terminal(not ended)
                        nextvalues = next_value #values of next state
                    else:
                        nextnonterminal = 1.0 - dones[t + 1] #done at time t+1
                        nextvalues = values[t + 1] #next value at time t+1
                    #TD error at time t
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    #advantage at time t, computed using At GAE formula
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                #returns are just sum of both advantages and values
                returns = advantages + values 
                # different way to do returns, as the other way is sum of discounted rewards
                #  i.e. returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            else:
                #simpler discounted sum of future rewards way. Rt = rt + gamma* Rt+1
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    #return at time t, simple TD calculation
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                #At = Rt - V(st)
                advantages = returns - values
            
        # flatten the batch and store it corresponding to the number 
        # of steps in the environment. Need it as we break these batches into minibatches for training
        # so e.g. had 4*128 = 512 for total number in batch, then divide that by 4 to get a 128 minibatch. initial shape
        # may be like (128,4....)
        # we flatten these tensors to make it easier to work with them, single large batch. First dimension is num_steps*num_envs
        # this simplifies then splitting data into minibatches
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #get the minibatch now. Acquire all the indices of entire batch, and for each update_epoch -> shuffle these indices.
        # Then loop through the entire batch, one minibatch at a time(e.g. 128 in each one)
        # indices allow model to see varied data and avoid overfitting and give generalisation
        b_inds = np.arange(args.batch_size)
        clipfracs = [] # another debug variable -> measure how often the clip objective is actually triggered

        #Learning Phase - optimising the policy and value networks here
        # in each epoch, the entire batch of data is processed in minibatches
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)#shuffle batch - see the data in different order each time, training stability and generalisation
            #steps of minibatches, so get all the elements from the batch, but treat data in minibatches
            # training on entire batch can be comp expensive and lead to slower convergence. These allow for more frequent
            # updates to model parameters. Also introduce some noise
            for start in range(0, args.batch_size, args.minibatch_size): 
                end = start + args.minibatch_size #just the maths on calculating end of minibatch
                mb_inds = b_inds[start:end] #get the minibatch indices

                #do a forward pass on a minibatch of observations. Get newlogprob which is log prob of actions under current
                # policy, entropy which is measure of randomness/exploration in distribution and newvalue critic estimate
                # pass actions taken during rollout phase, evaluate the same actions under the updated policy which is necessary
                # for the computing policy loss
                _, newlogprob, entropy, newvalue = ac_model.get_action_plus_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds] #pass the minibatch actions, so that agent doesnt sample any new actions
                )
                #do logarithmic substraction between new log probabilities and old ones in policy rollout phase
                logratio = newlogprob - b_logprobs[mb_inds]
                #exponentiation of logration, which gives ratio of probabilities. ratio = new_prob/old_prob
                #it measures how much the policy has changed for the given actions. if its close to 1, then policy hasnt changed
                #much, and if its far from 1 than it has. Use it to make sure policy doesnt change too drastically
                ratio = logratio.exp()# at the first minibatch the ratio should be =1 as we havent modified the parameters yet

                #calculate approximate KL divergence here, used for monitoring
                # KL divergence => measures the difference between 2 probability distributions. PPO ensures policy doesnt 
                # change too much
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean() #debug variable -> helps understand how aggressively policy updates
                    #however, the following approximation is a better estimate
                    approx_kl = ((ratio -1) - logratio).mean()
                    #measure how often the PPO clipping mechanism is triggered. If ratio deviates too much from 1 outside the 
                    #clipping range [1-clip_coef,1+clip_coef] the mechanism is activated
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # PPO does advantage normalisation - substract their mean and divide by their standard deviation. zero mean and
                # unit variance, can help stabilise training. prevent them being too large or small
                # This happens at minibatch level. Dosnt affect performance much
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) #substract by mean and divide by st deviation. add a small scalar to make sure not divide by 0
                
                # PPO uses a Clipped Surrogate Objective - outperforms vanilla policy gradient
                # Polisy loss here. Ensure policy doesnt change too much
                pg_loss1 = -mb_advantages * ratio #unclipped policy loss
                #clipped policy loss here, ratio is constrained to lie within [1 - clip_coef, 1 + clip_coef]
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                #final policy is the mean of element wise maximum of pgloss1 and pgloss 2. pgloss2 is used if pgloss1 changed too fast
                pg_loss = torch.max(pg_loss1,pg_loss2).mean() # we are doing the max of negatives here, 
                #whilst the paper did min of positives. But its the same thing

                # PPO also does value loss clipping, similar to clipped surrogate objective. 
                # this doesnt really improve performance however, but useful for reproducibility of original paper

                # PPO minimises this loss:
                # L_V = max [ (V_theta_t - V_targ)^2, (clip(V_theta_t, V_theta_t-1 - epsilon, V_theta_t-1 + epsilon) - V_targ) ^ 2]
                # value loss:
                # ensure the critic prediction values(newvalue) are as close as possible to the actual returns(b_returns)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # clipped similarly to policy loss. Prevents value function changing too fast
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() 
                    # normally, the loss values is just MSE between predicted and emperical

                ## PPO also include entropy loss in its overall loss
                # entropy - measure of "chaos" in the action probability distribution. 
                # Maximising entropy encourages the agent to explore more

                entropy_loss = entropy.mean() #mean entropy over minibatch
                #this is the overall loss we have 
                # we want to minimise the policy loss and the value loss, and maximise the entropy loss
                # have hyperparameter controlling weight of entropy(higher -> more ephasis on exploration) 
                # and value loss(higher value places more emphasis on accurate value predictions)
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                loss_total += loss.item()
                pg_loss_total += pg_loss.item()
                v_loss_total += v_loss.item()
                entropy_loss_total += entropy_loss.item()
                loss_count += 1

                #backpropagation and optimising now
                optimizer.zero_grad()
                loss.backward()
                # PPO implements global gradient clipping. We set up a maximum gradient norm. Prevent gradients from exploding
                nn.utils.clip_grad_norm_(ac_model.parameters(), args.max_grad_norm)
                optimizer.step()

            #early stopping: helps to prevent policy changing too much in a single update. IF KL divergence extends threshold
            # the training loop is stopped to prevent unstable training
            # we implement it at batch level, can also do at minibatch level                
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        
        #logged output saved in wandb 
        if update % 10 == 0:
            print("")
            print(f"Loss = {str(loss.item())}") 
            print(f"Episodic Average Loss = {str(loss_total/loss_count)}")
            print(f"Time Step = {str(global_step)}")
            print("Last Reward = " + ', '.join(map(str, reward.flatten()))) 
            print("")
            ac_model.save_model()  
        
        if update % 1000 == 0:
            ac_model.save_model(f"models/ppo_iter_{update}.pt")

        #debug variable:explained variance - indicate if the value function is a good indicator of the returns ///
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        #measure stats
        writer.add_scalar("Charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("Charts/epochs", update,global_step)
        
        #add average loss, more representitive
        writer.add_scalar("losses/loss_episodic", loss_total/loss_count, global_step)
        writer.add_scalar("losses/value_loss_episodic", v_loss_total/loss_count, global_step)
        writer.add_scalar("losses/policy_loss_episodic", pg_loss_total/loss_count, global_step)
        writer.add_scalar("losses/entropy_episodic", entropy_loss_total/loss_count, global_step)

        #add last loss, useful for tracking quick changes
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("Charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    ac_model.save_model()
    envs.close()
    writer.close()