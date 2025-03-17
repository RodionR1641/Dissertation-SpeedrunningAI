import argparse
import os
import random
import time
from distutils.util import strtobool
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import RecordVideo
from mario import Mario
from model import MarioNet
from model_RND import RNDNetwork

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="SuperMarioBros-1-1-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, # 10 million timesteps
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, #reproduce experiments
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    #
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
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
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--rand-net-dist",type=bool,default=False,
        help="random network distillation to calculate intrinsic rewards that help explore novel states")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # the total batch size for learning, split into minibatches
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # minibatches used for training
    # fmt: on
    return args



def make_env(gym_id,seed,environment_num,cap_video,name):
    def one_env():
        env = Mario(device=device,env_id=gym_id,seed=seed)
        if(cap_video):
            if environment_num == 0:
                env = RecordVideo(env,f"videos/{name}")
        return env    
    return one_env

#use to compute the novelty reward
def compute_intrinsic_reward(obs):
    with torch.no_grad():
        target_features = rnd_target(obs)  # Fixed target network
    predicted_features = rnd_predictor(obs)  
    # the features just represent what the . When a state is novel, the difference is bigger so the reward is bigger for getting to that state
    intrinsic_reward = ((target_features - predicted_features) ** 2).mean(dim=1)  # MSE
    return intrinsic_reward


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    #visualisation toolkit to visualise training
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    for i in range(100):
        writer.add_scalar("test_loss", i*2 ,global_step=i)
    
    #seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )#vectorised environment

    assert isinstance(envs.single_action_space, gym.spaces.Discrete) #only for discrete actions here

    ac_model = MarioNet(envs,input_shape=envs.envs[0].observation_space.shape).to(device) #actor critic model.

    optimizer = optim.Adam(ac_model.parameters(), lr=args.learning_rate, eps=1e-5) #epsilon decay of 1e-5 for PPO

    if(args.rand_net_dist):
        #have a predictor and target network
        rnd_predictor = RNDNetwork(envs.single_observation_space.shape).to(device)#randomly initialised target network
        rnd_target = RNDNetwork(envs.single_observation_space.shape).to(device).eval()
        rnd_optimiser = optim.Adam(rnd_predictor.parameters(), lr=args.learning_rate, eps=1e-5)


    observations = envs.reset()
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


    # training loop -> the learning rate is annealed with each update
    #each update is one iteration of the training loop
    for update in range(1,num_updates+1):
        #learning rate annealing - the learning rate of adam decays linearly. Papers show this annealing allows agents to obtain higher episodic return
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # linearly decrease to 0 as update increases to num_updates
            curr_lr = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = curr_lr
        
        #policy rollout is itself a loop inside the training process
        for step in range(0,args.num_steps):
            global_step += 1 * args.num_envs # doing steps for all the envs, so add that many steps
            #store next observation and dones
            obs[step] = next_obs
            dones[step] = next_done

            #during roll out, dont catch any gradient
            with torch.no_grad():
                action,logprob,_,value = ac_model.get_action_plus_value(next_obs)
                values[step] = value.flatten() #1d tensor
            
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done, info = envs.step(action.cpu().numpy()) #on cpu
            
            if args.rand_net_dist:
                intrinsic_reward = compute_intrinsic_reward(next_obs) #give more reward for novel states
                reward += args.intr_reward_beta * intrinsic_reward #beta controls how much the intrinsic reward adds 
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)# to gpu
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)   #reassign variables

            #log episodic return and info
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
            
        # use General Advantage Estimation(GAE) to do advantage estimation

        #PPO bootstraps values if environments are not done. The values of next observations are estimated as the end of rollout values
        
        # TODO: go over this code
        with torch.no_grad():
            next_value = ac_model.get_value(next_obs).reshape(1, -1)
            #gae way - 
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values # different way to do returns, as the other way is sum of discounted rewards i.e. returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
            
        # flatten the batch and store it. Need it as we break these batches into minibatches for training
        # so e.g. had 4*128 = 512 for total number in batch, then divide that by 4 to get a 128 minibatch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #get the minibatch now. Acquire all the indices of a batch, and for each update_epoch -> shuffle these indices. Then loop
        # through the entire batch, one minibatch at a time(e.g. 128 in each one)
        b_inds = np.arange(args.batch_size)
        clipfracs = [] # another debug variable -> measure how often the clip objective is actually triggered ///

        #Learning Phase - optimising the policy and value networks here
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)#shuffle batch - get a minibatch here. We guarantee we fetch all the training data
            for start in range(0, args.batch_size, args.minibatch_size): #step of minibatch, so get all the elements from the batch, but treat data in minibatches
                end = start + args.minibatch_size #just the maths on calculating end of minibatch
                mb_inds = b_inds[start:end] #get the minibatch indices

                #do a forward pass on a minibatch of observations
                _, newlogprob, entropy, newvalue = ac_model.get_action_plus_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds] #pass the minibatch actions, so that agent doesnt sample any new actions
                )
                logratio = newlogprob - b_logprobs[mb_inds] #do logarithmic substraction between new log probabilities and old ones in policy rollout phase
                ratio = logratio.exp() #get the ratio of this difference. At the first minibatch, the ratio is one as we wouldnt have modified the parameters yet

                with torch.no_grad():
                    #calculate approximate_kl
                    old_approx_kl = (-logratio).mean() #debug variable -> helps understand how aggressively policy updates
                    #negative log ratio -> ///
                    #howver, following approximation is a better estimate
                    approx_kl = ((ratio -1) - logratio).mean()
                    #measure how often the clip objective is triggered
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # PPO does advantage normalisation - substract their mean and divide by their standard deviation. This happens at minibatch level. Dosnt affect performance much
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) #substract by mean and divide by st deviation. add a small scalar to make sure not divide by 0
                
                # PPO uses a clipped surrogate objective - outperforms vanilla policy gradient
                # Polisy loss here  
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1,pg_loss2).mean() # we are doing the max of negatives here, whilst the paper did min of positives. But its the same thing

                # PPO also does value loss clipping, similar to clipped surrogate objective. 
                # this doesnt really improve performance however, but useful for reproducibility of original paper

                # PPO minimises this loss: L_V = max [ (V_theta_t - V_targ)^2, (clip(V_theta_t, V_theta_t-1 - epsilon, V_theta_t-1 + epsilon) - V_targ) ^ 2]
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    #clipping, ///go over this
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
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() # normally, the loss values is just MSE between predicted and emperical

                ## PPO also include entropy loss in its overall loss
                # entropy - measure of "chaos" in the action probability distribution. Maximising entropy encourages the agent to explore more

                entropy_loss = entropy.mean()
                #this is the overall loss we have 
                # we want to minimise the policy loss and the value loss, and maximise the entropy loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                #intrinsic reward loss calculation
                if args.rand_net_dist:
                    rnd_loss = nn.MSELoss(rnd_predictor(b_obs[mb_inds]), rnd_target(b_obs[mb_inds]).detach())

                    rnd_optimiser.zero_grad()
                    rnd_loss.backward()
                    rnd_optimiser.step()

                #backpropagation and optimising now
                optimizer.zero_grad()
                loss.backward()
                # PPO implements global gradient clipping. We set up a maximum gradient norm, ///
                nn.utils.clip_grad_norm_(ac_model.parameters(), args.max_grad_norm)
                optimizer.step()

            #early stopping: if approx_kl go over threshold, stop ////
            # we implement it at batch level, can also do at minibatch level                
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        #debug variable:explained variance - indicate if the value function is a good indicator of the returns ///
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        #measure stats
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()