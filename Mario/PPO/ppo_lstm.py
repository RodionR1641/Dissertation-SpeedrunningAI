import argparse
import os
import random
import time
from distutils.util import strtobool
import wandb
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers import RecordVideo
from mario import Mario
from model_lstm import MarioNet
import datetime

#exception class to handle loading of models
class ModelLoadingError(Exception):
    pass

def print_info():
    print(f"Process started training on GPUs")

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


def test(env, device):

    state = env.reset()

    done = False
    next_done = torch.zeros(1).to(device)

    #using 1s instead of 0s to initialise the LSTM. Area of research on article https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    next_lstm_state = (
        torch.ones(ac_model.lstm.num_layers, 1, ac_model.lstm.hidden_size).to(device),
        torch.ones(ac_model.lstm.num_layers, 1, ac_model.lstm.hidden_size).to(device),
    )# hidden and cell states 
    
    while not done:
        time.sleep(0.01)
        # Convert state to tensor, add batch dimension
        state = np.array(state)
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get action from the model
        with torch.no_grad():  # No need to compute gradients during testing
            action, _, _, _, next_lstm_state = ac_model.get_action_plus_value(state,next_lstm_state,next_done)
        
        # Take action in the environment
        next_state, reward, done, info = env.step(action.item())
        
        state = next_state
        next_done = torch.tensor([0.0]) if not False else torch.tensor([1.0])
        
        # Render the environment (optional)
        env.render()
        if "episode" in info:
            episodic_return = info["episode"]["r"]
            episodic_len = info["episode"]["l"]
            print(f"episodic return = {episodic_return}, episodic len = {episodic_len}")
    
    env.close()

#these models take a while to train, want to save it and reload on start. Save both target and online for exact reproducibility
def save_models(num_updates,game_steps,num_completed_episodes,total_episodes,best_time_episode,
                 weights_filename="models/ppo_lstm/ppo_lstm_latest.pth"):
    #state_dict() -> dictionary of the states/weights in a given model
    # we override nn.Module, so this can be done

    checkpoint = {
        'ac_model_state_dict': ac_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': optimizer.param_groups[0]["lr"],  # Save the learning rate for the first group
        'num_updates': num_updates,      # Save the current epoch
        'game_steps': game_steps,  # Save the game step
        'num_completed_episodes': num_completed_episodes,
        'total_episodes': total_episodes,
        'best_time_episode': best_time_episode  
    }

    print("...saving checkpoint...")
    if not os.path.exists("models/ppo_lstm"):
            os.makedirs("models/ppo_lstm",exist_ok=True)
    torch.save(checkpoint,weights_filename)

#if model doesnt exist, we just have a random model
def load_models(weights_filename="models/ppo_lstm/ppo_lstm_latest.pth"):
    try:

        checkpoint = torch.load(weights_filename)
        ac_model.load_state_dict(checkpoint["ac_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer.param_groups[0]["lr"] = checkpoint['learning_rate']
        num_updates = checkpoint["num_updates"]
        game_steps = checkpoint["game_steps"]
        num_completed_episodes = checkpoint["num_completed_episodes"]
        total_episodes = checkpoint["total_episodes"]
        best_time_episode = checkpoint["best_time_episode"]

        ac_model.to(device)

        print(f"Loaded weights filename: {weights_filename}, curr_epoch_update = {num_updates}, \
                  game steps = {game_steps}, optimizer learning rate = { checkpoint['learning_rate']}")    

        return num_updates, game_steps, num_completed_episodes, total_episodes, best_time_episode            
    except Exception as e:
        print(f"No weights filename: {weights_filename}, using a random initialised model")
        print(f"Error: {e}")
        raise ModelLoadingError(f"Failed to load models from {weights_filename}") from e

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="PPO_LSTM_experiment",
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="SuperMarioBros-1-1-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=777,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000, # 50 million timesteps, roughly 50k episodes
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, #reproduce experiments
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    #tracking flag
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    #testing flag
    parser.add_argument("--testing", type=lambda x: bool(strtobool(x)), default= False , nargs="?", const=True,
        help="set to true if just want to test the agent playing the game")
    

    parser.add_argument("--wandb-project-name", type=str, default="RL_Mario",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    
    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8, #number of sub environments in a vector environment
        help="the number of parallel game environments")
    
    parser.add_argument("--num-steps", type=int, default=512,#higher value as using LSTM
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
    parser.add_argument("--num-minibatches", type=int, default=8,#more batches for better gradient estimates
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy") # ///
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,#higher than 0.5 to prevent underflow
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.03, #0.03 is quite lenient 
        help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # the total batch size for learning, split into minibatches
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # minibatches used for training
    # fmt: on
    return args



def make_env(gym_id,seed,environment_num,cap_video,run_name):
    def one_env():
        env = Mario(device=device,env_id=gym_id,seed=seed,num_stack=1) #num stack is 1 for LSTM
        if(cap_video):
            if environment_num == 0:
                env = RecordVideo(env,"videos/PPO_LSTM",name_prefix=f"{run_name}" 
                          ,episode_trigger=lambda x: x % 500 == 0)  # Record every 500th episode
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
    load_models_flag = True #decide if to load models or not
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_id_name = f"{args.exp_name}__{args.seed}__{args.gym_id}"

    testing = args.testing

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device PPO LSTM: ",device)
    print_info()

    #seeding
    seed_run()

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
    ac_model.to(device)

    optimizer = optim.Adam(ac_model.parameters(), lr=args.learning_rate, eps=1e-5) #epsilon decay of 1e-5 for PPO

    #track number of environment steps
    game_steps = 0
    curr_num_updates = 1
    num_completed_episodes = 0#how many games have ended in getting the flag
    total_episodes = 1 #total number of epochs/episodes of game playing that happened
    best_time_episode = 0 #lowest time remaining

    #load the model to continue training
    if load_models_flag == True:
        try:
            curr_num_updates, game_steps, num_completed_episodes, total_episodes,best_time_episode = load_models()
        except ModelLoadingError as e:
            pass #no need to do anything here, just keep game_steps and curr_updates 0

    if testing:
        env = Mario(device=device,env_id=args.gym_id,seed=args.seed,num_stack=1) #num stack 1 for LSTM
        test(env,device)
        exit() #dont need the rest of code just for a test run

    if args.track:
        run_id = None
        run_id_file = f"wandb_ids/{run_id_name}.txt"
        #if we ended a run, we can resume it in wandb just by getting the same run_id of that experiment as before. 
        #the models etc will also be loaded so as to resume the run
        if os.path.exists(run_id_file):
            with open(run_id_file,"r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip() #get the last run id for this experiment
                    run_id = last_line
        # Initialize or resume the W&B run
        try:
            if run_id:
                # Resume the existing run
                run = wandb.init(
                    id=run_id,
                    resume="allow", 
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    config=vars(args),
                    name=run_name,
                    monitor_gym=False, # Monitors videos, but for old gym. Doesn't work now
                    save_code=True
                )
                print(f"Resumed existing run with ID: {run_id}")
            else:
                # Start a new run
                run = wandb.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    config=vars(args),
                    name=run_name,
                    monitor_gym=False, # Monitors videos, but for old gym. Doesn't work now
                    save_code=True
                )
                print(f"Started new run with ID: {run.id}")

            # Save the current run_id to the file
            if not os.path.exists("wandb_ids"):
                os.makedirs("wandb_ids")

            # append the run_id to the file if it's not already there
            # if run_id is None -> there wasnt a previous one in this file, so need to append the current one to become first
            if run_id is None or (run.id+"\n") not in lines:
                with open(run_id_file, "a") as f:
                    f.write(f"{run.id}\n")  # Append the run_id as a new line

            run_id = run.id

            wandb.define_metric("game_steps")
            wandb.define_metric("episodes")

            # Define which metrics use which step
            # Define which metrics use which step
            wandb.define_metric("Charts/*", step_metric="game_steps")
            wandb.define_metric("Charts/*", step_metric="episodes")
            wandb.define_metric("losses/*", step_metric="game_steps")
            wandb.define_metric("losses/*", step_metric="episodes")
            wandb.define_metric("losses_avg/*", step_metric="game_steps")
            wandb.define_metric("losses_avg/*", step_metric="episodes")
            wandb.define_metric("losses_total/*", step_metric="game_steps")
            wandb.define_metric("losses_total/*", step_metric="episodes")

        except wandb.Error as e:
            print(f"Failed to initialize/resume W&B run: {e}")
            exit()

                
    # Storage setup - shape is to match num_steps * num_envs size
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    next_obs = torch.Tensor(envs.reset()).to(device) #store initial and subsequent observations
    next_done = torch.zeros(args.num_envs).to(device)
    #using 1s instead of 0s to initialise the LSTM. Area of research on article https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    next_lstm_state = (
        torch.ones(ac_model.lstm.num_layers, args.num_envs, ac_model.lstm.hidden_size).to(device),
        torch.ones(ac_model.lstm.num_layers, args.num_envs, ac_model.lstm.hidden_size).to(device),
    )# hidden and cell states 
    num_updates = args.total_timesteps // args.batch_size # e.g. 25k/512, this is how many iterations of updating training we have
    
    print(f"num_updates = {num_updates}")

    # training loop -> the learning rate is annealed with each update
    #each update is one iteration of the training loop
    episodic_reward = 0
    episodic_len = 0
    episodes = total_episodes

    sps = 0
    window_sec = 10 # number of seconds in which Steps Per Second(SPS) is calculated
    step_count_window = 0# number of time steps seen in the window time
    last_time = time.time()

    for update in range(curr_num_updates,num_updates+1):

        loss_total = 0
        v_loss_total = 0
        pg_loss_total = 0
        entropy_loss_total = 0 
        loss_count = 0

        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        #learning rate annealing - the learning rate of adam decays linearly. Papers show this annealing allows agents to obtain higher episodic return
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # linearly decrease to 0 as update increases to num_updates
            curr_lr = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = curr_lr
        
        #policy rollout is itself a loop inside the training process
        for step in range(0,args.num_steps):
            game_steps += 1 * args.num_envs # doing steps for all the envs, so add that many steps

            #track how many steps taken in given time window
            step_count_window += 1 * args.num_envs
            current_time = time.time()
            if current_time - last_time >= window_sec:
                sps = step_count_window / (current_time - last_time)
                step_count_window = 0
                last_time = current_time

            #store next observation and dones
            obs[step] = next_obs
            dones[step] = next_done

            #during roll out, dont catch any gradient. Just get the data for training
            with torch.no_grad():
                action,logprob,_,value,next_lstm_state = ac_model.get_action_plus_value(next_obs,next_lstm_state,next_done)
                values[step] = value.flatten() #1d tensor
            
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done, info = envs.step(action.cpu().numpy()) #on cpu
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)# to gpu
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)   #reassign variables

            #log episodic return and info
            for item in info:
                if "episode" in item.keys():
                    total_episodes += 1
                    episodes = total_episodes
                    episodic_reward = item["episode"]["r"]
                    episodic_len = item["episode"]["l"]

                    wandb.log({
                        "game_steps": game_steps,
                        "episodes": episodes,
                        # Log by game_steps (fine-grained steps)
                        "Charts/episodic_return": episodic_reward,
                        "Charts/episodic_length": episodic_len,
                    },commit=False)  # Default x-axis is game_steps

                    if item["flag_get"] == True:
                        num_completed_episodes += 1
                        #MOST IMPORTANT - time to complete game. See if we improve in speedrunning when we finish the game
                        # Log completion metrics (by game_steps)
                        wandb.log({
                            "game_steps": game_steps,  # Tracks the global step counter
                            "episodes": episodes,
                            "Charts/time_complete": item["time"],
                            "Charts/completion_rate": num_completed_episodes / total_episodes,
                        },commit=False)

                        if item["time"] > best_time_episode:
                            #find the previous file with this old best time
                            filename = f"models/ppo_lstm/best_{best_time_episode}.pth"
                            new_filename = f"models/ppo_lstm/best_{item['time']}.pth"

                            #rename so that not saving a new file for each new time
                            if os.path.exists(filename):
                                os.rename(filename,new_filename)
                            
                            #save this model that gave best time, if the model didnt exist then its just created
                            best_time_episode = item["time"]
                            save_models(update,game_steps,num_completed_episodes,total_episodes,best_time_episode
                                        ,weights_filename=new_filename)
            
        # use General Advantage Estimation(GAE) to do advantage estimation

        #PPO bootstraps values if environments are not done. The values of next observations are estimated as the end of rollout values
        
        with torch.no_grad():
            next_value = ac_model.get_value(next_obs,next_lstm_state,next_done).reshape(1, -1)
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
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #get the minibatch now. Acquire all the indices of a batch, and for each update_epoch -> shuffle these indices. Then loop
        # through the entire batch, one minibatch at a time(e.g. 128 in each one)
        
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        
        clipfracs = [] # another debug variable -> measure how often the clip objective is actually triggered ///

        #Learning Phase - optimising the policy and value networks here
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)#shuffle batch - get a minibatch here. We guarantee we fetch all the training data
            for start in range(0, args.num_envs, envsperbatch): #step of minibatch, so get all the elements from the batch, but treat data in minibatches
                end = start + envsperbatch #just the maths on calculating end of minibatch
                mbenvinds = envinds[start:end] #get the minibatch indices
                mb_inds = flatinds[:, mbenvinds].ravel()  # careful about the index

                #do a forward pass on a minibatch of observations
                _, newlogprob, entropy, newvalue, _ = ac_model.get_action_plus_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds] #do logarithmic substraction between new log probabilities and old ones in policy rollout phase
                ratio = logratio.exp() #get the ratio of this difference. At the first minibatch, the ratio is one as we wouldnt have modified the parameters yet

                with torch.no_grad():
                    #calculate approximate_kl
                    old_approx_kl = (-logratio).mean() #debug variable -> helps understand how aggressively policy updates
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
                    #clipping value loss
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

                loss_total += loss.item()
                pg_loss_total += pg_loss.item()
                v_loss_total += v_loss.item()
                entropy_loss_total += entropy_loss.item()
                loss_count += 1

                #backpropagation and optimising now
                optimizer.zero_grad()
                loss.backward()

                if game_steps % 500 == 0:
                    #Track gradient norms for monitoring stability and see exploding or vanishing gradients
                    total_norm = 0.0
                    for p in ac_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)  # L2 norm here - get the gradient tensor, calculate the L2 norm
                            #which is the square root of sum of squared values
                            total_norm += param_norm.item() ** 2 # square each parameters norm and adds to total_norm
                    total_norm = total_norm ** 0.5  #Overall gradient norm - square root of total
                    
                    #calculate per-layer gradient norms. Map layer names to their norms
                    layer_norms = {
                        name: p.grad.detach().norm(2).item() #map name and gradient norm of that layer
                        for name, p in ac_model.named_parameters() 
                        if p.grad is not None
                    }
                    wandb.log({
                        "game_steps": game_steps,
                        "Gradient/gradient_norm_total": total_norm,
                        **{f"Gradient/gradients/gradient_{name}": norm for name, norm in layer_norms.items()}
                    },commit=False)

                # PPO implements global gradient clipping just like rainbow etc. We set up a maximum gradient norm
                nn.utils.clip_grad_norm_(ac_model.parameters(), args.max_grad_norm)
                optimizer.step()

            #early stopping: if approx_kl go over threshold
            # we implement it at batch level, can also do at minibatch level                
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        
        #logged output saved in wandb 
        if update % 10 == 0:
            print("")
            if loss_count > 0:
                    print(f"SPS = {sps}, Episode return = {episodic_reward} \
                            ,Episode len = {episodic_len}, Episode loss = {loss_total}, Average loss = {loss_total/loss_count} \
                            ,Update = {update},Time Steps = {game_steps}, Learning Rate ={optimizer.param_groups[0]['lr']}")
            print("")
        if update % 10 == 0:
            save_models(num_updates=update,game_steps=game_steps,num_completed_episodes=num_completed_episodes,
                        total_episodes=total_episodes,best_time_episode=best_time_episode) 
        
        if update % 1000 == 0:
            save_models(num_updates=update,game_steps=game_steps,num_completed_episodes=num_completed_episodes,
                        total_episodes=total_episodes, best_time_episode=best_time_episode,
                        weights_filename=f"models/ppo_lstm/ppo_lstm_iter_{update}.pth")

        #debug variable:explained variance - indicate if the value function is a good indicator of the returns ///
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        wandb.log({
            # Learning rate and epoch tracking
            "game_steps": game_steps,
            "episodes": episodes,
            "Charts/learning_rate": optimizer.param_groups[0]["lr"],
            "Charts/epochs": update,
            
            #Total loss for the episode
            "losses_total/total_loss": loss_total,
            "losses_total/value_loss": v_loss_total,
            "losses_total/policy_loss": pg_loss_total,
            "losses_total/entropy": entropy_loss_total,

            # Average losses (more representative)
            "losses_avg/loss": loss_total/loss_count,
            "losses_avg/value_loss": v_loss_total/loss_count,
            "losses_avg/policy_loss": pg_loss_total/loss_count,
            "losses_avg/entropy": entropy_loss_total/loss_count,
            
            # Instantaneous losses (for tracking quick changes)
            "losses/loss": loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,

            "Charts/SPS": sps
        })

    save_models(num_updates=update,game_steps=game_steps,num_completed_episodes=num_completed_episodes,
                total_episodes=total_episodes,best_time_episode=best_time_episode)
    envs.close()
    wandb.finish()