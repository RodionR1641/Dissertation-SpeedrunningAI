import argparse
import numpy as np
from a2c_agent import Agent
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import RecordVideo
import numpy as np
import os
import datetime
import random
import torch
from gym.vector import SyncVectorEnv
from mario import Mario
import time
import wandb
from wandb.integration.tensorboard import patch

load_models_flag = True 

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


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="A2C_experiment",
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="SuperMarioBros-1-1-v0",
        help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=777,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, #reproduce experiments
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="RL_Mario",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    #testing flag
    parser.add_argument("--testing", type=lambda x: bool(strtobool(x)), default= False , nargs="?", const=True,
        help="set to true if just want to test the agent playing the game")
    #resume run - if we have a run that hasnt finished with the same experiment,seed and we can resume it
    parser.add_argument("--resume-run", type=lambda x: bool(strtobool(x)), default= True , nargs="?", const=True,
        help="set to true if just want to test the agent playing the game")

    
    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8, #number of sub environments in a vector environment
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--n-epochs", type=int, default=200_000,
        help="total timesteps of the experiments")
    parser.add_argument("--lam", type=float, default=0.95,
        help="lam parameter of GAE"),
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient for entropy bonus (encourage exploration)"),
    parser.add_argument("--n-steps-per-update", type=int, default=128,
        help="coefficient for entropy bonus (encourage exploration)")
    
    
    args = parser.parse_args()
    return args

def make_env(gym_id,seed,environment_num,cap_video,run_name):
    def one_env():
        env = Mario(env_id=gym_id,seed=seed)
        if(cap_video):
            if environment_num == 0:
                env = RecordVideo(env,"videos/A2C",name_prefix=f"{run_name}" 
                          ,episode_trigger=lambda x: x % 100 == 0)  # Record every 100th episode
        return env    
    return one_env

def seed_run():
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(args.seed)
        #torch.backends.cudnn.benchmark = False # could be useful for reproducibility, but setting to False affects performance
        torch.backends.cudnn.deterministic = args.torch_deterministic
    
    np.random.seed(args.seed)
    random.seed(args.seed)


def train(env,device,args):
    alpha = args.learning_rate
    gamma = args.gamma
    n_actions = env.envs[0].action_num
    n_steps_per_update = args.n_steps_per_update

    lam = args.lam
    entropy = 0
    ent_coef = args.ent_coef # coefficient for entropy bonus (encourage exploration)

    agent = Agent(input_shape=env.envs[0].observation_space.shape,device=device,lr_rate=alpha,
                  n_actions=n_actions,gamma=gamma,num_envs=num_envs)
    
    if load_models_flag == True:
        agent.load_models()

    n_epochs = args.n_epochs

    game_steps = agent.game_steps

    states = env.reset() #reset once at the start

    for epoch in range(agent.curr_epoch,n_epochs+1):

        # Reset lists to collect experiences
        ep_value_preds = torch.zeros(n_steps_per_update, num_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, num_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, num_envs, device=device)
        masks = torch.zeros(n_steps_per_update, num_envs, device=device)

        #no need to reset each time as using vectorised environments, just let them keep going

        for step in range(n_steps_per_update):
            states = torch.as_tensor(states, dtype=torch.float32,device=device)

            action, log_probs, state_value, entropy = agent.choose_action_entropy(states) #get a list of actions chosen for each env
            next_states, rewards, dones, info = env.step(action)

            game_steps += num_envs #8 envs took a step

            # Store experiences
            ep_value_preds[step] = torch.squeeze(state_value)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = log_probs
            masks[step] = torch.tensor([not done for done in dones])

            states = next_states

            #log episodic return and info, this does it on vectorised envs 
            for item in info:
                if "episode" in item.keys():#check if current env completed episode
                    agent.total_episodes += 1
                    episodic_reward = item["episode"]["r"]
                    episodic_len = item["episode"]["l"]

                    print(f"game_step={game_steps}, episodic_return={episodic_reward}, episodic len={episodic_len}")
                    writer.add_scalar("Charts/episodic_return", episodic_reward, game_steps) 
                    writer.add_scalar("Charts/episodic_length", episodic_len, game_steps)
                    #episode graphs instead of game_steps
                    writer.add_scalar("Charts/episodic_return_episode", episodic_reward, agent.total_episodes) 
                    writer.add_scalar("Charts/episodic_length_episode", episodic_len, agent.total_episodes)

                    if item["flag_get"] == True:
                        agent.num_completed_episodes += 1
                        #MOST IMPORTANT - time to complete game. See if we improve in speedrunning when we finish the game
                        writer.add_scalar("Complete/time_complete", item["time"],game_steps)
                        #completion compared to total episodes
                        writer.add_scalar("Complete/completion_rate",agent.num_completed_episodes/agent.total_episodes,game_steps)
        
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,ep_action_log_probs,ep_value_preds,entropy,masks,gamma,lam,ent_coef
        )

        loss = agent.update_params(critic_loss,actor_loss)

        writer.add_scalar("Charts/learning_rate", agent.optimizer.param_groups[0]["lr"], game_steps)
        writer.add_scalar("Charts/epochs", epoch, game_steps)

        writer.add_scalar("Charts/entropy", entropy, game_steps)
        #add last loss, useful for tracking quick changes
        writer.add_scalar("losses/loss", loss, game_steps)

        if epoch % 10 == 0:
            print("")
            print(f"Loss = {loss}, Epoch = {epoch}, \
                        Time Steps = {game_steps}, entropy = {entropy}")
            print("")

        if epoch % 10 == 0:
            agent.save_models(epoch=epoch) #save model every 100th epoch

        if epoch % 100 == 0:
            agent.save_models(epoch=epoch,weights_filename=f"models/a2c/a2c_{epoch}_{game_steps}.pt")
    
    agent.save_models(epoch)
    env.close()
    writer.close()


def test(env,device,args):
    alpha = args.learning_rate
    gamma = args.gamma
    n_actions = env.action_num

    agent = Agent(input_shape=env.observation_space.shape,lr_rate=alpha,
                  n_actions=n_actions,gamma=gamma,num_envs=num_envs)
    
    if load_models_flag == True:
        agent.load_models() #make sure the latest model is loaded
    done = False
    state = env.reset()

    while not done:
        state = np.array(state)#make a single array for efficincy
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # Get action from the agent
        with torch.no_grad():  # No need to compute gradients during testing
            action, _, _, _ = agent.choose_action_entropy(state)

            next_state, reward, done, info = env.step(action.item())

            state = next_state

            env.render()
            time.sleep(0.01)
            if "episode" in info:
                    episodic_return = info["episode"]["r"]
                    episodic_len = info["episode"]["l"]
                    print(f"episodic return = {episodic_return}, episodic len = {episodic_len}")
    
    env.close()

if __name__ == "__main__":
    args = parse_args()
    print(os.getcwd())
    testing = True

    # Define log file name (per process)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    print_info()
    testing = args.testing
    resume_run = args.resume_run
    num_envs = args.num_envs
    
    run_name = f"{args.exp_name}__{args.seed}__{args.gym_id}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_id_name = f"{args.exp_name}__{args.seed}__{args.gym_id}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if(testing):
        env = Mario(args.gym_id,args.seed)

        test(env=env,device=device,args=args)
        exit()#only test 

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
            if run_id and resume_run:
                # Resume the existing run
                run = wandb.init(
                    id=run_id,
                    resume="must",  # Only resume if the run_id exists
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

            patch()  # Make sure TensorBoard graphs are saved to wandb
            run_id = run.id

        except wandb.Error as e:
            print(f"Failed to initialize/resume W&B run: {e}")
            exit()
        
    #visualisation toolkit to visualise training - Tensorboard, allows to see the metrics like loss and see hyperparameters
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    seed_run()

    env = SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(num_envs)]
    )#vectorised environment

    train(env=env,device=device,args=args)

    if args.track:
        wandb.finish()
        #remove the run_id from the file as its done now
        if os.path.exists(run_id_file):
            with open(run_id_file,"r") as f:
                lines = f.readlines()

            #filter out the run_id lines
            lines = [line for line in lines if line.strip() != f"{run_id}\n".strip()]

            # Write the remaining lines back to the file
            with open(run_id_file, "w") as f:
                f.writelines(lines)

            print(f"Removed run_id: {run_id.strip()}")