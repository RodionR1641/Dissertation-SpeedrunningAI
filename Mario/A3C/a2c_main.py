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
    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="a2c-experiment",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--testing", type=lambda x: bool(strtobool(x)), default= False, nargs="?", const=True,
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

def make_env(gym_id,seed,environment_num,cap_video,name):
    def one_env():
        env = Mario(env_id=gym_id,seed=seed)
        if(cap_video):
            if environment_num == 0:
                env = RecordVideo(env,f"videos/{name}")
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
    ent_coef = args.ent_coef # coefficient for entropy bonus (encourage exploration)

    agent = Agent(input_shape=env.envs[0].observation_space.shape,lr_rate=alpha,
                  n_actions=n_actions,gamma=gamma,num_envs=num_envs)
    
    if load_models_flag == True:
        agent.load_models()

    n_epochs = args.n_epochs

    game_steps = agent.game_steps
    
    for epoch in range(agent.curr_epoch,n_epochs+1):

        # Reset lists to collect experiences
        ep_value_preds = torch.zeros(n_steps_per_update, num_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, num_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, num_envs, device=device)
        masks = torch.zeros(n_steps_per_update, num_envs, device=device)

        states = env.reset()

        for step in range(n_steps_per_update):
            states = torch.as_tensor(states, dtype=torch.float32,device=device)

            action, log_probs, state_value, entropy = agent.choose_action_entropy(states) #get a list of actions chosen for each env
            next_states, rewards, dones, info = env.step(action)

            game_steps += num_envs #8 envs took a step

            # Store experiences
            ep_value_preds[step] = torch.squeeze(state_value)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = log_probs
            masks[step] = torch.tensor([not done for done in dones], device=device)

            states = next_states

            #log episodic return and info, this does it on vectorised envs 
            for item in info:
                if "episode" in item.keys():#check if current env completed episode
                    episodic_reward = item["episode"]["r"]
                    episodic_len = item["episode"]["l"]

                    print(f"global_step={game_steps}, episodic_return={episodic_reward}, episodic len={episodic_len}")
                    writer.add_scalar("Charts/episodic_return", episodic_reward, game_steps) 
                    writer.add_scalar("Charts/episodic_length", episodic_len, game_steps)
                    # episodic length(number of steps)
                    break
        
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,ep_action_log_probs,ep_value_preds,entropy,masks,gamma,lam,ent_coef
        )

        loss = agent.update_params(critic_loss,actor_loss)

        writer.add_scalar("Charts/learning_rate", agent.optimiser.param_groups[0]["lr"], game_steps)
        writer.add_scalar("Charts/epochs", epoch, global_step=game_steps)

        #add last loss, useful for tracking quick changes
        writer.add_scalar("losses/loss", loss, game_steps)

        if epoch % 10 == 0:
            print("")
            print(f"Loss = {loss}, Epoch = {epoch}, \
                        Time Steps = {game_steps}, Last Rewards = {', '.join(map(str, rewards.flatten()))}")
            print("")

        if epoch % 100 == 0:
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
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Get action from the agent
        with torch.no_grad():  # No need to compute gradients during testing
            action, _, _, _ = agent.choose_action_entropy(state)

            next_state, reward, done, info = env.step(action)

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
    testing = False

    # Define log file name (per process)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    video_folder = "" #TODO: make a folder here

    print_info()
    testing = args.testing
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if(testing):
        env = Mario(args.gym_id,args.seed)
        env = RecordVideo(env,f"videos/{run_name}")

        test(env=env,device=device,args=args)
        exit()#only test 

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

    seed_run()
    num_envs = args.num_envs

    env = SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(num_envs)]
    )#vectorised environment

    train(env=env,device=device,args=args)