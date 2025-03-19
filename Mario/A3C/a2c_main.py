import argparse
import numpy as np
from a2c_agent import Agent
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import RecordVideo
import logging
import numpy as np
import time
import os
import logging
import datetime
import random
import torch
import time
from gym.vector import SyncVectorEnv
from mario import Mario
from plot import LivePlot
import cProfile


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


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
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
    
    args = parser.parse_args()
    return args

def main():
    print_info()
    testing = False
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 777
    seed_run(seed)

    env = SyncVectorEnv(
        [make_env(args.gym_id, seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )#vectorised environment

    if(testing):
        pass
    else:
        train(env=env,device=device,args=args)

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

def train(env,device,args,num_envs=1):
    alpha = args.learning_rate
    gamma = args.gamma
    n_actions = env.envs[0].action_num
    agent = Agent(input_shape=env.envs[0].observation_space.shape,lr_rate=alpha,n_actions=n_actions,gamma=gamma)

    n_epochs = args.n_epochs
    plotter = LivePlot()   
    
    stats = {"Returns":[],"Loss": [],"AverageLoss": []}
    for epoch in range(1,n_epochs+1):

        states = env.reset()
        dones = [False] * num_envs
        ep_returns = [0] * num_envs
        ep_losses = [0] * num_envs
        game_steps = 0
        
        while not all(dones):
            #make a state tensor here instead of doing it twice in choose action and learn method
            states = torch.as_tensor(states, dtype=torch.float32,device=device)
            
            action = agent.choose_action(states) #get a list of actions chosen for each env

            next_states,rewards,dones,info = env.step(action)
            game_steps += num_envs

            loss = agent.learn(states,rewards,next_states,dones)

            for i in range(num_envs):
                ep_losses[i] += loss    
                ep_returns[i] += rewards[i]

            states = next_states
            if(game_steps % 100 == 0):
                print(f"im here {game_steps}")
        
        stats["Returns"].extend(ep_returns) #append returns for all environments
        stats["Loss"].append(ep_losses)
        
        print("Total loss = "+str(ep_losses))
        print("Time Steps = "+str(game_steps))

        if epoch % 10 == 0:
            agent.save_models(weights_filename=f"models/a2c_latest_{alpha}_{gamma}.pt") #save model every 10th epoch

            #average_returns = np.mean(stats["Returns"][-100:]) #average of the last 100 returns
            average_loss = np.mean(stats["Loss"][-100:])
            #graph can turn too big if we try to plot everything through. Only update a graph data point for every 10 epochs

            stats["AverageLoss"].append(average_loss)

            if(len(stats["Loss"]) > 100):
                logging.info(f"Epoch: {epoch} - Average loss: {np.mean(stats['Loss'][-100:])} ")
            else:
                #for the first 100 iterations, just return the episode return,otherwise return the average like above
                logging.info(f"Epoch: {epoch} - Episode loss: {np.mean(stats['Loss'][-1:])}")

        if epoch % 100 == 0:
            plotter.update_plot(stats)
        
        if epoch % 1000 == 0:
            agent.save_models(f"models/a2c_epoch{epoch}_{alpha}_{gamma}.pt")

    # can plot stuff here


def main():
    print_info()
    testing = args.testing
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_run()

    env = SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )#vectorised environment

    if(testing):
        pass
    else:
        train(env=env,device=device,args=args)


if __name__ == "__main__":
    args = parse_args()
    print(os.getcwd())
    #sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    #sys.path.append('../')

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

    main()