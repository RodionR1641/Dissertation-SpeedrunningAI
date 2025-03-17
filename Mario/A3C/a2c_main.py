import gym
import numpy as np
from a2c_agent import Agent
import logging
import numpy as np
import time
import os
import logging
import datetime
import random
import torch
import sys
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym.vector import SyncVectorEnv
import cProfile

print(os.getcwd())
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#sys.path.append('../')
from mario import Mario
from plot import LivePlot

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


def make_env(device,seed=None):
    def one_env():
        env = Mario(device=device)
        env.seed = seed
        return env    
    return one_env

def seed_run(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    np.random.seed(seed)
    random.seed(seed)

def train(env,device,num_envs=1):
    alpha = 1e-5
    gamma = 0.99
    n_actions = env.envs[0].action_num
    agent = Agent(input_shape=env.envs[0].observation_space.shape,lr_rate=alpha,n_actions=n_actions,gamma=gamma)

    n_epochs = 10000
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
            game_steps += 1

            loss = agent.learn(states,rewards,next_states,dones)

            for i in range(num_envs):
                ep_losses[i] += loss    
                ep_returns[i] += rewards[i]

            states = next_states
            if(game_steps % 100 == 0):
                print(f"im here {game_steps}")
        
        stats["Returns"].extend(ep_returns) #TODO: vectorised env, think how to append it here
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
    
    testing = False
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 999
    seed_run(seed)
    num_envs = 8
    env = SyncVectorEnv([make_env(device=device,seed=seed) for _ in range(num_envs)])

    if(testing):
        pass
    else:
        train(env=env,device=device)

main()