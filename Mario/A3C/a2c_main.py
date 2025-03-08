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



def train(env,device):
    alpha = 1e-5
    gamma = 0.99
    n_actions = env.action_num
    agent = Agent(input_shape=env.observation_space.shape,lr_rate=alpha,n_actions=n_actions,gamma=gamma)

    n_epochs = 10000
    plotter = LivePlot()   
    
    stats = {"Returns":[],"Loss": [],"AverageLoss": []}
    for epoch in range(1,n_epochs+1):

        state = env.reset()
        done = False
        ep_return = 0
        ep_loss = 0
        game_steps = 0
        
        while not done:
            action = agent.choose_action(state)

            next_state,reward,done,info = env.step(action)
            game_steps += 1

            loss = agent.learn(state,reward,next_state,done)

            ep_loss += loss    
            ep_return += reward

            state = next_state
            print(f"im here {game_steps}")
        
        stats["Returns"].append(ep_return)
        stats["Loss"].append(ep_loss)
        
        print("Total loss = "+str(ep_loss))
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


if __name__ == "__main__":
    
    testing = False
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = Mario(device=device)

    if(testing):
        pass
    else:
        train(env=env,device=device)