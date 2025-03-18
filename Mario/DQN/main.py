import torch
import os
import keyboard
#from Rainbow.agent import Agent
from Rainbow_RND.agent import Agent
from mario import DQN_Mario
import numpy as np
import random

def seed_run(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    np.random.seed(seed)
    random.seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_vit = False

environment = DQN_Mario(use_vit=use_vit)

num_actions = environment.action_num

user_input = False

if(user_input):
    done = False

    #rewrite for mario
    while not done:
        action = None
        if keyboard.is_pressed("s"):
            action = 1 # fire
        elif keyboard.is_pressed("a"):
            action = 3#left
        elif keyboard.is_pressed("d"):
            action = 2#right
        else:
            action = 0#do nothing

        _,_,done, _= environment.step(action)

else:
    seed = 999
    environment.seed(seed)
    seed_run(seed)
    #nb_warmup -> time it takes for epsilon to decay from 1 to min_epsilon
    agent = Agent(input_dims=environment.observation_space.shape,
                  env=environment,
                  device=device,
                  nb_actions=num_actions,
                  memory_capacity=10_000,
                  batch_size=32,use_vit=use_vit)

    agent.train(epochs=200000) #pass the Mario environment for agent to train on