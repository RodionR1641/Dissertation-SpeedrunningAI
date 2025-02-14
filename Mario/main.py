import torch
import os
from breakout import *
import keyboard
from model import AtariNet
from agent import Agent


os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

environment = DQN_Mario(device=device)

user_input = False

model = MarioNet(nb_actions=5) #4 actions for agent can do in this game
model.to(device) # move torch module/network to a specific device

model.load_model(device=device)

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
    #nb_warmup -> time it takes for epsilon to decay from 1 to min_epsilon
    agent = Agent(model=model,device=device,epsilon=1.0,nb_warmup=5000,
                  nb_actions=4,
                  learning_rate=0.00001, #having it lower than 0.00001 is needed to get really good, but for starting out its
                  #fine to keep it here. as agent gets better, can decrease it with code
                  memory_capacity=100000,
                  batch_size=64)

    agent.train(env=environment, epochs=200000) #pass the DQNBreakout environment for agent to train on
