import torch
from PIL import Image
import numpy as np
import gym
import os
from breakout import *
import keyboard
from model import AtariNet
from agent import Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

environment = DQNBreakout(device=device,render_mode="human")

user_input = False

model = AtariNet(nb_actions=4)
model.to(device) # move torch module/network to a specific device

model.load_model()

if(user_input):
    done = False

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
                  memory_capacity=1000000,
                  batch_size=64)

    agent.test(env=environment) #now call it to test it