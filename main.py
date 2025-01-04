import torch
from PIL import Image
import numpy as np
import gym
import os
from breakout import *
import keyboard
from model import AtariNet

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

environment = DQNBreakout(device=device, render_mode="human")

user_input = False

model = AtariNet(nb_actions=4)
model.to(device) # make sure its on the right device

model.load_model()

state = environment.reset()

print(model.forward(state))

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
    
    for _ in range(100):
        action = environment.action_space.sample()
        # we dont tell anything to the enviroment to tell that the lives losing is bad
        #environment = environment.unwrapped
        
        state, reward, done, info = environment.step(action) #state is the observation here i.e. a processed image