import torch
import os
import keyboard
from agent import Agent
from mario import DQN_Mario

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

environment = DQN_Mario(device=device)

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
    #nb_warmup -> time it takes for epsilon to decay from 1 to min_epsilon
    agent = Agent(input_dims=environment.observation_space.shape,device=device,epsilon=1.0,nb_warmup=5000,
                  nb_actions=num_actions,
                  learning_rate=0.00001, #having it lower than 0.00001 is needed to get really good, but for starting out its
                  #fine to keep it here. as agent gets better, can decrease it with code
                  memory_capacity=10_000,
                  batch_size=32)

    agent.train(env=environment, epochs=200000) #pass the DQNBreakout environment for agent to train on
