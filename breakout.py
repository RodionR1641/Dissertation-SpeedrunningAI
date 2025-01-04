import collections
import gym
import numpy as np
import cv2
from PIL import Image
import torch

#subclass of wrapper
class DQNBreakout(gym.Wrapper):
    #rgb_array gives pixel info of game for us to work with
    # human mode actually allows to see
    def __init__(self, render_mode='rgb_array',repeat=4,device='cpu'):
        env = gym.make("BreakoutNoFrameskip-v4",render_mode=render_mode)

        super(DQNBreakout,self).__init__(env)

        #self.env = env
        self.repeat = repeat
        self.lives = env.ale.lives()#need to train our agent to learn that losing a life in game is bad
        self.frame_buffer = []
        self.device = device
        self.image_shape = (84,84)

    #take action on an environment ->returns a state etc
    def step(self,action):
        total_reward = 0
        done = False

        #dont want agent to think about step every frame, dont need to react every single frame
        # take same action 4 frames in a row, what the frame means basically
        # take max of the last 2 frames
        for i in range(self.repeat):
            observation,reward,done, info = self.env.step(action)

            total_reward += reward #add to total_reward from cycle. necessary because of our repeating action
            #caption the number of lives
            print(info,total_reward)

            current_lives = info['lives']

            #start of with 5 lives
            if current_lives < self.lives:
                total_reward = total_reward - 1 # can be any number. We want to have losing live to have same impact though. 
                #positve impact for scoring, same amount of negative impact if agent lost a live compared to getting a reward
                self.lives = current_lives

            #print(f"lives: {self.lives}, Total reward: {total_reward}")
            
            self.frame_buffer.append(observation) #need to store frames

            if done:
                break
        
        max_frame = np.max(self.frame_buffer[-2:],axis=0) # grab last 2 frames, take max
        #can store this stuff in a buffer
        #what does .to() do?

        #take the frame with most pixels as some images can get blurry and faded, so want better frames basically
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device) #done processing max frame

        total_reward = torch.tensor(total_reward).view(1,-1).float() # making sure data is standardasised, can be then used in batches
        total_reward = total_reward.to(self.device) # send this to cpu/gpu for processing

        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)

        return max_frame,total_reward,done,info
    
    #observation is image
    # take images of various types and sizes -> standardasi
    # also reduce Complexity of what the network needs to learn on. Shrink it
    # render it grayscale too, and divide by 255(get a range from 0 to 1), kind of like normalising values
    def process_observation(self,observation):
        #TODO: add content

        img = Image.fromarray(observation)#represent an image from this array of observation
        img = img.resize(self.image_shape)
        img = img.convert("L") #grayscale
        img = np.array(img)
        img = torch.from_numpy(img) #tensors and numpy arrays are similar but tensors more efficient for ML on cpus/gpus for
        #backtracking as they track gradients in gradient descent
        
        #unsqueeze the image, later we pass a batch size number, so we need something in the
        #batch size column and image channel column
        img = img.unsqueeze(0)
        #add 2 dimensions, do twice
        img = img.unsqueeze(0)

        #divide by 255, so range is 0-1
        img = img / 255.0

        img = img.to(self.device)
        return img

    #2 functions we use the most -> reset and step
    #reset is to bring it back to setup state
    #overriding it here compared to default
    def reset(self):
        self.frame_buffer = []#clear the buffer

        observation = self.env.reset() # _ underscore as we dont care about the second value returned here, only observation

        self.lives = self.env.ale.lives()

        observation = self.process_observation(observation)

        return observation