import torch.nn as nn
import torch
import numpy as np

class RNDNetwork(nn.Module):
    def __init__(self, input_shape):
        super(RNDNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # Same as MarioNet
            nn.ReLU(),
            nn.Linear(512, 512),  # Output feature vector
        )


"""
# TODO: go over this 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0): #use sqrt 2 as standard deviation
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer""
"""