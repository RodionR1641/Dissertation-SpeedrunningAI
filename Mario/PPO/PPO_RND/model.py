import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from abc import ABC

# Note: PPO is meant to be run primarily on the CPU, so dont put this on gpu
class MarioNet(nn.Module,ABC):
    def __init__(self,input_shape):
        super(MarioNet,self).__init__()
        
        self.cnn = nn.Sequential(
            (nn.Conv2d(input_shape[0],32,8,stride=4)),
            nn.ReLU(),
            (nn.Conv2d(32,64,4,stride=2)),
            nn.ReLU(),
            (nn.Conv2d(64,64,3,stride=1)),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        flat_size = self.get_flat_size(input_shape)

        self.fc1 = nn.Linear(in_features=flat_size,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=448)

        self.extra_value_fc = nn.Linear(in_features=448,out_features=448)
        self.extra_policy_fc = nn.Linear(in_features=448, out_features=448)

        #actor part
        self.policy = nn.Linear(in_features=448, out_features=self.n_actions)
        #dual value for intrinsic and extrinsic rewards
        #extrinsic come from environment, rewards provided by environment
        #intrinsic comes from intrinsic motivations, self generated
        self.int_value = nn.Linear(in_features=448, out_features=1)
        self.ext_value = nn.Linear(in_features=448, out_features=1)

        for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    layer.bias.data.zero_()
        
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc1.bias.data.zero_()
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.fc2.bias.data.zero_()

        nn.init.orthogonal_(self.extra_policy_fc.weight, gain=np.sqrt(0.1))
        self.extra_policy_fc.bias.data.zero_()
        nn.init.orthogonal_(self.extra_value_fc.weight, gain=np.sqrt(0.1))
        self.extra_value_fc.bias.data.zero_()

        nn.init.orthogonal_(self.policy.weight, gain=np.sqrt(0.01))
        self.policy.bias.data.zero_()
        nn.init.orthogonal_(self.int_value.weight, gain=np.sqrt(0.01))
        self.int_value.bias.data.zero_()
        nn.init.orthogonal_(self.ext_value.weight, gain=np.sqrt(0.01))
        self.ext_value.bias.data.zero_()

    def forward(self,input):
        x = input /255.0 #make sure the value of pixels is normalised within 0 and 1
        x = self.cnn(x)
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_v = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)
        policy = self.policy(x_pi)
        probs = F.softmax(policy, dim=1)
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs

    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass
            dummy_input = torch.zeros(1,*input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x).shape[1] #get number of features after flattening

# Target model for RND predictions
class TargetModel(nn.Module):
    def __init__(self, input_shape):
        super(TargetModel, self).__init__()

        self.cnn = nn.Sequential(
            (nn.Conv2d(input_shape[0],32,8,stride=4)),
            nn.LeakyReLU(),
            (nn.Conv2d(32,64,4,stride=2)),
            nn.LeakyReLU(),
            (nn.Conv2d(64,64,3,stride=1)),
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()
        flatten_size = self.get_flat_size(input_shape)
        self.encoded_features = nn.Linear(in_features=flatten_size, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = self.cnn(x)
        x = self.flatten(x)
    
        return self.encoded_features(x)

    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass
            dummy_input = torch.zeros(1,*input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x).shape[1] #get number of features after flattening

# Target model for RND predictions
class PredictorModel(nn.Module,ABC):
    def __init__(self, input_shape):
        super(PredictorModel, self).__init__()

        self.cnn = nn.Sequential(
            (nn.Conv2d(input_shape[0],32,8,stride=4)),
            nn.LeakyReLU(),
            (nn.Conv2d(32,64,4,stride=2)),
            nn.LeakyReLU(),
            (nn.Conv2d(64,64,3,stride=1)),
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()
        flatten_size = self.get_flat_size(input_shape)
        
        self.fc1 = nn.Linear(in_features=flatten_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.encoded_features = nn.Linear(in_features=512, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = self.cnn(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
    
        return self.encoded_features(x)

    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass
            dummy_input = torch.zeros(1,*input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x).shape[1] #get number of features after flattening