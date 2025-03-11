import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#takes care of neural network,
class MarioNet(nn.Module):

    #setup all the methods in init and call in forward method

    def __init__(self,
                 input_shape,
                 support,
                 out_dim,
                 atom_size
                 ,device="cpu"
                 ):
        super(MarioNet,self).__init__()

        #Categorical DQN
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.relu = nn.ReLU()


        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=(8,8),stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU(),
        )

        #flattening layer before they go into the fully connected layer
        self.flatten = nn.Flatten()
        
        flat_size = self.get_flat_size(input_shape)
        print("flattened size = "+str(flat_size))
        
        #value of the image state
        self.action_value1 = NoisyLinear(flat_size,1024) # 1024 neurons we use in fully connected layer
        self.action_value2 = NoisyLinear(1024,1024)
        self.advantage_layer = NoisyLinear(1024,out_dim * atom_size) # TODO: understand this better

        #now do similar for State value
        self.state_value1 = NoisyLinear(flat_size,1024)
        self.state_value2 = NoisyLinear(1024,1024)
        self.value_layer = NoisyLinear(1024,atom_size) # TODO: understand this better

        self.device = device
        self.to(self.device)
    #function that gets called when network is being called
    #x is input to the network
    def forward(self,x):
        x.to(self.device)
        
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
        """
        #relu used after conv layers(their output). 
        x = self.relu(self.conv1(x)) # relu takes value, anything under 0 gets nullified. Add non linearity
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x) #basically take multidimensional array and turn it into 1D

        #we can reuse x and state_value as relu and dropout dont have any parameters to learn anyway
        # the state_value1,state_value2 etc actually learns the necessary parameters
        # basically get output from these layers, 
        state_value = self.relu(self.state_value1(x))
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.state_value3(state_value) #no relu

        action_value = self.relu(self.action_value1(x))
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.action_value3(action_value) #dont have relu on last layer as dont want to limit
        #to only negative actions
        
        output = state_value + (action_value - action_value.mean())
        return output
        """
    
    #get the softmax distribution of q values
    def dist(self,x):

        #get distribution for atoms
        feature = self.feature_layer(x)
        feature = self.flatten(feature)

        #dueling network passing through
        adv_hid1 = self.relu(self.action_value1(feature))
        adv_hid2 = self.relu(self.action_value2(adv_hid1))
        val_hid1 = self.relu(self.state_value1(feature))
        val_hid2 = self.relu(self.state_value2(val_hid1))

        advantage = self.advantage_layer(adv_hid2).view(
            -1,self.out_dim,self.atom_size
        )
        value = self.value_layer(val_hid2).view(-1,1,self.atom_size)#out dimension is just 1 here

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms,dim=-1)
        dist = dist.clamp(min=1e-3) #to avoid nans

        return dist

    #pass dummy input through conv layers to get flatten size dynamically
    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass
            dummy_input = torch.zeros(1,*input_shape)
            x = self.feature_layer(dummy_input)
            return self.flatten(x).shape[1] #get number of features after flattening
    
    #Reset all noisy layers
    def reset_noise(self):
        self.action_value1.reset_noise()
        self.action_value2.reset_noise()
        self.advantage_layer.reset_noise()

        self.state_value1.reset_noise()
        self.state_value2.reset_noise() 
        self.value_layer.reset_noise()

    #these models take a while to train, want to save it and reload on start
    #use pt format
    def save_model(self, weights_filename="models/latest.pt"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done
        print("...saving checkpoint...")
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(self.state_dict(),weights_filename)
    
    def load_model(self, weights_filename="models/latest.pt",device="cpu"):
        try:
            self.load_state_dict(torch.load(weights_filename,map_location=device,weights_only=True))
            print(f"Loaded weights filename: {weights_filename}")            
        except Exception as e:
            print(f"No weights filename: {weights_filename}")
            print(f"Error: {e}")
    



class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())