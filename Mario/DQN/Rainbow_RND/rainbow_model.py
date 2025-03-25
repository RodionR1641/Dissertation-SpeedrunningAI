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
        
        flat_size = get_flat_size(input_shape,self.feature_layer)
        print("flattened size = "+str(flat_size))
        
        #value of the image state
        self.action_value1 = NoisyLinear(flat_size,1024) # 1024 neurons we use in fully connected layer
        self.action_value2 = NoisyLinear(1024,1024)
        self.advantage_layer = NoisyLinear(1024,out_dim * atom_size) # output a distribution of probabilites 
        # of potential returns(atoms) for each action(out_dim)

        #now do similar for State value
        self.state_value1 = NoisyLinear(flat_size,1024)
        self.state_value2 = NoisyLinear(1024,1024)
        self.value_layer = NoisyLinear(1024,atom_size)

        self.device = device
        self.to(self.device)
    #function that gets called when network is being called
    #x is input to the network
    def forward(self,x):
        x = x / 255.0#normalise between 0 and 1, better gradient stability
        if x.device != self.device:
            x.to(self.device)
        
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)#do sum over last dimension(atom_size)
        
        return q#output is (batch_size,out_dim)
    
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
        )# shape (batch_size,out_dim,atom_size). Advantage is a distribution over atoms for each action
        value = self.value_layer(val_hid2).view(-1,1,self.atom_size)#out dimension is just 1 here. The state value is the same for all actions

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True) #final shape is (batch_size,1,atom_size)

        dist = F.softmax(q_atoms,dim=-1)
        dist = dist.clamp(min=1e-3) #to avoid nans

        return dist #dimension is (batch_size, out_dim, atom_size)
    
    #Reset all noisy layers
    def reset_noise(self):
        self.action_value1.reset_noise()
        self.action_value2.reset_noise()
        self.advantage_layer.reset_noise()

        self.state_value1.reset_noise()
        self.state_value2.reset_noise() 
        self.value_layer.reset_noise()
    

# introduce noise into neural networks to encourage exploration
# sample from a distribution of weights rather than using fixed epsilon exploration rate
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

        #mu is the mean of weights and std is standard deviation. Both are learnable and we can learn how much noise is beneficial
        #they are both model parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        #hold the noise added to the weights during forward pass. not a model parameter
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()#init weight and biases
        self.reset_noise()#init noise values

    #init the trainable parameters
    #start with small random values, small and controlled noise
    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features) #divide by number of input features
        self.weight_mu.data.uniform_(-mu_range, mu_range) #in between the range, uniform
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )#std_init is a hyperparameter, control initial magnitude of noise. noise is normalised based on input features
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))# combine the noise vectors of in and out. Produce a matrix of size
        #outfeatures x in_features, match the shape of original matrix of all the weights we have
        # ger computes the outer product of 2 vectors
        self.bias_epsilon.copy_(epsilon_out)#bias noise is simply eps_out vector, biases have the same size as output dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        #y = ax + b formula but a is mu_w * sigma_w * w_eps and b is mu_b * sigma_b * b_eps
        # applying linear transformation: y = x @ W.T + b where @ is matrix multiplication and .T denotes transpose
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise) which is computationally efficient compared to sampling new noise
        for each weight"""
        x = torch.randn(size) #random noise data based on size. This is standard normal distribution: 0 mean, unit(1) variance

        #get the absolute value and sqrt it for scaling
        #then preserve the sign of the original x
        return x.sign().mul(x.abs().sqrt())
    
#calculates the intrinsic reward: make a prediction on a given state and see how well the learned model matches a random target
class RND_model(nn.Module):
    def __init__(self,
                 input_shape
                 ,device="cpu"
                 ):
        super(RND_model,self).__init__()

        self.relu = nn.ReLU()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=(8,8),stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()        
        flat_size = get_flat_size(input_shape,self.feature_layer)

        self.fc1 = nn.Linear(flat_size,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.out_layer = nn.Linear(1024,1024) #see what out dimension should be

        self.device = device
        self.to(device)

    def forward(self,x):
        if x.device != self.device:
            x.to(self.device)

        x = x/255.0 #normalise to be between 0 and 1
        x.to(self.device)
        x = self.feature_layer(x)
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out_layer(x)
        return x


def get_flat_size(input_shape,feature_layer):
    #pass dummy input through conv layers to get flatten size dynamically

    with torch.no_grad():#no gradient computation, just a dummy pass
        dummy_input = torch.zeros(1,*input_shape)
        x = feature_layer(dummy_input)
        flatten = nn.Flatten() #need instance of this to calculate shape
        flattened_x = flatten(x)
        return flattened_x.shape[1] #get number of features after flattening