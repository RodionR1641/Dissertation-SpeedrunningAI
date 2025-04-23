import torch
import torch.nn as nn

#takes care of neural network,
class MarioNet(nn.Module):

    #setup all the methods in init and call in forward method

    #pass number of actions network can take for flexibility
    def __init__(self,input_shape,device="cpu",nb_actions=5):
        super(MarioNet,self).__init__()

        self.relu = nn.ReLU()

        #input_shape[0] tells the C number of channels in C,H,W input shape
        self.conv1 = nn.Conv2d(input_shape[0],32,kernel_size=(8,8),stride=(4,4))
        #arguments -> number of channels(1 for grayscale),number of channels out,kernel size, stride
        #each layer shrinks image, pull more info out of the image and store this info in kernel layers(weights)
        #since conv1 gives 32 channels out(filters we use), the next one has to take 32 in
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        #in each layer, the image size shrinks so use smaller filter size
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1))

        #flattening layer before they go into the fully connected layer
        self.flatten = nn.Flatten()

        # randomly drop out certain number of weights in a network as you pass through it
        # a way to add randomness and stop network getting stuck in the "valleys"(flat solution space) without finding best solution
        self.dropout = nn.Dropout(p=0.2) #0.2 chance of dropping out for each layer we apply it to

        #fully connected layers now
        
        flat_size = self.get_flat_size(input_shape)
        print("flattened size = "+str(flat_size))
        
        self.hidden = nn.Linear(flat_size,512)
        self.action_value = nn.Linear(512,nb_actions)
        self.state_value = nn.Linear(512,1)
        self.device = device
        self.to(self.device)
    #function that gets called when network is being called
    #x is input to the network
    def forward(self,x):
        #passing through the layers, starting with the raw image data

        if x.device != self.device:
            x = x.to(self.device)
        x = x/255.0 #normalise in range 0-1
        #relu used after conv layers(their output). 
        x = self.relu(self.conv1(x)) # relu takes value, anything under 0 gets nullified. Add non linearity
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x) #basically take multidimensional array and turn it into 1D

        hidden = self.relu(self.hidden(x))
        action_value = self.action_value(hidden)
        state_value = self.state_value(hidden)

        #action_value is array of values, we will be selecting the highest of them
        # so substract action_value mean from action_value array so we get a representation of the value of the choices 
        # substract from mean -> can tell how useful an action is compared to the rest
        # if we did just state_value + action_value -> we overrepresent the state_value as we add it to the total action_values
        # and not what differentiates the action_values
        output = state_value + (action_value - action_value.mean())
        #output = state_value + (action_value - action_value.mean(dim=1, keepdim=True))
        return output
    
    #pass dummy input through conv layers to get flatten size dynamically
    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass
            dummy_input = torch.zeros(1,*input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x).shape[1] #get number of features after flattening

    