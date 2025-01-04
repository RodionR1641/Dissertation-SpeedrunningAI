import torch
import torch.nn as nn

#takes care of neural network,
class AtariNet(nn.Module):

    #setup all the methods in init and call in forward method

    #pass number of actions network can take for flexibility
    def __init__(self,nb_actions=4):
        super(AtariNet,self).__init__()#nn.Module init method, cover necessary class initialisation

        self.relu = nn.ReLU()

        #have 3 conv layers
        # CNN networks courses cover this btw
        # larger stride -> smaller output
        
        #these parameters are from the paper in the Joplin slide where such architecture was used
        self.conv1 = nn.Conv2d(1,32,kernel_size=(8,8),stride=(4,4)) #arguments -> number of channels(1 as grayscale),number of channels out,kernel size, stride
        #each layer shrinks image, pull more info out of the image and store this info in kernel layers(weights)
        #since conv1 gives 32 channels out(filters we use), the next one has to take 32 in
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        #in each layer, the image size shrinks so use smaller filter size
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1))

        #flattening layer before they go into the fully connected layer
        self.flatten = nn.Flatten()

        # randomly drop out certain number of weights in a network as you pass through it
        # a way to add randomness and stop network getting stuck in the "valleys"(flat solution space) without finding best solution
        # TODO: Cover dropout in studying
        self.dropout = nn.Dropout(p=0.2) #0.2 chance of dropping out for each layer we apply it to

        #fully connected layers now
        
        #now network interacts with rest of code
        
        #value of the image state
        #3136 = size of value when we flatten
        self.action_value1 = nn.Linear(3136,1024) # 1024 neurons we use in fully connected layer
        #want more of these hidden layers
        self.action_value2 = nn.Linear(1024,1024)
        self.action_value3 = nn.Linear(1024,nb_actions) # nb_actions output -> probability of actions selected

        #now do similar for State value
        self.state_value1 = nn.Linear(3136,1024)
        self.state_value2 = nn.Linear(1024,1024)
        self.state_value3 = nn.Linear(1024,1) # single value output, tells us if given state is valuable to the agent

    #function that gets called when network is being called
    #x is input to the network
    def forward(self,x):
        #passing through the layers, starting with the raw image data

        x = torch.Tensor(x) #casting, make data in tensor format(the image)
        #relu used after conv layers(their output). 
        x = self.relu(self.conv1(x)) # relu takes value, anything under 0 gets nullified. Add non linearity
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x) #basically take multidimensional array and turn it into 1D

        #we can reuse x and state_value as relu and dropout dont have any parameters to learn anyway
        # the state_value1,state_value2 etc actually learns the necessary parameters
        # basically get output from these layers, 
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))
        #no dropout in the end. It is used to avoid overfitting, but dont want it in final prediction. If we get e.g one/4 value
        #, dont want it to be dropped 0.2 percent of the time. not useful for learning

        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value3(action_value))
        
        #action_value is array of values, we will be selecting the highest of them
        # so substract action_value mean from action_value array so we get a representation of the value of the choices 
        # substract from mean -> can tell how useful an action is compared to the rest
        # if we did just state_value + action_value -> we overrepresent the state_value as we add it to the total action_values
        # and not what differentiates the action_values
        output = state_value + (action_value - action_value.mean())

        return output
    
    #these models take a while to train, want to save it and reload on start
    #use pt format
    def save_model(self, weights_filename="models/latest.pt"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done
        torch.save(self.state_dict(),weights_filename)
    
    def load_model(self, weights_filename="models/latest.pt"):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Loaded weights filename: {weights_filename}")
        except:
            print(f"No weights filename: {weights_filename}")
    