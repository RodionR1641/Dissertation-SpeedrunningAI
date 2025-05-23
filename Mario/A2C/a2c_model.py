import torch
import torch.nn as nn
import os

#agent class handles multiprocessing of our program
#choose action is put in network class because of this
class ActorCritic(nn.Module):
    def __init__(self, input_shape,n_actions,device="cpu"):
        super(ActorCritic,self).__init__()

        self.device = device

        self.n_actions = n_actions
        self.relu = nn.ReLU()

        #combined input network, but output has 2 values

        self.conv1 = nn.Conv2d(input_shape[0],32,kernel_size=(8,8),stride=(4,4))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1))

        self.flatten = nn.Flatten()
        flat_size = self.get_flat_size(input_shape)

        self.shared = nn.Linear(flat_size,512) #use this layer as first FC layer to then branch off into actor and critic
        self.actor = nn.Linear(512,n_actions)
        self.critic = nn.Linear(512,1)
        """
        self.actor_pi1 = nn.Linear(flat_size,1024)
        self.actor_pi2 = nn.Linear(1024,1024)
        self.actor_pi3 = nn.Linear(1024,n_actions) #probabiltiy distribution

        self.critic_value1 = nn.Linear(flat_size,1024)
        self.critic_value2 = nn.Linear(1024,1024)
        self.critic_value3 = nn.Linear(1024,1)
        """
        self.to(device)
        print(f"Model device: {next(self.parameters()).device}")
    
    def forward(self,state):

        if state.device != self.device:
            state = state.to(self.device)
        state = state/255.0 #normalise to range 0-1
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        
        shared = self.relu(self.shared(x))
        actor_pi = self.actor(shared)
        critic_value = self.critic(shared)
        """
        actor_pi = self.relu(self.actor_pi1(x))
        actor_pi = self.relu(self.actor_pi2(actor_pi))
        actor_pi = self.actor_pi3(actor_pi) #shape [n_envs,n_acitions]

        critic_value = self.relu(self.critic_value1(x))
        critic_value = self.relu(self.critic_value2(critic_value))
        critic_value = self.critic_value3(critic_value) #shape [n_envs]
        """
        return (actor_pi, critic_value) #return probabilities and critic value

    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass

            self.to(self.device)
            # Then create dummy input on same device
            dummy_input = torch.zeros(1, *input_shape, device=self.device)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x).shape[1] #get number of features after flattening
        

    def save_model(self,weights_filename="models/a2c_latest.pt"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done
        print("...saving checkpoint...")
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(self.state_dict(),weights_filename)
    
    def load_model(self, weights_filename="models/a2c_latest.pt",device="cpu"):
        try:
            self.load_state_dict(torch.load(weights_filename,map_location=device,weights_only=True))
            print(f"Loaded weights filename: {weights_filename}")            
        except Exception as e:
            print(f"No weights filename: {weights_filename}")
            print(f"Error: {e}")
      