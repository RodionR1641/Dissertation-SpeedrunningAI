import torch
import torch.nn as nn
import timm
import os

#extract the features of the image first using this hybrid Cnn Vit
class MobileViTFeatureExtractor(nn.Module):
    def __init__(self,device="cpu",model_name="mobilevit_xxs"):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.feature_dim = self.model.feature_info.channels()[-1] #the last layer size, 320
        self.device = device
        self.to(self.device)
    
    def forward(self, x):
        x.to(self.device)
        features = self.model(x)[-1]  # Get the last feature map
        return features.view(x.size(0), -1)  # Flatten

class DuelingDQN(nn.Module):
    def __init__(self,num_actions=5,device="cpu"):
        super().__init__()
        self.feature_extractor = MobileViTFeatureExtractor(device)

        self.action_value1 = nn.Linear(self.feature_extractor.feature_dim,1024)
        self.action_value2 = nn.Linear(1024,1024)
        self.action_value3 = nn.Linear(1024,num_actions)
        
        self.state_value1 = nn.Linear(self.feature_extractor.feature_dim,1024)
        self.state_value2 = nn.Linear(1024,1024)
        self.state_value3 = nn.Linear(1024,1)
        
        self.device = device
        self.to(self.device)
    
    def forward(self,x):
        x = torch.Tensor(x)
        x.to(self.device)

        features = self.feature_extractor(x)
        features = self.fully_connected(features) #flatten
        
        x = torch.relu(features)

        state_value = self.relu(self.state_value1(x))
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.state_value3(state_value)

        action_value = self.relu(self.action_value1(x))
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.action_value3(action_value)

        return state_value + (action_value - action_value.mean())


    #pass dummy input through conv layers to get flatten size dynamically
    def get_flat_size(self,input_shape):

        with torch.no_grad():#no gradient computation, just a dummy pass
            dummy_input = torch.zeros(1,*input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x).shape[1] #get number of features after flattening

    #these models take a while to train, want to save it and reload on start
    #use pt format
    def save_model(self, weights_filename="models/latest.pt"):
        #state_dict() -> dictionary of the states/weights in a given model
        # we override nn.Module, so this can be done
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(self.state_dict(),weights_filename)
    
    def load_model(self, weights_filename="models/latest.pt",device="cpu"):
        try:
            self.load_state_dict(torch.load(weights_filename,map_location=device))
            print(f"Loaded weights filename: {weights_filename}")            
        except Exception as e:
            print(f"No weights filename: {weights_filename}")
            print(f"Error: {e}")


MobileViTFeatureExtractor()