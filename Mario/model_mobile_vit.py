import torch
import torch.nn as nn
import timm

#extract the features of the image first using this hybrid Cnn Vit
class MobileViTFeatureExtractor(nn.Module):
    def __init__(self, model_name="mobilevit_xxs"):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.feature_dim = self.model.feature_info.channels()[-1]
    
    def forward(self, x):
        features = self.model(x)[-1]  # Get the last feature map
        return features.view(x.size(0), -1)  # Flatten

class DuelingDQN(nn.Module):
    def __init__(self,num_actions=5,device="cpu"):
        super().__init__()
        self.feature_extractor = MobileViTFeatureExtractor()

        self.fully_connected = nn.Linear(self.feature_extractor.feature_dim,1024)

        flat_size = 1024

        self.action_value1 = nn.Linear(flat_size,1024)
        self.action_value2 = nn.Linear(1024,1024)
        self.action_value3 = nn.Linear(1024,num_actions)
        
        self.state_value1 = nn.Linear(flat_size,1024)
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