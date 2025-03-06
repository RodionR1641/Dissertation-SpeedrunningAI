import torch
import torch.nn as nn
import timm
from timm.layers.conv_bn_act import ConvNormAct
import os

#extract the features of the image first using this hybrid Cnn Vit
class MobileViTFeatureExtractor(nn.Module):
    def __init__(self,device="cpu",model_name="mobilevit_xxs"):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.feature_dim = 20480 #the last layer size flattened
        self.device = device

        #this is to make the model accept greyscale input instead of rgb 3 channel input
        first_conv_layer = self.model.stem
        self.model.stem = ConvNormAct(
            in_channels=1,  # Grayscale input
            out_channels=first_conv_layer.conv.out_channels,
            kernel_size=first_conv_layer.conv.kernel_size,
            stride=first_conv_layer.conv.stride,
            padding=first_conv_layer.conv.padding,
            bias=first_conv_layer.conv.bias is not None,
            dilation=first_conv_layer.conv.dilation,
            groups=first_conv_layer.conv.groups
        )
        
        """apply_norm=first_conv_layer.conv.apply_norm, 
        apply_act=first_conv_layer.conv.apply_act,
        norm_layer=first_conv_layer.conv.norm_layer,
        act_layer=first_conv_layer.conv.act_layer,
        aa_layer=first_conv_layer.conv.aa_layer,
        drop_layer=first_conv_layer.conv.drop_layer,
        conv_kwargs=first_conv_layer.conv.conv_kwargs,
        norm_kwargs=first_conv_layer.conv.norm_kwargs,
        act_kwargs=first_conv_layer.conv.act_kwargs"""

        self.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        features = self.model(x)
        features_last = features[-1]  # Get the last feature map
        return features_last.flatten(start_dim=1)  # Flatten to have shape (batch_size, flattened_feature_num)

class MarioNet_ViT(nn.Module):
    def __init__(self,nb_actions=5,device="cpu"):
        super().__init__()
        self.feature_extractor = MobileViTFeatureExtractor(device)

        self.action_value1 = nn.Linear(self.feature_extractor.feature_dim,1024)
        self.action_value2 = nn.Linear(1024,1024)
        self.action_value3 = nn.Linear(1024,nb_actions)
        
        self.state_value1 = nn.Linear(self.feature_extractor.feature_dim,1024)
        self.state_value2 = nn.Linear(1024,1024)
        self.state_value3 = nn.Linear(1024,1)
        
        self.device = device
        self.relu = nn.ReLU()
        self.to(self.device)
    
    def forward(self,x):
        x = torch.tensor(x,device=self.device)

        #get the flattened features from the ViT
        features = self.feature_extractor(x)

        state_value = self.relu(self.state_value1(features))
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.state_value3(state_value)

        action_value = self.relu(self.action_value1(features))
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

"""
sample_input = torch.randn(1, 1, 256, 256)
model = MarioNet_ViT(nb_actions=5, device="cpu")
output = model(sample_input)
print(output.shape)
"""

#MobileViTFeatureExtractor()