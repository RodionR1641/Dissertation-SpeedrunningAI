#from torchvision.models import vit_tiny_patch16_224
#from timm.models.tiny_vit import tiny_vit_5m  # Actual TinyViT (5M params)

import timm
from timm.layers.conv_bn_act import ConvNormAct 
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np



#extract the features of the image first using this hybrid Cnn Vit
class MobileViTFeatureExtractor(nn.Module):
    def __init__(self,device="cpu",model_name="vit_tiny_patch16_224"):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        #self.feature_dim = self.model.head.in_features #the last layer size flattened
        self.device = device

        #this is to make the model accept greyscale input instead of rgb 3 channel input

        # Modify input channels
        original_conv = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(
            in_channels=4,  # Your input channels
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )


        self.feature_dim = self.model.embed_dim
        self.to(self.device)
    
    def forward(self, x):

        # ViT forward
        x = self.model.patch_embed(x)
        return x[:, 0]  # Return class token
        

class MarioNet(nn.Module):
    def __init__(self,envs,input_shape,device="cpu"):
        super().__init__()
        self.feature_extractor = MobileViTFeatureExtractor(device)


        # ViT's embedding dimension is 192
        self.projection = nn.Sequential(
            layer_init(nn.Linear(self.feature_extractor.feature_dim, 512)),
            nn.ReLU()
        )

        self.critic = layer_init(nn.Linear(512,1),std=1)
        self.actor = layer_init(nn.Linear(512,envs.single_action_space.n),std=0.01)
        
        self.device = device
        self.relu = nn.ReLU()
        self.to(self.device)
    
    def get_value(self,x):
        if x.device != self.device:
            x = x.to(self.device)
        #divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1
        hidden = self.projection(self.feature_extractor(x/255.0))
        return self.critic(hidden) #go through cnn first then critic

    def get_action_plus_value(self,x,action=None):  
        if x.device != self.device:
            x.to(self.device)
        #divide by 255 -> the image observation has a range 0-255, we get it range of 0 to 1
        hidden = self.projection(self.feature_extractor(x / 255.0)) #get the hidden layer output, after CNN input
        logits = self.actor(hidden) #unnormalised action probabilities
        probabilities = Categorical(logits=logits) #softmax operation to get the action probability distribution we need
        if action is None:
            action = probabilities.sample()
        #return actions, log probabilities, entropies and values from critic
        return action,probabilities.log_prob(action), probabilities.entropy(),self.critic(hidden)



def layer_init(layer, std=np.sqrt(2), bias_const=0.0): #use sqrt 2 as standard deviation
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer
