# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:55:27 2020

@author: abobashe
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models

class AugmentedModel_v1(nn.Module):
    def __init__(self, class_count, add_data_dim, finetuning=False):
        super(AugmentedModel_v1, self).__init__()
         
        self.add_data_dim = add_data_dim
       
        # load object classification model
        self.aux_logits = True
        #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False) #models.inception_v3(pretrained=True, aux_logits=False)
        self.cnn = torchvision.models.inception_v3(pretrained=True, aux_logits=self.aux_logits)#, init_weights=False)
        
        if (not finetuning):
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        self.cnn.AuxLogits.fc = nn.Linear(self.cnn.AuxLogits.fc.in_features, class_count)
        self.classifier = nn.Linear( self.cnn.fc.in_features + self.add_data_dim, class_count)
        
        self.cnn.fc = nn.Identity()
        
       
    def forward(self, image, add_input):
        
        aux_defined = self.cnn.training and self.cnn.aux_logits
        
        if aux_defined:
            x1, aux = self.cnn(image)
        else:
            aux = None
            x1 = self.cnn(image)
        
        x2 = add_input
        
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        
        if aux_defined:
            return x, aux
        else:
            return x

#%%
class AugmentedModel_v2(nn.Module):
    def __init__(self, class_count, add_data_dim, finetuning=False):
        super(AugmentedModel_v2, self).__init__()
         
        self.add_data_dim = add_data_dim
       
        # load object classification model
        self.aux_logits = True
        #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False) #models.inception_v3(pretrained=True, aux_logits=False)
        self.cnn = torchvision.models.inception_v3(pretrained=True, aux_logits=self.aux_logits, init_weights=False)
        
        if (not finetuning):
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        self.cnn.AuxLogits.fc = nn.Linear(self.cnn.AuxLogits.fc.in_features, class_count)
        self.classifier = nn.Linear( self.cnn.fc.in_features + self.add_data_dim, class_count)
        
        self.cnn.fc = nn.Identity()
        
       
    def forward(self, image, add_input):
        
        aux_defined = self.cnn.training and self.cnn.aux_logits
        
        if aux_defined:
            x1, aux = self.cnn(image)
        else:
            aux = None
            x1 = self.cnn(image)
        
        x2 = add_input
        
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        
        if aux_defined:
            return x, aux
        else:
            return x
        
#%%
class AugmentedModel_v3(nn.Module):
    def __init__(self, class_count, add_data_dim, finetuning=False):
        super(AugmentedModel_v3, self).__init__()
         
        self.add_data_dim = add_data_dim
       
        # load object classification model
        self.aux_logits = True
        #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False) #models.inception_v3(pretrained=True, aux_logits=False)
        self.cnn = torchvision.models.inception_v3(pretrained=True, aux_logits=self.aux_logits, init_weights=False)
        
        if (not finetuning):
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        self.cnn.AuxLogits.fc = nn.Linear(self.cnn.AuxLogits.fc.in_features, class_count)
        self.classifier = nn.Linear( self.cnn.fc.in_features + self.add_data_dim, class_count)
        
        self.cnn.fc = nn.Identity()
        
       
    def forward(self, image, add_input):
        
        aux_defined = self.cnn.training and self.cnn.aux_logits
        
        if aux_defined:
            x1, aux = self.cnn(image)
        else:
            aux = None
            x1 = self.cnn(image)
        
        x2 = add_input
        
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        
        if aux_defined:
            return x, aux
        else:
            return x