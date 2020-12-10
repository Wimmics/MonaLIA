# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:37:26 2020

@author: abobashe
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models

from .identity import Identity


class EnsembleModel(nn.Module):
    def __init__(self, class_count, input_model_checkpoint_file, finetuning=False):
        super(EnsembleModel, self).__init__()
        
        #  load themes model asinput model
        self.input_cnn = torchvision.models.inception_v3(pretrained=False, aux_logits=True, init_weights=False)

        checkpoint = torch.load(input_model_checkpoint_file)
        
        self.add_data_dim = len(checkpoint['classes'])

        self.input_cnn.AuxLogits.fc = nn.Linear(self.input_cnn.AuxLogits.fc.in_features, self.add_data_dim)
        self.input_cnn.fc = nn.Linear(self.input_cnn.fc.in_features, self.add_data_dim)
            
        self.input_cnn.load_state_dict(checkpoint['state_dict'])
        
        self.input_cnn.aux_logits = False
        self.input_cnn.eval()
        
        # load object classification model
        self.aux_logits = True
        #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False) #models.inception_v3(pretrained=True, aux_logits=False)
        self.cnn = torchvision.models.inception_v3(pretrained=True, aux_logits=self.aux_logits, init_weights=False)
        
        #self.cnn.transform_input = False
        if (not finetuning):
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        self.cnn.AuxLogits.fc = nn.Linear(self.cnn.AuxLogits.fc.in_features, class_count)
        self.classifier = nn.Linear( self.cnn.fc.in_features + self.add_data_dim, class_count)
        
        self.cnn.fc = Identity()
        
       
    def forward(self, image):
        
        aux_defined = self.cnn.training and self.cnn.aux_logits
        
        if aux_defined:
            x1, aux = self.cnn(image)
        else:
            aux = None
            x1 = self.cnn(image)
        
        self.input_cnn.eval()
        x2 = self.input_cnn(image)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        
        if aux_defined:
            return x, aux
        else:
            return x

        #return x, aux
 #%%       
class EnsembleModel_v2(nn.Module):
    def __init__(self, class_count, input_model_checkpoint_file, finetuning=False):
        super(EnsembleModel_v2, self).__init__()
        
        #  load themes model asinput model
        self.input_cnn = torchvision.models.inception_v3(pretrained=False, aux_logits=True, init_weights=False)
        checkpoint = torch.load(input_model_checkpoint_file)
        
        self.add_data_dim = len(checkpoint['classes'])

        self.input_cnn.AuxLogits.fc = nn.Linear(self.input_cnn.AuxLogits.fc.in_features, self.add_data_dim)
        self.input_cnn.fc = nn.Linear(self.input_cnn.fc.in_features, self.add_data_dim)
            
        self.input_cnn.load_state_dict(checkpoint['state_dict'])
        
        self.input_cnn.aux_logits = False
        self.input_cnn.eval()
        
        # load object classification model
        self.aux_logits = True
        #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False) #models.inception_v3(pretrained=True, aux_logits=False)
        self.cnn = torchvision.models.inception_v3(pretrained=True, aux_logits=self.aux_logits, init_weights=False)
        
        #self.cnn.transform_input = False
        if (not finetuning):
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        self.cnn.AuxLogits.fc = nn.Linear(self.cnn.AuxLogits.fc.in_features, class_count)
        self.cnn.fc = nn.Linear( self.cnn.fc.in_features, class_count)
        
        self.classifier = nn.Linear( self.cnn.fc.out_features + self.add_data_dim, class_count)
        
       
    def forward(self, image):
        
        aux_defined = self.cnn.training and self.cnn.aux_logits
        
        if aux_defined:
            x1, aux = self.cnn(image)
        else:
            aux = None
            x1 = self.cnn(image)
        
        self.input_cnn.eval()
        x2 = self.input_cnn(image)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        
        if aux_defined:
            return x, aux
        else:
            return x

        #return x, aux
        
#%%      
class EnsembleModel_v3(nn.Module):
    def __init__(self, modelA , modelB):
        super(EnsembleModel_v3, self).__init__()
        
        self.modelA = modelA
        self.modelB = modelB
       
        #TODO: generalize  . For now assume inception
        self.modelA_output_features = list(self.modelA.children())[-1].out_features
        self.modelB_output_features = list(self.modelB.children())[-1].out_features
        
        self.classifier = nn.Linear(self.modelA_output_features + self.modelB_output_features, self.modelB_output_features)
        
        #freeze layers
        for param in self.modelA.parameters():
            param.requires_grad = False
        for param in self.modelB.parameters():
            param.requires_grad = False
            
        self.modelA.eval()
        self.modelB.eval()
        
        self.aux_logits = self.modelA.aux_logits = self.modelB.aux_logits = False
        
    def forward(self, x):
        x1 = self.modelA(x.clone())
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x