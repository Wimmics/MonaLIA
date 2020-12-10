# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:52:41 2020

@author: abobashe
"""
import torch
import torch.nn as nn
#from torch.utils import data
#import torchvision
from torchvision.models import vgg16_bn, inception_v3
#from torchvision import transforms
#from torchvision import datasets
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.image as mpimg
#import cv2
import copy

#import sys
#import os

#import pandas as pd

#%%
class GradCAM_Inception(nn.Module):
    def __init__(self, class_count, param_file=None):
        super(GradCAM_Inception, self).__init__()
        
        # get the pretrained Inceptionv3 network
        self.net = inception_v3(pretrained=(param_file is None))
        self.net.transform_input = False
        
        # modify the last layer 
        self.net.AuxLogits.fc = nn.Linear(self.net.AuxLogits.fc.in_features, class_count)
        self.net.fc = nn.Linear(self.net.fc.in_features, class_count)
                
        # read trained parameters
        if(param_file is not None):
            if (param_file.endswith('pth')):
                self.net.load_state_dict(torch.load(param_file, map_location=lambda storage, loc: storage))
            else:
                checkpoint = torch.load(param_file)
                self.net.load_state_dict(checkpoint['state_dict'])
        
        # placeholder for the gradients
        self.gradients = None
        
        # placeholder for the activations
        self.activations = None
   
        self.input = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
    
    # forward function is copied from the pytorch documentation 
    # TODO: verify with actual code
    def forward(self, x):
        
        x = self._transform_input(x)
        
        self.input = copy.deepcopy(x)
        
        # N x 3 x 299 x 299
        x = self.net.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.net.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.net.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.net.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.net.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.net.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.net.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.net.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.net.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6e(x)
        # N x 768 x 17 x 17
        ###aux_defined = self.net.training and self.net.aux_logits
        ###if aux_defined:
        ###    aux = self.net.AuxLogits(x)
        ###else:
        ###    aux = None
        # N x 768 x 17 x 17
        x = self.net.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.net.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.net.Mixed_7c(x)
        # N x 2048 x 8 x 8
        
        # ADDED: register the hook
        x.register_hook(self.activations_hook)
                # placeholder for the activations
        self.activations = x
        
        # apply the remaining pooling
        
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.net.fc(x)
        # N x 1000 (num_classes)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self):
        return self.activations

    def _transform_input(self, x):
        if self.net.transform_input:
            
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x
    
#%%
class GradCAM_VGG(nn.Module):
    def __init__(self, class_count, param_file=None):
        super(GradCAM_VGG, self).__init__()
        
        # get the pretrained VGG16 network
        self.vgg = vgg16_bn(pretrained=(param_file is None))
        
        # modify the last layer 
        self.vgg.classifier[-1] = nn.Linear(self.vgg.classifier[-1].in_features, class_count) 
        
        # read trained parameters
        if(param_file is not None):
            self.vgg.load_state_dict(torch.load(param_file, map_location=lambda storage, loc: storage))
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:43]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg16
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None

    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
#%%
def run_gradCAM(img, model, class_index = None, counterfact = False):
    
    pred = model(img)
    
    if class_index == None:
        pred.max().backward()
    else:
        pred[0, class_index].backward()

    gradients = model.get_activations_gradient()
    
    if counterfact:
        
        gradients *= -1
        
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations().detach()

    n_channels = gradients.shape[1]
    
    # weight the channels by corresponding gradients
    for i in range(n_channels):

        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
   
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    #image_size = img.shape[2]
    #heatmap = cv2.resize(np.float32(heatmap), dsize=(image_size,image_size))
    
    return pred.detach(), heatmap