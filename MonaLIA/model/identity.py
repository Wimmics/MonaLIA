# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:44:02 2019

@author: abobashe
"""

import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x