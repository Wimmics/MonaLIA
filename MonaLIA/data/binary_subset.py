# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:34:03 2019

@author: abobashe
"""
#import os

import numpy as np
#import pandas as pd
#import torchvision.datasets as dset
#import torchvision.transforms as transforms

import torch.utils.data.dataset 

class BinarySubset(torch.utils.data.dataset.Subset):
    """
      Extension of the pytorch Subset
    """
    def __init__(self, dataset, subset_class, max_size=None):
        #self.dataset = ds
        #self.indices = indices
        self.class_idx =  dataset.class_to_idx[subset_class]
        
        targets_array = np.array(dataset.targets)
        self.subset_idx_pos = np.argwhere(targets_array[:, self.class_idx ] == 1).flatten()
        if max_size is not None:
            self.subset_idx_pos = self.subset_idx_pos[np.random.randint(0, self.subset_idx_pos.shape[0], size=min(self.subset_idx_pos.shape[0], max_size))]
        
        self.subset_idx_neg = np.argwhere(targets_array[:, self.class_idx ] != 1)
        self.subset_idx_neg = self.subset_idx_neg[np.random.randint(0, self.subset_idx_neg.shape[0], size=self.subset_idx_pos.shape[0])]
        self.subset_idx_neg = self.subset_idx_neg.flatten()
        
        subset_idx =  np.concatenate( (self.subset_idx_pos, self.subset_idx_neg) )
        
        super(BinarySubset, self ).__init__(dataset , subset_idx)
                      
        self.classes = ['~' + subset_class, subset_class]
        self.class_to_idx = { '~' + subset_class: 0, subset_class: 1}
    
#        self.samples = [ (self.dataset.samples[x][0], 
#                          [self.dataset.samples[x][1][self.class_idx] ^ 1,
#                           self.dataset.samples[x][1][self.class_idx] ] )   for x in self.indices ]
        self.samples = [ self.__transform_sample(self.dataset.samples[i])  for i in self.indices ]
        self.targets = self.targets = [s[1] for s in self.samples]
    
        self.imgs = self.samples
        
        #self.transform = self.dataset.transform

        self.labels_count = dict(zip(self.classes, [self.subset_idx_neg.shape[0] , self.subset_idx_pos.shape[0]]))
        
    def __getitem__(self, idx):
        
        img, target = self.dataset[self.indices[idx]]    
        
        return img, [ target[self.class_idx] ^ 1 , target[self.class_idx]]
    
    @property    
    def transform(self):
        return self.dataset.transform

    def extra_repr(self):
        return '\n'.join(['Number of classes: {}'.format(len(self.classes)),
                          'Number of positive labels: {}'.format(self.subset_idx_pos.shape[0]),
                          'Number of negative labels: {}'.format(self.subset_idx_neg.shape[0]) ] )

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.dataset.root is not None:
                body.append("Root location: {}".format(self.dataset.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self.dataset._repr_indent + line for line in body]
        return '\n'.join(lines)
    
    def __transform_sample(self, sample):
        sample_list = list(sample)
        sample_list[1] = [sample[1][self.class_idx] ^ 1 , sample[1][self.class_idx] ]
        return tuple(sample_list)
        