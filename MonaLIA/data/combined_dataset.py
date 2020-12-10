# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:24:57 2019

@author: abobashe
"""

import os

import numpy as np
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.utils.data.dataset 

class CombinedDataset(torch.utils.data.dataset.ConcatDataset):
    """
      Extension of the pytorch ConcatDataset
    """
    def __init__(self, datasets):
        super(CombinedDataset, self ).__init__(datasets)
        #assert len(datasets) > 0, 'datasets should not be an empty iterable'
        #self.datasets = list(datasets)
        #for d in self.datasets:
        #    assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        #self.cumulative_sizes = self.cumsum(self.datasets)

        self.classes = datasets[0].classes
        self.class_to_idx = datasets[0].class_to_idx
        self.targets = [y for x in self.datasets for y in x.targets] 
        self.samples = [y for x in self.datasets for y in x.samples]
        self.imgs = self.samples
        
        if isinstance(self.targets[0], int): #single label
            labels_per_class  = [sum([x == i  for x in self.targets])  for i in range(len(self.classes))]
        else:                                #multi label 
            labels_per_class = [sum([x[i] for x in self.targets]) for i in range(len(self.classes))]

        self.labels_count = dict(zip(self.classes, labels_per_class))
        
    def extra_repr(self):
        return '\n'.join(['Number of classes: {}'.format(len(self.classes)),
                          'Number of uniqie labels: {}'.format(np.unique(np.array(self.targets), axis=0).shape[0],
                          'Number of class labels: {}'.format(sum(self.labels_count.values())))])

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        for d in self.datasets:
            if d.root is not None:
                body.append("Root location: {}".format(d.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self.datasets[0]._repr_indent + line for line in body]
        return '\n'.join(lines)