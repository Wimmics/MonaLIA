# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:22:49 2020

@author: abobashe
"""


#%%
import torch
import torchvision

import os

import model.ensemble

dataset= torchvision.datasets.FakeData(size=12, image_size=(3, 299, 299), num_classes=2, transform=torchvision.transforms.ToTensor(), target_transform=None, random_offset=0)

dl = torch.utils.data.DataLoader(dataset, batch_size=4)

i, (img, tgt) = next(enumerate(dl))



os.getcwd()
dir = 'output'
filename = 'inception_v3_Joconde_themes.1000.2.checkpoint.pth.tar'
checkpoint_file = os.path.join(dir, filename)
net = model.ensemble.EnsembleModel(class_count=2, input_model_checkpoint_file=checkpoin_file)

net(img)
#%%
#convert to one-hot encoding
np.eye(15, dtype=np.dtype('I'))[test_set.targets]