#!/usr/bin/env python
# coding: utf-8

# ##  Cross Validation
# Inseption v3 model adapted from:
# 
# https://pytorch.org/docs/stable/torchvision/models.html
# 
# https://scikit-learn.org/stable/modules/cross_validation.html
# 
# https://scikit-learn.org/stable/modules/grid_search.html#grid-search
# 
# https://github.com/skorch-dev/skorch
# 

# In[1]:


from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

import os
import datetime

import numpy as np

from  data.image_dataset import JocondeDataset
import data.image_transforms as image_transforms
import model.train as train

import config as cfg

images_root = os.path.join(cfg.home_dir, 'Joconde/joconde')
descr_path = os.path.join(cfg.home_dir, 'Joconde/joconde', 'Humans and Horses')
dataset = 'dataset2_cv.csv'
dataset_name = 'human_vs_equidae'
image_description_file = os.path.join(descr_path, dataset)
folds_num = 10
is_disjoint = True
multi_label = False
batch_size = 4  

if (is_disjoint):
    exclude_labels = ['équidé+être humain', []]  
else:
    exclude_labels = []

model_name = 'inception_v3'
input_size = 299
finetuning = True
learning_rate = 0.001
epoch_count = 10
score_threshold = 0.5
criterion = nn.BCEWithLogitsLoss()
activation = torch.softmax
decision = train.decision_by_max
decision_param = 0.5

use_cuda = torch.cuda.is_available()

model_param_folder = './output/'
model_param_file = '%s_Joconde_%s.%s.cv.pth' % (model_name, dataset_name, '1000.2')


# %%

        
# In[43]:

def preload_dataset():
 
    
    dataset_mean = image_transforms.joconde_mean_animals 
    dataset_std  = image_transforms.joconde_std_animals 
    
    train_trans = transforms.Compose([
        #transforms.RandomResizedCrop(input_size),
        transforms.Resize(256), 
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = dataset_mean,
                             std = dataset_std),
    ])
    
    val_trans = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = dataset_mean,
                             std = dataset_std),
    ])

    
    folds = ['train' + str(x) for x in range(folds_num)]
    folds.remove('train0')
    
    train_set = JocondeDataset(image_description_file, 
                            images_root,
                            exclude_labels=exclude_labels ,
                            multiple_labels = multi_label,
                            filter_dict= {'usage': folds}, 
                            transform=train_trans)
    
    val_set = JocondeDataset(image_description_file, 
                            images_root,
                            exclude_labels=exclude_labels  ,
                            multiple_labels = multi_label, 
                            filter_dict= {'usage': ['train0']}, 
                            transform=val_trans)
   
    
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True,
                     num_workers=2)
    val_loader = torch.utils.data.DataLoader(
                    dataset=val_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2)
    
    
    class_names = val_set.classes
    
    print ('total trainning set size: {}'.format(len(train_set)))
    print ('total trainning batch count: {}'.format(len(train_loader)))
    
    print ('total validation set size: {}'.format(len(val_set)))
    print ('total validation batch count: {}'.format(len(val_loader)))
    
    
    print (class_names)

    return train_loader, val_loader

#%%

def cross_validation():
    
    output_dir = create_if_not_exsists(model_param_folder)
    
    train_loader, val_loader = preload_dataset()
    train_trans = train_loader.dataset.transform
    val_trans = val_loader.dataset.transform
    class_count = len(train_loader.dataset.classes)
    
    folds = ['train' + str(x) for x in range(folds_num)]
    
    model_accuracy = []
    total_elapsed_time = datetime.timedelta(0)
    
    for f, fold in enumerate(folds):
     
        print('Cross validation run %d of %d' % (f+1, len(folds)) )
        
        val_fold = [fold]
        train_folds = folds.copy()
        train_folds.remove(fold)
        
        train_loader.dataset = JocondeDataset(image_description_file, 
                                                images_root,
                                                exclude_labels=exclude_labels ,
                                                multiple_labels = multi_label,
                                                filter_dict= {'usage': train_folds}, 
                                                transform=train_trans)
    
        val_loader.dataset = JocondeDataset(image_description_file, 
                                                images_root,
                                                exclude_labels=exclude_labels  ,
                                                multiple_labels = multi_label, 
                                                filter_dict= {'usage': val_fold}, 
                                                transform=val_trans)
        
        net = train.load_net(model_name= model_name , class_count=class_count, finetuning=finetuning)
        
        if use_cuda:
            net = net.cuda()
        
        optimizer = train.set_optimizer(net, optim.SGD, learning_rate=learning_rate)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
            
        net, best_epoch, best_loss, best_accuracy, elapsed_time = train.train(model = net,
                                                                    criterion= criterion,
                                                                    optimizer=optimizer,
                                                                    scheduler= scheduler,
                                                                    train_loader = train_loader,
                                                                    val_loader=val_loader,
                                                                    activation_function = activation,
                                                                    decision_function = decision,
                                                                    decision_param = decision_param,
                                                                    epochs_num = epoch_count)
    
        print('Best model training accuracy: %.4f' % best_accuracy['train'])
        print('Best model validation accuracy: %.4f' % best_accuracy['val'])
        print('Best model epoch: %d' % (best_epoch + 1) )
        print('Training time: %s' % elapsed_time)
        
        best_val_acc = best_accuracy['val']
        
        if ( (len(model_accuracy) == 0) or (best_val_acc > max(model_accuracy)) ):
            torch.save(net.state_dict(), os.path.join(output_dir, model_param_file)) #save the best model
        
        model_accuracy.append(best_val_acc)
        total_elapsed_time += elapsed_time        
        
    print(model_accuracy)    
    print("Model Accuracy: %0.2f (+/- %0.2f)" % (np.asarray(model_accuracy).mean(), np.asarray(model_accuracy).std() * 2))
    print('Elapsed time: %s' % total_elapsed_time)
    
    output_file = 'CV__%s__%s' % (model_name , dataset)
    np.savetxt( os.path.join(output_dir, output_file) , np.asarray(model_accuracy), fmt='%.16f', delimiter=',')    

    print()
    print(net)
    print('Training params: ', len([x.requires_grad for x in net.parameters() if x.requires_grad == True]))

    
#%%
def create_if_not_exsists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

