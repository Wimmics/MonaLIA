#!/usr/bin/env python
# coding: utf-8

# ##  Model training
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
import torchvision.datasets as dset

import os

from  data.image_dataset import JocondeDataset
from  data.combined_dataset import CombinedDataset
from  data.binary_subset import BinarySubset
import data.image_transforms as image_transforms
import data.target_transforms as target_transforms
import model.train as train

from livelossplot import PlotLosses
import pandas as pd

import config as cfg


#parameters_ver = '2000.2'



#multi_label = True
#multi_crop = False
#batch_size = 4
#num_workers = 2 

#model_name = 'inception_v3'

#if(model_name == 'inception_v3'):
#    model_image_size = 299
#else:
#    model_image_size = 224
    
#model_finetuning = True

#optimization = optim.Adam
#initial_learning_rate = 0.0001
#learning_rate_decay_step = 4
#learning_rate_decay_rate = 0.1
#epoch_count = 100
#
#loss = nn.BCEWithLogitsLoss()
#use_weights = True
#
#activation_function = torch.sigmoid
#decision = train.decision_by_threshold
#decision_param = 0.5

#reusing this object as a container for the metrics
#it's also makes it easy to plot the loss if needed
liveloss = PlotLosses(max_epoch=cfg.args.epochs) #refactor

#import sys
#sys.stdout = open('temp', 'w')
#%%
def preload_subset(train_loader, val_loader, cls):
    
    train_subset = BinarySubset(train_loader.dataset, cls, 1000)
    val_subset = BinarySubset(val_loader.dataset, cls, 100)
    
    subset_train_loader = torch.utils.data.DataLoader(
                     dataset=train_subset,
                     batch_size= train_loader.batch_size,
                     shuffle=True,
                     drop_last = train_loader.drop_last,
                     num_workers=train_loader.num_workers)
    
    subset_val_loader = torch.utils.data.DataLoader(
                     dataset=val_subset,
                     batch_size= val_loader.batch_size,
                     shuffle=False,
                     drop_last = False, #val_loader.drop_last,
                     num_workers=val_loader.num_workers)
    
    print('Training', train_subset)
    print('    Labels:', train_subset.labels_count)
    print()
    
    print('Validation', val_subset)
    print('    Labels:',val_subset.labels_count)
    print()
    
    return    subset_train_loader,  subset_val_loader   
#%%
def preload_combined_dataset():
#TODO: Generalize
    train_loader, val_loader = preload_dataset()
    
    train_set = train_loader.dataset
    
    def _target_re_encoder (target, from_dataset, to_dataset):
        cls = from_dataset.classes[target]
        new_target = [0] * len(to_dataset.classes)
        new_target[train_set.class_to_idx[cls]] = 1
        return new_target

    kaggle_images_root = '~/Datasets/Kaggle'
    kaggle_images_root = os.path.realpath(os.path.normpath(os.path.expanduser(kaggle_images_root)))
    kaggle_train_set = dset.ImageFolder(root=os.path.join(kaggle_images_root, 'train'),
                                     transform=train_set.transform)
                       
    kaggle_train_set.samples = [(image, _target_re_encoder(target, kaggle_train_set, train_set)) for (image, target) in kaggle_train_set.samples]
    kaggle_train_set.targets = [s[1] for s in kaggle_train_set.samples]
    kaggle_train_set.classes = train_set.classes
    kaggle_train_set.class_to_idx = train_set.class_to_idx

    bam_images_root = '~/Datasets/BAM'
    bam_images_root = os.path.realpath(os.path.normpath(os.path.expanduser(bam_images_root)))
    bam_train_set = dset.ImageFolder(root=os.path.join(bam_images_root, 'train'),
                                     transform=train_set.transform)
    

    bam_train_set.samples = [(image, _target_re_encoder(target, bam_train_set, train_set)) for (image, target) in bam_train_set.samples]
    bam_train_set.targets = [s[1] for s in bam_train_set.samples]
    bam_train_set.classes = train_set.classes
    bam_train_set.class_to_idx = train_set.class_to_idx
    

    pin_images_root = '~/Datasets/Pinterest'
    pin_train_set = dset.ImageFolder(root=os.path.join(pin_images_root, 'train'),
                                     transform=train_set.transform)
    

    pin_train_set.samples = [(image, _target_re_encoder(target, pin_train_set, train_set)) for (image, target) in pin_train_set.samples]
    pin_train_set.targets = [s[1] for s in pin_train_set.samples]
    pin_train_set.classes = train_set.classes
    pin_train_set.class_to_idx = train_set.class_to_idx

    new_train_set = CombinedDataset([train_set, kaggle_train_set,  pin_train_set]) # bam_train_set,
    
    train_loader = torch.utils.data.DataLoader(
                     dataset=new_train_set,
                     batch_size= cfg.args.batch_size,
                     shuffle=True,
                     drop_last = True,
                     num_workers=cfg.args.workers)
    
    print('Combined Training', new_train_set)
    print('    Labels:', new_train_set.labels_count)
    print()
    
    return train_loader, val_loader

    

#%%
def preload_dataset():
#TODO: simplify with just imagenet means and std 
    if (cfg.args.arch == 'inception_v3') :
        dataset_mean =  [0.5, 0.5, 0.5]
        dataset_std  =  [0.5, 0.5, 0.5]
    
    elif cfg.args.arch == 'vgg16_bn':
        if cfg.args.database == 'Joconde':
            dataset_mean =  image_transforms.joconde_mean_animals 
            dataset_std  =  image_transforms.joconde_std_animals 
        else:
            dataset_mean =  image_transforms.imagenet_mean 
            dataset_std  =  image_transforms.imagenet_std
            
    else:
        dataset_mean =  image_transforms.imagenet_mean 
        dataset_std  =  image_transforms.imagenet_std
    
    train_trans = transforms.Compose([
        transforms.Resize(cfg.args.input_image_size + 32), 
        transforms.RandomResizedCrop(cfg.args.input_image_size),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean =  dataset_mean, std =   dataset_std ),
    ])

    val_trans = transforms.Compose([
        #transforms.Resize((model_image_size, model_image_size)), 
        transforms.Resize(cfg.args.input_image_size),
        transforms.CenterCrop(cfg.args.input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = dataset_mean, std = dataset_std),
    ])


    
    if cfg.args.database == 'Joconde':
        train_set = JocondeDataset(cfg.args.dataset_descr_file, 
                                    cfg.args.image_root,
                                    label_column= cfg.args.dataset_label_column,
                                    exclude_labels=cfg.args.dataset_exclude_labels,
                                    multiple_labels = cfg.args.multi_label,
                                    filter_dict= {'usage': ['train']}, 
                                    transform=train_trans)
    
        val_set = JocondeDataset(cfg.args.dataset_descr_file, 
                                    cfg.args.image_root,
                                    label_column= cfg.args.dataset_label_column,
                                    exclude_labels=cfg.args.dataset_exclude_labels,
                                    multiple_labels = cfg.args.multi_label, 
                                    filter_dict= {'usage': ['val']}, 
                                    transform=val_trans)
        
    elif cfg.args.database == 'ImageNet':
        #TODO: refactor; create a class
        train_set = dset.ImageFolder(root=os.path.join(cfg.args.images_root, 'train'), transform=train_trans)
        val_set = dset.ImageFolder(root=os.path.join(cfg.args.images_root, 'val'),  transform=val_trans)
        test_set = dset.ImageFolder(root=os.path.join(cfg.args.images_root, 'test'),  transform=val_trans)
        
        if cfg.args.multi_label:
            class_count = len(train_set.classes)
            train_set.target_transform = target_transforms.OneClassToOneHotOneClassToOneHot(class_count)
            val_set.target_transform = target_transforms.OneClassToOneHot(class_count)
            test_set.target_transform = target_transforms.OneClassToOneHot(class_count)
    
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size= cfg.args.batch_size,
                     shuffle=True,
                     drop_last = True,
                     num_workers=cfg.args.workers)
    
    val_loader = torch.utils.data.DataLoader(
                    dataset=val_set,
                    batch_size= cfg.args.batch_size,
                    shuffle=False,
                    drop_last = True,
                    num_workers=cfg.args.workers)
    
    print('Training', train_set)
    print('    Labels:', train_set.labels_count)
    print()
    
    print('Validation', val_set)
    print('    Labels:',val_set.labels_count)
    print()

    return train_loader, val_loader

#%%
def add_themes_to_datasets(datasets):
    
    themes_descr_path = '~/Datasets/Joconde/Themes'
    themes_image_description_file = os.path.join(themes_descr_path, 'dataset2.full.csv')

    themes_set = JocondeDataset(themes_image_description_file, 
                            cfg.args.image_root,
                            label_column='theme_label',
                            #exclude_labels=exclude_labels  ,
                            multiple_labels = True)
                            #filter_dict= {'usage': ['train']}, 
                            #transform=train_trans)
    print(themes_set.classes)
    
    for data_set in datasets:
        for i, sample in enumerate(data_set.samples):
            #print(sample)
            theme = [item for item in themes_set.samples if item[2] == sample[2]] #[0][1]
            #print(theme)
            if len(theme) > 0:
                theme = theme [0][1]
                #print(theme)
            else:
                theme = [0] * len(themes_set.classes)
                #print(theme)
            if len(data_set.samples[i]) < 5:
                data_set.samples[i] = sample + (theme,)
        #print(train_set.samples[i])
        #break
    print('modified datasets')
    
    return datasets
#%%

def model_training():
    
    model_param_folder = os.path.dirname(cfg.args.model_param_file)
    create_if_not_exsists(model_param_folder)
    
    train_loader, val_loader = preload_dataset()
    class_count = len(train_loader.dataset.classes)
    
    #####################################################################
    # Modify datasets
    #
    # add_themes_to_datasets([train_loader.dataset, val_loader.dataset])
    # print(train_loader.dataset.samples[0])
    #####################################################################       
    net = train.load_net(model_name= cfg.args.arch ,
                         class_count=class_count,
                         finetuning=cfg.args.model_finetuning)

    #####################################################################
    # load parameters from other model 
    #
    #model_checkpoint_file = './output/inception_v3_Joconde_40_classes.1000.no_sched.checkpoint.pth.tar'
    #checkpoint = torch.load(model_checkpoint_file)
    #print('Load checkpoint: ', model_checkpoint_file, checkpoint.keys())
    #net.load_state_dict(checkpoint['state_dict'])
    #####################################################################

    print('Model:', type(net))
    #print(net)
    print('    Training Parameters: ', len([x.requires_grad for x in net.parameters() if x.requires_grad == True]))
    
    activation = train.set_activation(cfg.args.activation)
    print('Activation:', activation)
    
    criterion = train.set_loss_criterion(cfg.args.loss, cfg.args.use_weights, train_loader)
    print('Loss function:', criterion, (criterion.pos_weight.data if (cfg.args.use_weights and isinstance(criterion.pos_weight, torch.Tensor) ) else 'No weights'))
           
    optimizer = train.set_optimizer(net, cfg.args.optim, learning_rate=cfg.args.lr, weight_decay= cfg.args.weight_decay, momentum=cfg.args.momentum )
    print('Optimizer:', optimizer)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size= cfg.args.lr_scheduler[0],
                                          gamma=cfg.args.lr_scheduler[1])
    print('LR Scheduler: step=',  scheduler.step_size, 'gamma=', scheduler.gamma)
    
    
    decision, decision_param = train.set_decision(cfg.args.decision, cfg.args.decision_param)
    print('Accuracy decision: func=',  decision, 'param=', decision_param)
   
    #net, best_epoch, best_loss, best_accuracy, elapsed_time = train.train(  model = net,
    net, score_th, elapsed_time = train.train(  model = net,
                                                criterion= criterion,
                                                optimizer=optimizer,
                                                scheduler= scheduler,
                                                train_loader = train_loader,
                                                val_loader=val_loader,
                                                activation_function = activation,
                                                decision_function = decision,
                                                decision_param = decision_param,
                                                save_best_model=True,
                                                epochs_num = cfg.args.epochs,
                                                end_of_epoch_callback=end_epoch,
                                                print_frequency=cfg.args.print_freq,
                                                checkpoint_dir = cfg.args.output_dir,
                                                checkpoint_prefix = os.path.splitext(os.path.basename(cfg.args.model_param_file))[0])


    pd.options.display.float_format = '{:,.3f}'.format
    pd.options.display.max_rows = 999
    logs_df = pd.DataFrame(liveloss.logs)
    #logs_df.to_csv(cfg.args.train_metrics_file)
    print(logs_df)
    print()
    
    print('Best model training accuracy: %.4f [epoch %d]'   % (logs_df.accuracy.max(), logs_df.accuracy.idxmax() + 1))
    print('Best model validation accuracy: %.4f [epoch %d]' % (logs_df.val_accuracy.max() , logs_df.val_accuracy.idxmax() + 1))
    print('Best model validation mAP: %.4f [epoch %d]'      % (logs_df.val_mAP.max(), logs_df.val_mAP.idxmax() + 1))
    print('Training time: %s' % elapsed_time)
   
     
#%%
def end_epoch (epoch_stats):
    liveloss.update(epoch_stats)
    
#%%
def create_if_not_exsists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

#%%

def binary_model_training():
    
    model_param_folder = os.path.dirname(cfg.args.model_param_file)
    create_if_not_exsists(model_param_folder)
    
    train_loader, val_loader = preload_dataset()
    class_count = 2 #len(train_loader.dataset.classes)
            
    net = train.load_net(model_name= cfg.args.arch ,
                         class_count=class_count,
                         finetuning=cfg.args.model_finetuning)
    print('Model:', type(net))
    #print(net)
    print('    Training Parameters: ', len([x.requires_grad for x in net.parameters() if x.requires_grad == True]))
    
    activation = train.set_activation('sigmoid')
    print('Activation:', activation)
    
    criterion = train.set_loss_criterion('BCE')
    print('Loss function:', criterion, (criterion.pos_weight.data if isinstance(criterion.pos_weight, torch.Tensor) else 'No weights'))
           
    optimizer = train.set_optimizer(net, cfg.args.optim, learning_rate=cfg.args.lr, weight_decay= cfg.args.weight_decay, momentum=cfg.args.momentum )
    print('Optimizer:', optimizer)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size= cfg.args.lr_scheduler[0],
                                          gamma=cfg.args.lr_scheduler[1])
    print('LR Scheduler: step=',  scheduler.step_size, 'gamma=', scheduler.gamma)
    
    
    decision, decision_param = train.set_decision('threshold', 0.5)
    print('Accuracy decision: func=',  decision, 'param=', decision_param)
   
    for one_class in train_loader.dataset.classes:
        print('*' * 120)
        print('    ', one_class)
        print('*' * 120)
        
        subset_train_loader, subset_val_loader = preload_subset(train_loader, val_loader, one_class)
        
        net = train.load_net(model_name= cfg.args.arch ,
                         class_count=class_count,
                         finetuning=cfg.args.model_finetuning)

        optimizer = train.set_optimizer(net, cfg.args.optim, learning_rate=cfg.args.lr, weight_decay= cfg.args.weight_decay, momentum=cfg.args.momentum )
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size= cfg.args.lr_scheduler[0],
                                              gamma=cfg.args.lr_scheduler[1])
        
        prefix =  '%s_%s_%s.%s.%s' % (cfg.args.arch, cfg.args.database, cfg.args.dataset, one_class,  cfg.args.param_file_suffix)

        net, score_th, elapsed_time = train.train(  model = net,
                                                    criterion= criterion,
                                                    optimizer=optimizer,
                                                    scheduler= scheduler,
                                                    train_loader = subset_train_loader,
                                                    val_loader=subset_val_loader,
                                                    activation_function = activation,
                                                    decision_function = decision,
                                                    decision_param = decision_param,
                                                    save_best_model=False,
                                                    epochs_num = cfg.args.epochs,
                                                    end_of_epoch_callback=end_epoch,
                                                    print_frequency=cfg.args.print_freq,
                                                    checkpoint_dir = cfg.args.output_dir,
                                                    checkpoint_prefix = prefix)
    
        pd.options.display.float_format = '{:,.3f}'.format
        pd.options.display.max_rows = 999
        logs_df = pd.DataFrame(liveloss.logs)
        #logs_df.to_csv(os.path.join(cfg.args.output_dir, prefix + '.csv') )
        print(logs_df)
        print()
    

        print('Best model training accuracy: %.4f' % logs_df.accuracy.max())
        print('Best model validation accuracy: %.4f' % logs_df.val_accuracy.max())
        print('Best model epoch: %d' %  (logs_df.val_accuracy.idxmax() + 1))
        print('Training time: %s' % elapsed_time)
  
        liveloss.logs.clear()