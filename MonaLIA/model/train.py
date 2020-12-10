# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:46:34 2019

@author: abobashe
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models

import numpy as np

import sklearn.metrics as metrics

import shutil

import os
#import sys
import copy
import datetime

import warnings

from .ensemble import EnsembleModel, EnsembleModel_v2, EnsembleModel_v3

from .augmented import AugmentedModel_v1, AugmentedModel_v2, AugmentedModel_v3

#%%

def load_net (model_name = "vgg16_bn", class_count=2, finetuning=False):
    
    if model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        net = torchvision.models.inception_v3(pretrained=True, init_weights=False)
        net.transform_input = False
        if (not finetuning):
            for param in net.parameters():
                param.requires_grad = False

        net.AuxLogits.fc = nn.Linear(net.AuxLogits.fc.in_features, class_count)
        net.fc = nn.Linear(net.fc.in_features, class_count)

    elif model_name == 'vgg16_bn':
        net = torchvision.models.vgg16_bn(pretrained=True)
        
        if (not finetuning):
            for param in net.parameters():
                param.requires_grad = False
        
        # Replace the last layer and it will unfreeze it as well
        net.classifier[len(net.classifier)-1] = nn.Linear(net.classifier[len(net.classifier)-1].in_features, class_count)    
    
        # Attach this attribute to the network to be able to distinguish between having and not having the auxilary output
        net.__setattr__('aux_logits', False)
        
    elif model_name == 'resnet_50':
        net = torchvision.models.resnet50(pretrained=True)
        
        if (not finetuning):
            for param in net.parameters():
                param.requires_grad = False
        
        # Replace the last layer and it will unfreeze it as well
        net.fc = nn.Linear(net.fc.in_features, class_count)
    
        # Attach this attribute to the network to be able to distinguish between having and not having the auxilary output
        net.__setattr__('aux_logits', False)   
        
    elif model_name == 'resnet_101':
        net = torchvision.models.resnet101(pretrained=True)
        
        if (not finetuning):
            for param in net.parameters():
                param.requires_grad = False
        
        # Replace the last layer and it will unfreeze it as well
        net.fc = nn.Linear(net.fc.in_features, class_count)
    
        # Attach this attribute to the network to be able to distinguish between having and not having the auxilary output
        net.__setattr__('aux_logits', False)   
        
    elif model_name == 'wide_resnet_101':
        net = torchvision.models.wide_resnet101_2(pretrained=True)
        
        if (not finetuning):
            for param in net.parameters():
                param.requires_grad = False
        
        # Replace the last layer and it will unfreeze it as well
        net.fc = nn.Linear(net.fc.in_features, class_count)
    
        # Attach this attribute to the network to be able to distinguish between having and not having the auxilary output
        net.__setattr__('aux_logits', False)  
        
    elif model_name == 'ensemble':
        print(os.getcwd())
        dir = 'output'
        filename = 'inception_v3_Joconde_themes.1000.4.best.pth.tar'
        checkpoin_file = os.path.join(dir, filename)
        net = EnsembleModel(class_count, input_model_checkpoint_file = checkpoin_file,finetuning=finetuning)
        
    elif model_name == 'ensemble_v2':
        print(os.getcwd())
        dir = 'output'
        filename = 'inception_v3_Joconde_themes.1000.3.checkpoint.pth.tar'
        checkpoin_file = os.path.join(dir, filename)
        net = EnsembleModel_v2(class_count, input_model_checkpoint_file = checkpoin_file,finetuning=finetuning)
        
    elif model_name == 'ensemble_v3':
        print(os.getcwd())
        ############# pretrained model A
        themes_model_checkpoint_file = os.path.join('output','inception_v3_Joconde_themes.1000.3.checkpoint.pth.tar')
        themes_checkpoint = torch.load(themes_model_checkpoint_file)
        
        net_themes = torchvision.models.inception_v3(pretrained=False, init_weights=False)

        net_themes.AuxLogits.fc = nn.Linear(net_themes.AuxLogits.fc.in_features, len(themes_checkpoint['classes']))
        net_themes.fc = nn.Linear(net_themes.fc.in_features, len(themes_checkpoint['classes']))
        net_themes.load_state_dict(themes_checkpoint['state_dict'])
        
        ############# pretrained model B
        objects_model_checkpoint_file = os.path.join('output' , 'inception_v3_Joconde_10_classes.1000.21.checkpoint.pth.tar')
        objects_checkpoint = torch.load(objects_model_checkpoint_file)
        
        net_objects = torchvision.models.inception_v3(pretrained=False, init_weights=False)

        net_objects.AuxLogits.fc = nn.Linear(net_objects.AuxLogits.fc.in_features, len(objects_checkpoint['classes']))
        net_objects.fc = nn.Linear(net_objects.fc.in_features, len(objects_checkpoint['classes']))
        net_objects.load_state_dict(objects_checkpoint['state_dict'])
        
        ############# ensemble model
        net = EnsembleModel_v3(net_themes, net_objects)
        
    elif model_name == 'augmented_v1':  
        net = AugmentedModel_v1(class_count, 18, finetuning=finetuning)
        
    return  net   
# %%

def set_optimizer(model, optimizer_type, learning_rate = 0.001, momentum=0.9, weight_decay=0, print_parameters= False):
    """
        argument optimizer_type can be string or optimizer function
    """
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            if print_parameters:
                print("\t",name)

    if isinstance(optimizer_type, str):
        optim_map = {'SGD':  optim.SGD, 
                     'ADAM': optim.Adam }
        optimizer_type = optim_map[optimizer_type.upper()]
  
    if optimizer_type == optim.SGD : 
        optimizer = optimizer_type(params_to_update, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optimizer_type(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    
    return optimizer

#%% 
def set_loss_criterion(criterion, use_weights=False, data_loader=None):
    """
        argument criterion can be string or loss function
    """
    if isinstance(criterion, str):
        loss_map = { 'crossentropy' : nn.CrossEntropyLoss(),
                     'bce' : nn.BCEWithLogitsLoss(),
                     'binarycrossentropy' : nn.BCEWithLogitsLoss()}
        criterion = loss_map[criterion.lower()]
        
    if use_weights and data_loader == None: 
        raise TypeError('data is required to calculate weights')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if(use_weights):
        pos_weight = calculate_positive_weights(data_loader)
        if isinstance(criterion, nn.BCEWithLogitsLoss):     
            criterion.pos_weight = pos_weight.to(device)
      
    #elif isinstance(criterion, nn.CrossEntropyLoss):     
       #TODO: find out the proper weight calculation
       # chcek this out: sklearn.utils.class_weight.compute_class_weight('balanced' , [0,1,2,3], train_set.targets[:2800]) 
       #criterion.weight = pos_weight.to(device)
      
    return criterion

def set_activation(activation_function):
    """
        argument activation_function can be string or activation function
    """
    if isinstance(activation_function, str):
        act_map = { 'softmax': __softmax,
                    'sigmoid' : torch.sigmoid}
        activation_function = act_map[activation_function.lower()] 

    if (activation_function.__name__ == 'softmax'):
        activation_function = __softmax
        
    return activation_function

# TODO: refactor using the callable object
def set_decision(decision_function, decision_param):
    """
        argument decision_function can be string or function
        argument decision_param can be string or function
    """
    decision = decision_function
    if isinstance(decision_function, str):
        func_map = { 'threshold': decision_by_threshold,
                     'threshold-per-class' : decision_by_class_threshold,
                     'top-k' :    decision_by_topk,
                     'max' : decision_by_max}
        
        decision = func_map[decision_function.lower()] 
   
    param = decision_param 
    if isinstance(decision_param, str):   
        param_map = { 'threshold': float,
                      'threshold-per-class': float, ##TODO Take care of list 
                      'top-k' :    int}
        param = param_map[decision_function.lower()](decision_param)
        
    return decision, param
    

#%%
def calculate_positive_weights(data_loader):
    train_set = data_loader.dataset
    class_count = len(train_set.classes)
    
    if isinstance(train_set.targets[0], int): #single label
        #TODO: np.bincount
        train_labels_per_class = [sum([x == i  for x in train_set.targets]) for i in range(class_count)]
    else:                                    #multi label 
        train_labels_per_class = [sum([x[i] for x in train_set.targets]) for i in range(class_count)]
        
    tot = sum(train_labels_per_class)
    pos = np.array(train_labels_per_class)
    #neg = tot - pos
    pos_weight =  torch.Tensor((tot - pos) / pos)    
   
    return pos_weight


#%%
t0 = datetime.datetime.now()  

def train(model, 
            criterion, optimizer, scheduler, 
            train_loader, val_loader,  
            activation_function, decision_function , decision_param,
            epochs_num = 10,
            end_of_epoch_callback=None,
            print_frequency=1000,
            save_best_model=False ,
            checkpoint_dir = '..\output',
            checkpoint_prefix = ''):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
        
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_mAP = 0.0
    is_best = False
    
    metrics_history = []
    
    global t0
    t0 = datetime.datetime.now()
    
    th = 0.5
    
    for epoch in range(epochs_num):

        print()
        print('Epoch %d of %d' % (epoch+1, epochs_num) )
    
        eval_metrics = {}
        
        print('****** train ******')
        train_loss, train_acc, _ = train_epoch(model, 
                                              train_loader,
                                              criterion, optimizer, scheduler,
                                              activation_function,
                                              decision_function , decision_param,
                                              print_frequency=print_frequency )
        
        #store metrics values for plotting
        eval_metrics['loss'] = train_loss
        eval_metrics['accuracy'] = train_acc
       
        # evaluate on validation set
        print('****** validate ******')
        val_loss, val_acc, _, val_scores = validate(model, 
                                                    val_loader,  
                                                    criterion, 
                                                    activation_function,
                                                    decision_function , decision_param,
                                                    print_frequency=print_frequency )
            
        #store values for plotting
        eval_metrics['val_loss'] = val_loss
        eval_metrics['val_accuracy'] = val_acc
        
        #calculate mAP
        val_mAP = 0.0
        if (val_scores.nelement() > 0 ):
            y_true = np.array(val_loader.dataset.targets[:val_scores.shape[0]])
            
            if (y_true.ndim == 1):
                y_score = val_scores.cpu().detach().numpy()[:, 1]
            else:
                y_score = val_scores.cpu().detach().numpy()
            
            if(isinstance(val_loader.dataset[0][1], list)): #multi-label only
                val_mAP = metrics.average_precision_score(y_true, y_score, average='macro')
                
        
        eval_metrics['mAP'] = val_mAP # a bit of fudging for display sake
        eval_metrics['val_mAP'] = val_mAP
        
        #calculate threshold 
        if ('threshold' in decision_function.__name__):
            per_class = (decision_function.__name__ == 'decision_by_class_threshold')
            th = th_pcut(train_loader.dataset, val_loader.dataset, val_scores, per_class=per_class)
            if per_class :
                print('Thresholds %r' % th)
                #eval_metrics['threshold'] = th.mean()
            else:
                eval_metrics['threshold'] = th
            decision_param = th
            
        metrics_history = metrics_history + [eval_metrics]
        
        # determine best model
        is_best = False
        if (val_acc > best_val_acc and epoch > 0 and val_mAP == 0.0): #multi-class
            is_best = True
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            
        if (val_mAP > best_val_mAP and epoch > 0): #multi-label
            is_best = True
            best_val_mAP = val_mAP
            best_model_weights = copy.deepcopy(model.state_dict())
            
            
        save_checkpoint({   'epoch': epoch + 1,
                            'arch': model.__class__.__name__,
                            'state_dict': model.state_dict(),
                            'best_acc': best_val_acc,
                            'classes' : train_loader.dataset.classes,
                            'threshold': th,
                            'elapsed_time': datetime.datetime.now() - t0,
                            'metrics_history': metrics_history},
                             (save_best_model and is_best),
                             dir=checkpoint_dir,
                             filename= '%s.checkpoint.pth.tar' % checkpoint_prefix)
            
        if (end_of_epoch_callback is not None):
            end_of_epoch_callback(eval_metrics)
        print() #end of epoch
    
    # load best model weights
    if (save_best_model):
        model.load_state_dict(best_model_weights)
   
    t1 = datetime.datetime.now()
    elapsed_time = t1-t0
    
    return model, th, elapsed_time
    
#%%
def train_epoch(model, 
                data_loader, 
                criterion, optimizer, scheduler, 
                activation_function, 
                decision_function, decision_param,
                print_frequency = 1000):   
        
    one_hot_encoding = isinstance(data_loader.dataset[0][1], list)
    class_count = len(data_loader.dataset.classes)
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")     

    global t0 
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    model.train()  # Set model to training mode
    #device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu") # Use GPU if available
    
    #scheduler.step() 

    for batch_idx, (img, target, *add_data) in enumerate(data_loader):

        if(one_hot_encoding):
            target = torch.stack(target).t().float()

        # use GPU if possible
        img = img.to(device)
        target =  target.to(device) 
        
        #enable autograd
        img_var = torch.autograd.Variable(img)
        target_var = torch.autograd.Variable(target)
				
        if add_data:
            add_data_var = torch.autograd.Variable(torch.stack(add_data[0]) 
                                                   .t() 
                                                   .float() 
                                                   .to(device))
        
        multi_crop = img.dim() > 4
        ncrops = 1
        if multi_crop:
            bs, ncrops, c, h, w = img_var.size()
            img_var = img_var.view(-1, c, h, w)
            #TODO: finish multicrop

        #forward
        if(model.aux_logits):
            out, aux_out = model(img_var) if not add_data else model(img_var , add_data_var) 
            loss1 = criterion(out, target_var)
            loss2 = criterion(aux_out, target_var)
            loss = loss1 + 0.4*loss2
        else:
            out = model(img_var) if not add_data else model(img_var , add_data_var) 
            loss = criterion(out, target_var)

        optimizer.zero_grad()         # zero the parameter gradients
        loss.backward()
        optimizer.step()
                    
                    #multi_crop = img.dim() > 4
                    #if (img.dim() < 5):
                    #    output, aux_output = net(img)
                    #else: # multi crops
                    #    bs, ncrops, c, h, w = img.size()
                    #    output, aux_output = net(img.view(-1, c, h, w)).view(bs, ncrops, -1).mean(1)  # fuse batch size and ncrops & avg crops
                        #output = output_crops.view(bs, ncrops, -1).mean(1) # avg over crops

                            
        out = activation_function(out) 
        
        pred = decision_function(out , decision_param, target)
          
        if(one_hot_encoding):
            outcome = (torch.sum(pred==target.byte(), dim=1) == class_count).view(target.shape[0] , 1).float()
        else:
            outcome = (pred == target.data).view(-1,1).float()
           
        # statistics
        running_loss += loss.item() * img.size(0)
        running_corrects += torch.sum(outcome)
        running_total += target.size(0)

        t1 = datetime.datetime.now()
        if ((running_total % print_frequency) == 0):
            print('\rImages processed: %d      Time elapsed: %s' % (running_total-target.size(0) , str(t1-t0)), end='')

    print('\rImages processed: %d      Time elapsed: %s' % (running_total-target.size(0) , str(t1-t0)))

    loss = running_loss / running_total
    accuracy = running_corrects.double().item() / running_total 
    elapsed_time = t1-t0
    
    scheduler.step() 

    return loss, accuracy, elapsed_time

#%%
def validate(model, 
             data_loader,  
             criterion, 
             activation_function, 
             decision_function , decision_param,
             print_frequency=1000 ):
    
    one_hot_encoding = isinstance(data_loader.dataset[0][1], list)
    class_count = len(data_loader.dataset.classes)
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    
    model.eval()   # Set model to evaluate mode   

    global t0 

    val_scores = torch.empty(0, dtype=torch.float).to(device)

    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    for batch_idx, (img, target, *add_data) in enumerate(data_loader):

        if(one_hot_encoding):
            target = torch.stack(target).t().float()

        # use GPU if possible
        img = img.to(device)
        target =  target.to(device) 
		
        if add_data:
            add_data_var = torch.stack(add_data[0]) \
                            .t() \
                            .float() \
                            .to(device) \
                                                   
        #out = model(img)
        with torch.set_grad_enabled(False):
            if (img.dim() < 5):
                out = model(img) if not add_data else model(img , add_data_var) 
            else: # multi crops
                bs, ncrops, c, h, w = img.size()
                out_raw = model(img.view(-1, c, h, w)) if not add_data else model(img.view(-1, c, h, w) , add_data_var) 
                out = out_raw.view(bs, ncrops, -1).max(1) # fuse batch size and ncrops & max crops
        
        loss = criterion(out, target)
                             
        out = activation_function(out) 
        
        pred = decision_function(out , decision_param)
  
        if(one_hot_encoding):
            outcome = (torch.sum(pred==target.byte(), dim=1) == class_count).view(target.shape[0] , 1).float()
        else:
            outcome = (pred == target.data).view(-1,1).float()
            
        val_scores = torch.cat((val_scores, out ))
           
        # statistics
        running_loss += loss.item() * img.size(0)
        running_corrects += torch.sum(outcome)
        running_total += target.size(0)

        #loss = running_loss / running_total
        #acc = running_corrects.double().item() / running_total 

#        mAP = 0.0
#        if (val_scores.nelement() > 0 ):
#            mAP = metrics.average_precision_score(np.array(data_loader.dataset.targets[:running_total]) , val_scores.cpu().detach().numpy())

        t1 = datetime.datetime.now()
        if ((running_total % print_frequency) == 0):
            print('\rImages processed: %d      Time elapsed: %s' % (running_total , str(t1-t0)), end='')
    
    print('\rImages processed: %d      Time elapsed: %s' % (running_total , str(t1-t0)))        

    loss = running_loss / running_total
    accuracy = running_corrects.double().item() / running_total 
    elapsed_time = t1-t0

    return loss, accuracy, elapsed_time, val_scores

#%%
class DecisionByThreshold(object):
    def __init__(self, threshold_value):
        self.th = threshold_value
        
    def __call__(self, output):
        return (output >= self.th) 

#TODO:refactor    
class DecisionByTopK(object):
    def __init__(self, k):
        self.k = k
    def __call__(self, output):

        ret = torch.stack( [self.__foo(x_i, self.k)  for i, x_i in enumerate(torch.unbind(output, dim=0), 0)]
                     ).to(output.device)
    
        return ret
    
    def __foo(_output_i, topk):
        _, idx = torch.topk(_output_i, topk)
        
        ret = torch.zeros(_output_i.shape[0], dtype=torch.uint8)
        ret[idx] = 1
        return ret


    
def decision_by_threshold(output, *args):
    threshold = args[0]
    return (output >= threshold) 

def decision_by_class_threshold(output, *args):
    threshold = args[0]
    #TODO: add dimensions check
    if isinstance(threshold, (list, np.ndarray)):
        threshold = output.new_tensor(threshold)
        return output >= threshold
    
    elif isinstance(threshold, torch.Tensor):
        threshold = threshold.to(output.device)    
        return output >= threshold
    
    else:
        threshold = output.new_full( (output.shape[1],) , threshold)
        return (output >= threshold) 


def decision_by_max_margin_margin(output, *args ):
    margin = args[0]
    return (output >= torch.topk(output, 1, 1)[0].add_(-(margin)))

def decision_by_max(output, *args):
    return torch.max(output, 1)[1]

def decision_by_topk(output, *args ):
    topk = int(args[0])
    
    def __foo(_output_i, topk):
        _, idx = torch.topk(_output_i, topk)
        
        ret = torch.zeros(_output_i.shape[0], dtype=torch.uint8)
        ret[idx] = 1
        return ret
    
    
    ret = torch.stack( [__foo(x_i, topk)  for i, x_i in enumerate(torch.unbind(output, dim=0), 0)]
                     ).to(output.device)
    
    return ret


# need to redefine the softmax to make the signature the same as sigmoid
def __softmax(input):
    return torch.softmax(input, 1)
#%%
def predict(model, image_loader, activation_function, decision_function, decision_param ,  print_frequency=1000):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  

    running_total = 0
    running_correct = 0
    
    if (activation_function.__name__ == 'softmax'):
        activation = __softmax
    else:
        activation = activation_function
   
    one_hot_encoding = isinstance(image_loader.dataset[0][1], list)
    class_count = len(image_loader.dataset.classes)
    
    # Scores tensor
    pred_scores = torch.empty(0, dtype=torch.float).to(device)
    mAP = 0.0
    
    for batch_idx, (img, target, *add_data) in enumerate(image_loader):
        
        if(one_hot_encoding):
            target = torch.stack(target).t().float()
       
        img, target = img.to(device), target.to(device)
		
        if add_data:
            add_data_var = torch.stack(add_data[0]) \
                                   .t() \
                                   .float() \
                                   .to(device)
                                   
        #run model
        with torch.set_grad_enabled(False):
            if (img.dim() < 5):
                out = model(img) if not add_data else model(img, add_data_var)
            else: # multi crops
                bs, ncrops, c, h, w = img.size()
                out_raw = model(img.view(-1, c, h, w)) if not add_data else model(img.view(-1, c, h, w), add_data_var)
                out = out_raw.view(bs, ncrops, -1).max(1) # fuse batch size and ncrops & avg crops
               
        out = activation(out) 

        pred_label = decision_function(out , decision_param)
  
        if(one_hot_encoding):
            outcome = (torch.sum(pred_label==target.byte(), dim=1) == class_count).view(target.shape[0] , 1).float()
        else:
            outcome = (pred_label == target.data).view(-1,1).float()
        
        pred_scores = torch.cat((pred_scores, out) )
        
        running_total += target.size()[0]
        running_correct += torch.sum(outcome)

        # display progress
        if ((running_total % print_frequency) == 0):
            if (one_hot_encoding):
                mAP = metrics.average_precision_score(np.array(image_loader.dataset.targets[:running_total]) , pred_scores.cpu().numpy())

            print( 'images total: {:.0f}, correct: {:.0f}, acc: {:.1f}% , mAP: {:.3f}'.format(
                    running_total, running_correct, running_correct * 100.0 / running_total, mAP))

    if (one_hot_encoding):
        mAP = metrics.average_precision_score(np.array(image_loader.dataset.targets[:running_total]) , pred_scores.cpu().numpy())
    print( 'images total: {:.0f}, correct: {:.0f}, acc: {:.1f}% , mAP: {:.3f}'.format(
            running_total, running_correct, running_correct * 100.0 / running_total, mAP))
    
    print('Finished prediction')
    
    return pred_scores

#%%
def score(model, image_loader, activation_function, print_frequency=1000, save_to_file=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  

    running_total = 0
    #running_correct = 0
    
    if (activation_function.__name__ == 'softmax'):
        activation = __softmax
    else:
        activation = activation_function
        
    dataset_size = len(image_loader.dataset)
   
    #one_hot_encoding = isinstance(image_loader.dataset[0][1], list)
    #class_count = len(image_loader.dataset.classes)
    
    # Scores tensor
    pred_scores = torch.empty(0, dtype=torch.float).to(device)
    #mAP = 0.0
    
    for batch_idx, (img, target, *add_data) in enumerate(image_loader):
        
        img = img.to(device)
        
        add_data_var = torch.stack(add_data[0]) \
                                   .t() \
                                   .float() \
                                   .to(device)
            
        #run model
        with torch.set_grad_enabled(False):
            if (img.dim() < 5):
                out = model(img) if not add_data else model(img, add_data_var)
                
            else: # multi crops
                bs, ncrops, c, h, w = img.size()
                out_raw = model(img.view(-1, c, h, w)) if not add_data else model(img.view(-1, c, h, w), add_data_var)
                out = out_raw.view(bs, ncrops, -1).max(1) # fuse batch size and ncrops & avg crops
               
        out = activation(out) 
        
        pred_scores = torch.cat((pred_scores, out) )
        
        running_total += img.size()[0] 

        # display progress
        if ((running_total % print_frequency) == 0):
            print( 'images total: {:d} of {:d}'.format(running_total, dataset_size ))


    print( 'images total: {:d} of {:d}'.format(running_total, dataset_size ))
    print('Finished scoring')
    
    if (save_to_file is not None):
        torch.save(pred_scores, save_to_file)
    
    return pred_scores

#%%
def th_pcut(train_set, val_set, val_scores, thresholds=np.arange(0.5, 1.0, 0.05, dtype = float), per_class=False):
    
    """ 
     PCut formula is taken from here 
     https://users.ics.aalto.fi/jesse/talks/Multilabel-Part01.pdf
    """
    #TODO:validate parameters
    
    class_count = len(val_set.classes)
    n = val_scores.shape[0]
    
    if per_class:
        pcut = np.zeros(class_count)

        for c in range(class_count):
            train_prop = np.array(train_set.targets)[:,c].sum()/len(train_set)
            val_prop = np.zeros(len(thresholds))

            for i, th in enumerate(thresholds):
                y_pr = (val_scores[:n,c] >=  th)
                val_prop[i] = y_pr.sum().float() / len(val_set)
                pcut[c] = thresholds[np.abs(val_prop - train_prop).argmin()]
            
        return pcut        
    else:
        
        train_prop = np.array(train_set.targets).sum()/len(train_set)
        val_prop = np.zeros(len(thresholds))
        for i, th in enumerate(thresholds):
            y_pr = (val_scores[:n] >= th)
            val_prop[i] = y_pr.sum().float() / len(val_set)
        pcut = thresholds[ np.abs(val_prop - train_prop).argmin()]

        return pcut

def th_internal_validation(val_set, val_scores, thresholds=np.arange(0.5, 1.0, 0.05, dtype = float)):
    #TODO:validate parameters
    class_count = len(val_set.classes)

    sums = np.zeros(len(thresholds))
    n = val_scores.shape[0]

    for i, th in enumerate(thresholds):
        y_true = np.array(val_set.targets[:n], dtype = np.dtype('B'))
        y_pred = (val_scores[:n] >= th).numpy()
        sums[i] = ((y_true==y_pred).sum(axis=1) == class_count).sum()

    th = thresholds[sums.argmax()]
    return th


#%%
def save_checkpoint(state, is_best, dir='output', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir, filename))
    if is_best:
        copy_filename = filename.replace('checkpoint', 'best')
        shutil.copyfile(os.path.join(dir, filename), os.path.join(dir, copy_filename))
#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res