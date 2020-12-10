# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:18:01 2019

@author: abobashe
"""

# Define the arguments
##############################################################################
import argparse

parser = argparse.ArgumentParser(description='Joconde Training')

tasks = ['train', 'test', 'cross-validate', 'show-config', 'score', 'train-binary', 'score-binary']
parser.add_argument('task',
                    choices= tasks,
                    default='show-config',
                    help='run task: %s' %  (' | '.join(tasks)))

# Data arguments
##############################################################################
database_names = ['Joconde', 'ImageNet']
parser.add_argument('-db', '--database',
                    metavar='DATABASE',
                    default = 'Joconde',
                    choices=database_names,
                    help='dataset name: %s (default: Joconde)' %  (' | '.join(database_names)))

import os
def _resolve_path(path_string):
    return os.path.realpath(os.path.normpath(os.path.expanduser(path_string)))


parser.add_argument('--image-root',
                    metavar='IMAGE_ROOT_DIR',
                    help='path to the image set)' ,
                    type=_resolve_path,
                    default='/Joconde/joconde')

parser.add_argument('--dataset',
                    metavar='DATASET',
                    default = 'Ten Classes',
                    help='dataset name: for ex. Animals_and_Humans (default: Ten Classes)')

parser.add_argument('--dataset-descr-file',
                    metavar='DATASET_DESCR_FILE_PATH',
                    default='/Datasets/Joconde/Ten Classes/dataset1.csv',
                    type=_resolve_path,
                    help='for Joconde database path to the dataset description csv file (default: Datasets/Joconde/Ten Classes/dataset1.csv)')

parser.add_argument('--dataset-label-column',
                    metavar='COL_NAME', 
                    default='label',
                    help='column in csv file that contains lables (default: label)' )

parser.add_argument('--dataset-exclude-label',
                    metavar='LABEL',
                    action='append',
                    dest='dataset_exclude_labels',
                    default=[],
                    help='images with these labels will be excluded from the dataset, this opton can be repeated')

parser.add_argument('--disable-one-hot-encoding',
                    action='store_false',
                    dest='multi_label',
                    help='disable one-hot encoding (default: enabled))')

parser.add_argument('--multi-crop',
                    action='store_true',
                    help='enable 5 crop image tansformation (default: disabled)')

parser.add_argument('--workers',
                    metavar='N',
                    default=4,
                    type=int,
                    help='number of data loading workers (default: 4)')

parser.add_argument('--batch-size',
                    metavar='N',
                    default=4,
                    type=int,
                    help='mini-batch size (default: 4)')

# Model arguments
##############################################################################

arch_names = ['vgg16_bn', 'inception_v3', 'resnet_50', 'resnet_101', 'wide_resnet_101', 'ensemble', 'ensemble_v2' , 'ensemble_v3']
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='inception_v3',
                    choices=arch_names,
                    #action=_store_image_size,
                    help='dataset name: %s (default: inception_v3)' %  (' | '.join(arch_names)))


#training_modes = ['features', 'finetuning']
#parser.add_argument('-m', '--training_mode',
#                    metavar='MODE',
#                    default='finetuning',
#                    choices=training_modes,
#                    help='training mode: %s (default: finetuning)' %  (' | '.join(training_modes)))

parser.add_argument('--finetuning',
                    action='store_true',
                    dest='model_finetuning',
                    help='enable all layers training (default: False)') 

# Training setup arguments
##############################################################################
parser.add_argument('--epochs',
                    metavar='N',
                    default=10,
                    type=int,
                    help='number of total epochs to run (default: 10)')

parser.add_argument('--output-dir',
                    metavar='DIR',
                    default='output',
                    type=_resolve_path,
                    help='path to the output files')
import datetime
def _timestamp(suffix):
    if(not suffix):
        suffix =  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    return suffix

parser.add_argument('--param-file-suffix',
                    metavar='SUFFIX',
                    default='',
                    type=_timestamp,
                    help='output parameter file name is <model>_<database>_<dataset>_<suffux>.pth, (default: timestamp)')

parser.add_argument('--print-freq',
                   metavar='N',
                   default=1000,
                   type=int,
                   help='print frequency in number of images processed (default: 1000)')

# Training Hyperparameters arguments
##############################################################################
optimizers = ['SGD', 'Adam']

parser.add_argument('--optim',
                    metavar='OPTIMIZER',
                    default='SGD',
                    choices=optimizers,
                    #type=_optimizer,
                    help='optimizer: %s (default: SGD)' %  (' | '.join(optimizers)))

parser.add_argument('--lr', 
                    metavar='LR',
                    default=0.0001,
                    type=float,
                    help='initial learning rate (default: 0.0001)')

parser.add_argument('--momentum',
                    metavar='M',
                    default=0.9,
                    type=float,
                    help='momentum')

parser.add_argument('--weight-decay',
                    metavar='W',
                    default=1e-4,
                    type=float,
                    help='weight decay (default: 1e-4)')

parser.add_argument('--lr-scheduler',
                    nargs = 2,
                    metavar=('STEP', 'GAMMA'),
                    default=[4, 0.1],
                    type=float,
                    help='decays the learning rate by GAMMA every STEP epochs (default: 4, 0.1)')

#parser.add_argument('--lr_scheduler_gamma',
#                    metavar='STEP_GAMMA',
#                    default=0.1,
#                    type=float,
#                    help='decays the learning rate by GAMMA every STEP_SIZE epochs (default: 0.1)')
#
#parser.add_argument('--lr_scheduler_step',
#                    metavar='STEP_SIZE',
#                    default=4,
#                    type=int,
#                    help='decays the learning rate every STEP_SIZE epochs (default: 4)')

loss_functions = ['CrossEntropy', 'BCE']

parser.add_argument('--loss',
                    metavar='FUNC',
                    default='BCE',
                    choices=loss_functions,
                    #type= _loss_func,
                    help='loss function: %s (default: BCE - BinaryCrossEntropy)' %  (' | '.join(loss_functions)))

parser.add_argument('--use-weights',
                    action='store_true',
                    help='calculate weights for pos_weights argument of BCEWithLogitsLoss, weights argument for CrossEnthopy Loss')


activation_functions = ['softmax', 'sigmoid']

parser.add_argument('--activation',
                    metavar='FUNC',
                    default='sigmoid',
                    choices=activation_functions,
                    #type = _activation_func,
                    #dest='activation_function',
                    help='last layer activation function: %s (default: Sigmoid)' %  (' | '.join(activation_functions)))


decision_functions = [ 'threshold', 'threshold-per-class', 'top-k', 'max']

#def _descision_func(string):
#    import model.train as train
#
#    decision = train.decision_by_topk
#    if (string == 'threshold'):
 #       decision = train.decision_by_threshold
#    return decision
 


parser.add_argument('--decision', 
                    metavar='FUNC',
                    default='threshold',
                    choices=decision_functions,
                    #type= _descision_func,
                    help='classification decision: %s (default: threshold)' %  (' | '.join(decision_functions)))

parser.add_argument('--decision-param',
                    metavar='N',
                    default=0.5,
                    type=float,
                    help='parameter for the classification decision making (default: threshold_per_class, thershold = 0.5 or top-k = 1)')


# Metadata arguments
##############################################################################
parser.add_argument('--repr-thesaurus-file',
                    metavar='REPR_THESAURUS_FILE_PATH',
                    default='/RDF/reprskos.rdf',
                    type=_resolve_path,
                    help='for Joconde meatadata path to the REPS thesaurus (default: /RDF/reprskos.rdf')

# to be implemented:

#
#parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
#
#parser.add_argument('--start-epoch',
#                    metavar='N',
#                    default=0,
#                    type=int,
#                    help='manual epoch number (useful on restarts)')



args = parser.parse_args()


# derived config variables
##############################################################################
#TODO refactor
input_image_size = 224
if ((args.arch == 'inception_v3') or args.arch.startswith('ensemble')):
    input_image_size = 299

setattr(args , 'input_image_size', input_image_size)



model_param_file = os.path.join(args.output_dir,
                                '%s_%s_%s.%s.pth' % (args.arch, args.database, args.dataset, args.param_file_suffix))

setattr(args , 'model_param_file', model_param_file)
                                
train_metrics_file = os.path.join(args.output_dir,
                                 '%s_%s_%s.%s.csv' % (args.arch, args.database, args.dataset, args.param_file_suffix))


setattr(args , 'train_metrics_file', train_metrics_file)















from pathlib import Path

if (os.name == 'nt'):
    home_dir = 'C:/'
else:
    home_dir = str(Path.home())
    
#database = args.database
#dataset = args.dataset



# input files
#if args.database == 'Joconde':
#    #TODO:custom action function
#    images_root = os.path.join(home_dir, args.image_root)
#    image_description_file = os.path.join(home_dir, args.dataset_descr_file)
#    
#    #TODO: add parameter
#    exclude_labels = []# ['espèce animale+être humain' , 'none+none'] 
#
#elif args.database == 'ImageNet':
#    images_root = os.path.join(home_dir, 'ImageNet')

# output files
#import datetime

#TODO: custom action 


#model_param_folder = args.output_dir #'./output/'
                                

#model_param_file = '%s_%s_%s.%s.pth' % (model_name, database, dataset, parameters_ver)
#metrics_file = '%s_%s_%s.%s.csv' % (model_name, database, dataset, parameters_ver)

# model 





