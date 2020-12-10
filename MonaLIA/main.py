# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:05:52 2019

@author: abobashe
"""

import config

import torch
import torchvision
#import torchvision.models as models
import training as t
import scoring as sc
import cross_validation as cv


def print_pretty_header():
      print('''
                                _ _           
  _ __ ___   ___  _ __   __ _  | (_) __ _
 | '_ ` _ \ / _ \| '_ \ / _` | | | |/ _` |
 | | | | | | (_) | | | | (_| | | | | (_| |
 |_| |_| |_|\___/|_| |_|\__,_| |_|_|\__,_|
          
          ''')  


def print_versions():
    print('PyTorch ver.' ,  torch.__version__ )
    print('torchvision ver.' ,  torchvision.__version__ )
    print('cuda ver.', torch.version.cuda, ', available = ', torch.cuda.is_available())
    #print('sckit-learn ver.', sklearn.__version__)
    print()
    
def print_args():
    print(str(config.args).replace(',', '\n').replace('Namespace(', 'Namespace(\n '))
    print()


def main():
    
    print_pretty_header()
    print_versions()
    print_args()

    if (config.args.task == 'train'):
        t.model_training()
    
    elif (config.args.task == 'test'):
        print('to be implemented')
        t.preload_combined_dataset()
        
    elif (config.args.task == 'cross-validate'):
        cv.cross_validation()
        
    elif (config.args.task == 'score'):
       sc.scoring()

    elif (config.args.task == 'train-binary'):
       t.binary_model_training()    
       
    elif (config.args.task == 'score-binary'):
       sc.binary_model_scoring()  
   
    
    
#%%
if __name__ == "__main__":
    main()
