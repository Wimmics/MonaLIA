# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:34:37 2019

With help from : https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
                 https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
                 

@author: abobashe
"""

import os

import numpy as np
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as transforms


class JocondeDataset(dset.ImageFolder):
    """Joconde dataset."""
   
    def __init__(self, 
                 csv_file, 
                 images_root_dir, 
                 dataset_name = '',
                 label_column='label', 
                 exclude_labels=[] ,
                 filename_column='imagePath',
                 filter_dict = {},  
                 add_columns = ['ref'],
                 transform=None, 
                 target_transform=None,
                 multiple_labels=False):
        """
        Args:
            csv_file (string): Path to the csv file with image file information.
            root_dir (string): Directory with all the images.
            label_column (string): Column in the csv file that defines the image label
            exclude_lables (list of strings): Exclude items with labels from this list from the dataset 
            filter_dict (dictionary): Define the filters as  a dictionary key =<column name> value=[list of values]
            add_columns (list of strings): Additional items included with the sample file
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        """
        # to curcumvent the exception in the base class constructor
        # output an empty file
        fake = self.__output_fake_dir(os.path.dirname(csv_file))
        
        super(JocondeDataset, self).__init__(images_root_dir,#os.path.dirname(csv_file),
                                             transform=transform,
                                             target_transform=target_transform)
                
        self.__remove_fake_dir(fake)
        
        classes, class_to_idx, samples = self._make_dataset(images_root_dir,
                                                          csv_file,
                                                          label_column=label_column, 
                                                          exclude_labels=exclude_labels ,
                                                          multi_label=multiple_labels,
                                                          filename_column=  filename_column, 
                                                          filter_dict=filter_dict,  
                                                          add_columns=add_columns)
        self.name = dataset_name
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
        
        self.descr_file = csv_file

        if isinstance(self.targets[0], int): #single label
            labels_per_class  = [sum([x == i  for x in self.targets])  for i in range(len(self.classes))]
        else:                                #multi label 
            labels_per_class = [sum([x[i] for x in self.targets]) for i in range(len(self.classes))]

        self.labels_count = dict(zip(self.classes, labels_per_class))

    def _make_dataset(self, 
                     root,
                     dataset_file, 
                     label_column='label', exclude_labels=[] ,
                     multi_label = False,
                     filename_column='imagePath',
                     filter_dict = {}, 
                     add_columns = []
                 ):
        images = []
        
        root = os.path.expanduser(root)

        df = pd.read_csv(dataset_file, na_filter=False)

        # get classes 
        #classes = df[label_column].dropna().unique() 
        classes = [x  for x in df[label_column].unique() if len(x) > 0]
        classes = [ elem for elem in classes if elem not in exclude_labels]
        
        if(multi_label):
            #exclude compositions of classes based on assumption that 
            #labels will be separated by '+'
            #classes = [ elem for elem in classes if elem.find('+') < 0 ]
            classes = sorted(set('+'.join(classes).split('+')))
        else:
            classes.sort()
        
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        #filter the datatset 
        if(len(exclude_labels) > 0):
            df = df.loc[df[label_column].isin(classes)]
           
        for column, values in filter_dict.items():    
            df = df[df[column].isin(values)] 

        #create the imagepath and target(s) list
        for index, row in df.iterrows():
            path = os.path.normpath(os.path.join(root, row[filename_column].strip('/\\')))
            
            if(multi_label):
                target = [1 if key in row[label_column].split('+') else 0 for key in class_to_idx.keys()]               
            else:
                target = class_to_idx[row[label_column]]
 
            item = (path , target)

            for i in range(len(add_columns)):
                item = item + (row[add_columns[i]],)

            images.append(item)

        return classes, class_to_idx, images   

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index][0]
        target = self.samples[index][1]
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __output_fake_dir(self, dir):
        
        fake_dir =  os.path.join(dir, 'foo') 
        if not os.path.exists(fake_dir ):
            os.makedirs(fake_dir)
        
        fake_file = os.path.join(fake_dir, 'bar.png')
        if not os.path.exists(fake_file ):
            open(fake_file, 'a').close()
            
        return fake_dir     
    
    def __remove_fake_dir(self, fake_dir):
        import shutil
        shutil.rmtree(fake_dir)
        
        
    def extra_repr(self):
        return '\n'.join(['Description file: {}'.format(self.descr_file),
                          'Number of classes: {}'.format(len(self.classes)),
                          'Number of uniqie labels: {}'.format(np.unique(np.array(self.targets), axis=0).shape[0],
                          'Number of class labels: {}'.format(sum(self.labels_count.values())))])
         
    def get_norm_values(self):
        for trans in self.transform.transforms:
            if(isinstance(trans, transforms.transforms.Normalize)) :
                return trans   
    
        return transforms.transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
#%%
#TODO: Test
class JocondeDataset_ext(JocondeDataset):
    
    def __init__(self, 
                 csv_file, 
                 images_root_dir, 
                 dataset_name = '',
                 label_column='label', 
                 exclude_labels=[] ,
                 filename_column='imagePath',
                 filter_dict = {},  
                 add_columns = ['ref'],
                 transform=None, 
                 target_transform=None,
                 multiple_labels=False):
        """
        Args:
            the same as in JocondeDataset
        """

        super(JocondeDataset_ext, self ).__init__(csv_file, 
                                                 images_root_dir, 
                                                 dataset_name,
                                                 label_column, 
                                                 exclude_labels,
                                                 filename_column,
                                                 filter_dict,  
                                                 add_columns,
                                                 transform, 
                                                 target_transform,
                                                 multiple_labels)
    
    def __getitem__(self, index):
        
        sample, target = super(JocondeDataset_ext, self ).__getitem__(index)
        
        if len(self.samples[index]) > 1:
            extra_data = self.samples[index][-1]
        else:
            extra_data = None
    
        return sample, target, extra_data
    
#%%
# Alternative approach to delivering extra data 
        
class JocondeDataset_wrap(object):
    def __init__(self, Joconde_dataset):
        
        if isinstance(Joconde_dataset, JocondeDataset):
            raise TypeError()
            
        self.dataset = Joconde_dataset
        
    def __getitem__(self, index):
        sample, target = self.dataset.__getitem__(index)
        
        if len(self.samples[index]) > 1:
            extra_data = self.samples[index][-1]
        else:
            extra_data = None
    
        return sample, target, extra_data
    