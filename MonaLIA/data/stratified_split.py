# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:25:25 2019

@author: abobashe
"""
import numpy as np
import pandas as pd


###############################################################################
# Stratified split
###############################################################################
def train_validate_test_split(df, train_percent=.8, val_percent=.1, test_percent= 0.1, seed=None, max_size=0):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    
    if (max_size > 0 and m > max_size):
        m = max_size
    
    #train_end = int(train_percent * m)
    test_end = int(test_percent * m)
    #val_end = int(val_percent * m) + train_end
    val_end = int(val_percent * m) + test_end
    
    
    
    #train_idx = perm[:train_end]
    test_idx = perm[:test_end]
    #val_idx   = perm[train_end:val_end]
    val_idx   = perm[test_end:val_end]
    #test_idx  = perm[val_end: m]
    train_idx  = perm[val_end: m]
    
   
    return train_idx, val_idx, test_idx

def train_cross_validate_test_split(df, trainval_percent=0.9, n_folds=10, test_percent= 0.1, seed=None, max_size=0):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    
    if (max_size > 0 and m > max_size):
        m = max_size

    test_end = int(test_percent * m)
    test_end = test_end + ((m - test_end) % n_folds)
    

    test_idx = perm[:test_end]
    trainval_idx = perm[test_end:m].reshape(10, -1)
  
    return trainval_idx, test_idx

def merge_image_sets_with_additional_classes(image_set_df_sm, image_set_df_lg):
    """
       Merge two dataset 
       - retaining the records of the classes that are in the smaller dataset (passed first)
       - updating the label of the smaller dataset with labels of the larger dataset (passed second)
       - concatenating the smaller dataset and records with new labels from the larger dataset
       
    """ 
    group_by = 'label'
    update_col = ['label', 'terms', 'term_count' , 'top_term_count']
    lsuffix='_sm'
    rsuffix='_lg'
    
    new_col = ['ref'] + [x + lsuffix for  x in update_col] + [y + rsuffix for  y in update_col] 
   
    #unnesesary intermediate step for clarity
    temp = image_set_df_sm.join(image_set_df_lg.set_index('ref'),
                               on='ref', how='left',lsuffix=lsuffix, rsuffix=rsuffix).loc[: , new_col]
   
    #add columns to the smaller set that contain information from both smaller and larger sets
    image_set_df_sm = image_set_df_sm.join(temp.set_index('ref'), on='ref')
   
    #replace lables in the smaller set with labels from the larger set
    for col in update_col:
        image_set_df_sm[col] = image_set_df_sm[col + rsuffix]

    image_set_df_sm.drop([y + rsuffix for  y in update_col], axis=1, inplace = True)
   
    # create label list that is new in 40 classes dataset
    temp = pd.DataFrame(image_set_df_sm.groupby(by=group_by).size()).join(pd.DataFrame(image_set_df_lg.groupby(by=group_by).size()),  how='right', lsuffix=lsuffix, rsuffix=rsuffix)

    new_label_list = temp[temp['0'+ lsuffix].isna()].index.tolist()

    # stack the smaller datset and the new labels from the larger dataset
    new_image_set = pd.concat([image_set_df_sm,  image_set_df_lg[(image_set_df_lg.label.isin(new_label_list))]], sort=False)

    return new_image_set