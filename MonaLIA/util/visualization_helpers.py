# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:19:45 2019

@author: abobashe
"""
import numpy as np
import pandas as pd
from itertools import compress

import torch
import torchvision

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as colormap
import seaborn as sns

from  textwrap import wrap

from sklearn import metrics

import warnings

import base64
from io import BytesIO
from IPython.display import display, HTML
from PIL import Image

import os
###############################################################################
# Tensor and Image Visualization
###############################################################################
def image_tensor_to_np(image_tensor,
                       norm_mean = [0.0, 0.0, 0.0],
                       norm_std = [1.0, 1.0, 1.0]):
    """
    Helper function to convert image tensor to numpy array for use in image display 
    """
    image_np = image_tensor.numpy().transpose(1,2,0)
    image_np = image_np * norm_std + norm_mean
    image_np = np.clip(image_np, 0, 1)
    return image_np

def get_dataset_normalization_transformation(dataset):
    """
    Helper function to read the normalization values
    """
    for trans in dataset.transform.transforms:
        if(isinstance(trans, torchvision.transforms.transforms.Normalize)) :
            return trans   
    
    return torchvision.transforms.transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                                               std = [ 0.5, 0.5, 0.5 ])

def show_random_images(data_set,
                       num=4,
                       is_small=True,
                       crop_idx=None):
    """
    Helper function for showing random images from a dataset
    """
    if not is_small:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

    for i in range(num):
        idx = np.random.randint(0, len(data_set)-1)
        image, target = data_set[idx] # get a random image

        print(data_set.samples[idx])

        plt.subplot(1, num, i+1)
        if (type(target) is int):
            plt.title('{}\n{}'.format(target, data_set.classes[target]))
        else:    
            plt.title('{}\n{}'.format(target, ', '.join(list(compress(data_set.classes,target)))))
        
        #support multi-crop
        if (image.dim() > 3):
            if(crop_idx is None):
                crop = np.random.randint(0, 5)
            else:
                crop = crop_idx
            image = image[crop]
                
        #read normalization values from the dataset
        trans = get_dataset_normalization_transformation(data_set)
        dataset_mean = trans.mean
        dataset_std  = trans.std
        
        plt.imshow(image_tensor_to_np(image, dataset_mean, dataset_std))
    plt.show()
    
#%%
def visualize_classification_matplotlib(data_set, pred_labels, score_tensor, index_pick, title='', crop_idx=None, show_max_pred_labels=0):
    """
    Helper function to visualize the test results in 3 columns
        1. Full size image
        2. Transformed image
        3. Classifiaction details
    """
    rows = index_pick.shape[0]
    
    plt.tight_layout()
    fig, axes = plt.subplots(nrows=len(index_pick), ncols=3,  gridspec_kw = {'width_ratios':[2, 1, 1]})
    fig.set_size_inches(18.5, 7 * rows)
    fontsize=14
    fontfamily='monospace'

    #read normalization values from the dataset
    trans = get_dataset_normalization_transformation(data_set)
    dataset_mean = trans.mean
    dataset_std  = trans.std    
    
    for i in range(rows):
        testset_idx = index_pick[i]

        image, target = data_set[testset_idx]
        pred = pred_labels[testset_idx]
        
        #support multi-crop
        if (image.dim() > 3):
            if(crop_idx is None):
                crop = np.random.randint(0, 5)
            else:
                crop = crop_idx
            image = image[crop]

        target_annotation = target_to_text(data_set.classes, target, separator=', ', wrap_after=40)

        prediction_annotation = prediction_to_text(data_set.classes, score_tensor[testset_idx], separator='\n')
        
        if (show_max_pred_labels > 0):
            pr_annot = prediction_annotation.split(sep='\n')
            show_max_pred_labels = min((show_max_pred_labels, len(pr_annot)))
            prediction_annotation = '\n'.join(pr_annot[:show_max_pred_labels])
            
        
        #add representation field for reference
        repr_field = ''
        if(len(data_set.samples[testset_idx]) > 3): 
            repr_field = data_set.samples[testset_idx][3] 
            repr_field = '\n'.join(wrap(repr_field, 40 ))
        
        notice_ref = ''
        if(len(data_set.samples[testset_idx]) > 2): 
            notice_ref = data_set.samples[testset_idx][2] 

        axes[i, 0].imshow(mpimg.imread(data_set.samples[testset_idx][0]))
        axes[i, 0].set_title(data_set.samples[testset_idx][0] , loc='left', fontsize=fontsize, family=fontfamily)

        axes[i, 1].imshow( image_tensor_to_np(image , dataset_mean, dataset_std)  )

        
        axes[i, 2].set_title(notice_ref, loc='left', fontsize=fontsize, family=fontfamily)
        axes[i, 2].text(0.05, 0.95, '%s\n' % np.array(target) +
                                    'labels: %s\n\n' % target_annotation +
                                    '%s\n' % pred + 
                                    '%s' % prediction_annotation +
                                    '\n\n' + 
                                    repr_field,
                                    fontsize=fontsize, family=fontfamily, verticalalignment='top',  wrap=True)
        axes[i,2].axis('off')
        axes[i,2].autoscale(False)

    fig.suptitle(title, fontsize=20)
    plt.show()    
#%%
def target_to_text(classes, target, separator=', ', wrap_after=0):
    
    if isinstance(target, list):
        target_annotation = separator.join(list(compress(classes,target)))
    elif isinstance(target, int):
        target_annotation = classes[target]
    else:
        target_annotation = 'cannot translate target'
            
        
    #target_annotation = ', '.join(list(compress(classes,target)))
    if(wrap_after > 0):
        target_annotation = '\n'.join(wrap(target_annotation, wrap_after ))
    
    return target_annotation

def prediction_to_text (classes, scores, sort=True, separator='\n', wrap_after=0):
    
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    pred = zip(classes, scores)
    if (sort):
        pred = sorted(pred, key=lambda tup: tup[1], reverse=True)

    prediction_annotation = separator.join(['%-20s: %.3f' % (cl, score.item()) for cl, score in pred])
    
    if(wrap_after > 0):
        prediction_annotation = '\n'.join(wrap(prediction_annotation, wrap_after ))
    
    return prediction_annotation
#%%
def show_confusion_matrix(y_true, y_pred, tick_labels):

    cnf_matrix = metrics.confusion_matrix(y_true= y_true,
                                          y_pred= y_pred)
    np.set_printoptions(precision=2)
    
    plt.tight_layout()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 6) #(10, 4)
    
    plt.subplot(1,2,1)
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu",fmt='d', square=True, 
                yticklabels= tick_labels, xticklabels=tick_labels, 
                vmin=0, vmax = cnf_matrix.sum(axis=1).max())
    plt.title("Confusion Matrix")
    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    
    
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.subplot(1,2,2)
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='.2f', square=True,
                yticklabels= tick_labels, xticklabels=tick_labels, 
                vmin=0, vmax=1.0 )
    plt.title("Normalized Confusion Matrix")
    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    
    plt.tight_layout()
    plt.show()
#%%
def multi_label_summary (y_true, y_pred, classes, normalize=False, return_expl=False):
    
    y_true_str = [target_to_text(classes, list(x), separator='+') for x in y_true]
    y_pred_str = [target_to_text(classes, list(x), separator='+') for x in y_pred]
    labels_str = sorted(set(y_true_str).union(y_pred_str).union(classes))
    
    cm = metrics.confusion_matrix(y_true= y_true_str,
                                  y_pred= y_pred_str,
                                  labels=labels_str)

    cm_df = pd.DataFrame(cm , index= labels_str, columns=labels_str)
    
    #classes = sorted([x for x in labels if ((len(x) > 0) and ('+' not in x))])
    #print(len(classes))
    #print(classes)
    #classes = sorted(list(set([y for x in labels for y in x.split('+')])))
    #if classes[0] == '':
    #    classes.pop(0)
        
    #print(len(classes))
    #print(classes)
    
    sum_df = pd.DataFrame(index=classes)

    none_lbl = ''
    
    expl = {}
    
    if (none_lbl not in labels_str):
        cm_df.insert(0,'', 0)
    
    for i in range(len(classes)):

        cl = classes[i]
        #print(cl)

        cl_oth_s = classes.copy()
        cl_oth_s.pop(i)
        #print(cl_oth_s)

        cl_m = [x for x in labels_str if (cl in x) and ('+' in x)]
        #print(cl_m)

        cl_oth_m = [x for x in labels_str if (cl not in x) and ('+' in x)]
        #print(cl_oth_m)
        
        ########################### total ##################################
        expl['cl_tot'] = 'total labels of the class'
        sum_df.loc[cl, 'cl_tot'] = cm_df.loc[cl_m + [cl], :].sum().sum()
               
        ########################### single ##################################
        
        expl['single_tot'] = 'labels of the class that appear w/o other labels, ex. (bird)'
        sum_df.loc[cl, 'single_tot']         = cm_df.loc[cl , :].sum() 
        
        expl['single_corr'] = 'single label and correct, ex. label=(bird), pred=(bird)'
        sum_df.loc[cl, 'single_corr']        = cm_df.loc[cl , cl]
        
        expl['single_missed'] = 'single label and not classified at all, ex. label=(bird), pred=()'
        sum_df.loc[cl, 'single_missed']      = cm_df.loc[cl, none_lbl]
        
        expl['single_err_s'] = 'single label classified as another single label, ex. label=(bird), pred=(flower)'
        sum_df.loc[cl, 'single_err_s']       = cm_df.loc[cl, cl_oth_s].sum()

        expl['single_part_err_m'] = 'single label classified along with other labels, ex. label=(bird), pred=(bird+tree)'
        sum_df.loc[cl, 'single_part_err_m']  = cm_df.loc[cl, cl_m].sum()
        
        expl['single_err_m'] = 'single label classified as multiple other labels, ex. label=(bird), pred=(tree+house)'
        sum_df.loc[cl, 'single_err_m']       = cm_df.loc[cl, cl_oth_m].sum()
        
        expl['single_top_err_val'] = 'number of the most confused labels (see "single_top_err_cl"), ex. label=(house), pred=(chirch)'
        sum_df.loc[cl, 'single_top_err_val'] = cm_df.loc[cl, cl_oth_s].max()
        
        expl['single_top_err_cl'] = 'the most confused with single label, ex. label=(bird), pred=(flower)'
        sum_df.loc[cl, 'single_top_err_cl']  = cm_df.loc[cl, cl_oth_s].idxmax() if cm_df.loc[cl, cl_oth_s].max() > 0 else ''
        
        assert (sum_df.loc[cl, 'single_corr']  + 
                sum_df.loc[cl, 'single_missed'] + 
                sum_df.loc[cl, 'single_err_s']  + 
                sum_df.loc[cl, 'single_part_err_m'] + 
                sum_df.loc[cl, 'single_err_m'] == sum_df.loc[cl, 'single_tot'] )

        ########################### multi ##################################
        expl['multi_tot'] = 'labels of the class that appear w/other labels, ex. (bird+tree)'
        sum_df.loc[cl, 'multi_tot']  = cm_df.loc[cl_m, :].sum().sum()
        
        expl['multi_corr'] = 'multiple labels and all correctly classified, ex. label=(bird+tree), pred=(bird+tree)'
        sum_df.loc[cl, 'multi_corr'] = np.diag(cm_df.loc[cl_m, cl_m]).sum()
        
        expl['multi_missed'] = 'multiple labels and not classified at all, ex. label=(bird+tree), pred=()'
        sum_df.loc[cl, 'multi_missed'] = cm_df.loc[cl_m, none_lbl].sum()
        
        expl['multi_err_s'] = 'multiple labels that are classified with just a single label from the list, ex. label=(bird+tree), pred=(bird)'
        sum_df.loc[cl, 'multi_err_s'] = cm_df.loc[cl_m, cl].sum()
        
        expl['multi_err_oth_s'] = 'multiple labels that are classified with just a single label not from the list, ex. label=(bird+tree), pred=(house)'
        sum_df.loc[cl, 'multi_err_oth_s'] = cm_df.loc[cl_m, cl_oth_s].sum().sum() 
      
        expl['multi_err_part_m'] = 'multiple labels that are classified as other labels including some of the labels in the list, ex. label=(bird+tree), pred=(tree+house)'
        sum_df.loc[cl, 'multi_err_part_m'] = cm_df.loc[cl_m, cl_m].sum().sum() - sum_df.loc[cl, 'multi_corr']

        expl['multi_err_part_m'] = 'multiple labels that are classified as other labels none of which is on the list, ex. label=(bird+tree), pred=(flower+house)'
        sum_df.loc[cl, 'multi_err_m'] = cm_df.loc[cl_m, cl_oth_m].sum().sum() 
        
        assert (sum_df.loc[cl, 'multi_corr'] +
                sum_df.loc[cl, 'multi_missed'] + 
                sum_df.loc[cl, 'multi_err_s'] +
                sum_df.loc[cl, 'multi_err_oth_s'] + 
                sum_df.loc[cl, 'multi_err_part_m'] +
                sum_df.loc[cl, 'multi_err_m'] == sum_df.loc[cl, 'multi_tot'])
        

        assert (sum_df.loc[cl, 'single_tot'] + sum_df.loc[cl, 'multi_tot'] == sum_df.loc[cl, 'cl_tot'])

    if normalize:
        sum_df_n = pd.DataFrame(index=classes)
        
        expl['single_of_total'] = 'percentage of the single labeled of a class in all images with this label in the dataset, ex. label=(bird)'
        sum_df_n['single_of_total'] = sum_df['single_tot'] / sum_df['cl_tot']
        
        single_cl_columns = [k for k in sum_df.columns if ('single' in k) and (k != 'single_top_err_cl') ]
        for col in single_cl_columns:
            sum_df_n[col] = sum_df[col] / sum_df['single_tot']
            
        sum_df_n['single_top_err_cl']   = sum_df['single_top_err_cl']
        
        expl['multi_of_total'] = 'percentage of the images that have this label along with other labels, ex. label=(bird+tree) (bird+house)'
        sum_df_n['multi_of_total'] = sum_df['multi_tot'] / sum_df['cl_tot']
        
        multi_cl_columns = [k for k in sum_df.columns if ('multi' in k) ]
        
        for col in multi_cl_columns:
            sum_df_n[col] = sum_df[col] / sum_df['multi_tot']
        
        return sum_df_n
    
    if return_expl:
        return sum_df , expl
        
    return sum_df
#%%

def get_thumbnail(path):
    if os.path.isfile(path):
        im = Image.open(path)
    else: 
        im = Image.new('RGB', (300,187), (192, 192, 192))
    
    im.thumbnail((300, 300), Image.LANCZOS)
    return im

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    
    if isinstance(im, torch.Tensor):
        #TODO: reverse normalization
        #norm_mean = torch.Tensor(joconde_mean_animals)
        #norm_std  = torch.Tensor(joconde_std_animals)
        #im = im.permute(1,2,0) #* norm_std + norm_mean
        im = torchvision.transforms.functional.to_pil_image(im)
        im.thumbnail((150, 150), Image.LANCZOS)
        
    if isinstance(im, np.ndarray):
        if (im.ndim >2) : 
            im = torchvision.transforms.functional.to_pil_image(torch.Tensor(im).permute(2,0,1))
        else: #heatmap
            cm_jet = colormap.get_cmap('jet')
            im = cm_jet(im/(im.max()-im.min()+0.0000001), alpha=0.4)
            im = np.uint8(im * 255)
            im = Image.fromarray(im[:,:,:3])
            
        im.thumbnail((150, 150), Image.LANCZOS)
        
    if (im.width < 150):
        im = im.resize((150,150) , Image.BOX)
        
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()
    
# def image_base64(im):
#     if isinstance(im, str):
#         im = get_thumbnail(im)
    
#     if isinstance(im, torch.Tensor):
#         #TODO: reverse normalization
#         #norm_mean = torch.Tensor( [0.5, 0.5, 0.5])
#         #norm_std  = torch.Tensor( [0.5, 0.5, 0.5])
#         #im = im.transpose(1,2,0) * norm_std + norm_mean
#         im = torchvision.transforms.functional.to_pil_image(im)
#         im.thumbnail((150, 150), Image.LANCZOS)
        
#     with BytesIO() as buffer:
#         im.save(buffer, 'jpeg')
#         return base64.b64encode(buffer.getvalue()).decode()

#def image_formatter(im):
#    return f'<img src="{im}">'

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def image_formatter_pop(im): #TODO: finish
    return f'<a href="#" onclick="window.open(\'{im}\'); return false" > <img src="data:image/jpeg;base64,{image_base64(im)}"> </a>'

def label_formatter(a):
    return a.replace('\n','<br>')

def prediction_formatter(a):
    #return '<br>'.join(a.split(sep='\n')[:10])
    return a.replace('\n','<br>')#.replace(' ', '&nbsp')

def repr_formatter(a):
    return '<br>'.join(wrap(a, 30 ))    

def image_url_formatter(url):
    return f'<a target="_blank" href={"%s"}> <img src={"%s"} width="150"> </a>' % (url, url)
#%%
def gather_annotation_target(image_set, index):
    file_name = image_set.samples[index][0].replace('/', '\\')
    ref = image_set.samples[index][2]
    repr = image_set.samples[index][3]
    target = image_set.targets[index]
    target_annot = target_to_text(image_set.classes, target, separator='\n')
        
    return '\n'.join( ('%s' % np.array(target),
                       '',
                       '%s' % target_annot,
                        '',
                        '\n'.join(wrap(repr, 40)),
                        '',
                        ref,
                        '',
                        file_name) )

def gather_annotation_pred(image_set, scores, y_pred, index, show_max_pred_labels=9999):
    
    show_max_pred_labels = min(len(image_set.classes), show_max_pred_labels)
    
    pred = y_pred[index]
    pred_annot = '\n'.join(prediction_to_text(image_set.classes, scores[index] , separator=',').split(',')[:show_max_pred_labels])
    #pred_annot = prediction_to_text(image_set.classes, scores[index] , separator='\n')
    
    return '\n'.join( ('%s' % np.array(pred),
                       '',
                        pred_annot) )

def gather_annotation(image_set, scores, y_pred, index):
    return '\n'.join( (gather_annotation_target(image_set, index),
                       '',
                       gather_annotation_pred(image_set, scores, y_pred, index) ))
#%%
def visualize_classification_HTML(data_set, pred_labels, score_tensor, index_pick,
                                                             title='', crop_idx=None, show_max_pred_labels=9999, 
                                                             file_name=None):
    df = pd.DataFrame(columns= ['image',  'crop', 'labels', 'prediction' ])

    df.image = [data_set.samples[i][0].replace('/', '\\') for i in index_pick]
    df.crop  = [data_set[i][0] for i in index_pick]
    df.labels = [gather_annotation_target(data_set, i) for i in index_pick ] 
    df.prediction = [gather_annotation_pred(data_set, score_tensor, pred_labels, i, show_max_pred_labels=show_max_pred_labels) for i in index_pick ] 
    #df.annotation = [gather_annotation(data_set, score_tensor, pred_labels, i) for i in testset_idx_pick ] 

    #df.head()
    pd.set_option('display.max_colwidth', None)
    pd.set_option('colheader_justify', 'center') 

    df.style.set_table_styles([ dict(selector='table', props=[('text-align', 'left') , 
                                                              ('vertical-align', 'top') ] ) ])
    
    formatters_dict={'image': image_formatter,
                     'crop': image_formatter,
                     'labels': label_formatter, 
                     'prediction': prediction_formatter, 
                     'annotation': label_formatter}
    

    return df.to_html( file_name,
                        formatters=formatters_dict, 
                        escape=False,
                        index=False)    

#%%
def compare_classification_HTML(data_set, pred_labels1, score_tensor1,
                                pred_labels2, score_tensor2,
                                index_pick,
                                title='', crop_idx=None, show_max_pred_labels=9999, 
                                file_name=None):
    df = pd.DataFrame(columns= ['image',  'labels', 'prediction1', 'prediction2' ])

    df.image = [data_set.samples[i][0].replace('/', '\\') for i in index_pick]

    df.labels = [gather_annotation_target(data_set, i) for i in index_pick ] 
    df.prediction1 = [gather_annotation_pred(data_set, score_tensor1, pred_labels1, i, show_max_pred_labels=show_max_pred_labels) for i in index_pick ] 
    df.prediction2 = [gather_annotation_pred(data_set, score_tensor2, pred_labels2, i, show_max_pred_labels=show_max_pred_labels) for i in index_pick ] 

    #df.annotation = [gather_annotation(data_set, score_tensor, pred_labels, i) for i in testset_idx_pick ] 

    #df.head()
    pd.set_option('display.max_colwidth', None)
    pd.set_option('colheader_justify', 'center') 

    df.style.set_table_styles([ dict(selector='table', props=[('text-align', 'left') , 
                                                              ('vertical-align', 'top') ] ) ])
    
    formatters_dict={'image': image_formatter,
                     'crop': image_formatter,
                     'labels': label_formatter, 
                     'prediction1': prediction_formatter, 
                     'prediction2': prediction_formatter, 
                     'annotation': label_formatter}
    

    return df.to_html( file_name,
                        formatters=formatters_dict, 
                        escape=False,
                        index=False)        