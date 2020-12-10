# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:20:33 2019

@author: abobashe
"""

from __future__ import print_function
import torch
#import torch.nn as nn
#import torch.optim as optim

import torchvision.transforms as transforms
#import torchvision.datasets as dset

import os

from  data.image_dataset import JocondeDataset
import data.image_transforms as image_transforms
#import data.target_transforms as target_transforms
import model.train as train

from util import metadata_helpers as metadata
from util.metadata_helpers import monalia, jcl, notice, thesaurus

import pandas as pd

import config as cfg

from rdflib import Graph, URIRef, BNode, Literal
from rdflib import RDF, RDFS, XSD
from rdflib.namespace import SKOS

def preload_dataset():
 
    if cfg.args.arch == 'inception_v3':
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
        raise ValueError('unexplored model')

    test_trans = transforms.Compose([
        #transforms.Resize((model_image_size, model_image_size)), 
        transforms.Resize(cfg.args.input_image_size),
        transforms.CenterCrop(cfg.args.input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = dataset_mean, std = dataset_std),
    ])


    
    full_set = JocondeDataset(cfg.args.dataset_descr_file, 
                                cfg.args.image_root,
                                dataset_name = cfg.args.dataset,
                                label_column= cfg.args.dataset_label_column,
                                exclude_labels=cfg.args.dataset_exclude_labels,
                                multiple_labels = cfg.args.multi_label, 
                                #filter_dict= {'usage': ['val']}, 
                                transform=test_trans)
    
    full_loader = torch.utils.data.DataLoader(
                    dataset=full_set,
                    batch_size= cfg.args.batch_size,
                    shuffle=False,
                    drop_last = False,
                    num_workers=cfg.args.workers)
    
    
    print('Full set', full_set)
    print('    Labels:',full_set.labels_count)
    print()

    return full_loader

#%%
def scoring():
    
    checkpoint_dir = cfg.args.output_dir
    checkpoint_prefix = os.path.splitext(os.path.basename(cfg.args.model_param_file))[0]
    filename= '%s.checkpoint.pth.tar' % checkpoint_prefix
    model_checkpoint_file = os.path.join(checkpoint_dir, filename)
    
    score_file_name =  '%s.scores.pt' % checkpoint_prefix
    score_file = os.path.join(checkpoint_dir, score_file_name)
	
    rdf_file_name = '%s.ttl' %  checkpoint_prefix
    rdf_file = os.path.join(checkpoint_dir, rdf_file_name)
    
    checkpoint = torch.load(model_checkpoint_file)
    print(checkpoint.keys())
    
    class_count = next(reversed(checkpoint['state_dict'].values())).shape[0]
    print('Model Class Count:' , class_count)
    
    classes = checkpoint['classes']
    print('Model Classes:' , classes)
    
    full_loader = preload_dataset()
    #class_count = len(full_loader.dataset.classes)
            
    net = train.load_net(model_name= cfg.args.arch ,
                         class_count=class_count)
    net.load_state_dict(checkpoint['state_dict'])
    print('Model:', type(net))
     
    activation = train.set_activation(cfg.args.activation)
    print('Activation:', activation)
    
    scores = train.score(net, full_loader, activation, save_to_file= score_file)
   
    output_rdf(full_loader.dataset, scores.cpu(), rdf_file)
 
#%%
def output_rdf(data_set , scores, filename):

    g = metadata.create_graph()

    classifier_vocab = "REPR" #"DOMN"
    classifier_name = cfg.args.dataset
    classifier_descr = "Classifier trained on images labeled by the top %s terms from the MiC's list of 100" % len(data_set.classes)
    classifier_type = monalia.classifierRepresentedSubject
    classifier_id = monalia['classifier%s' % cfg.args.dataset.replace(' ', '')]
    
    # output classifier class
    clsfier = URIRef(classifier_type)
    g.add((clsfier , RDF.type, RDFS.Class))
    g.add((clsfier , monalia.vocabID , Literal(classifier_vocab))) #Literal('DOMN')))
    
    
    clsfier_sp = URIRef(classifier_id) 
    g.add((clsfier_sp , RDFS.subClassOf, clsfier))
    g.add((clsfier_sp , RDFS.label , Literal(classifier_name))) 
    g.add((clsfier_sp , RDFS.comment , Literal(classifier_descr))) 
    
    class_terms = get_term_URIs(data_set.classes)
    for i in class_terms.index:
        t = BNode()
        g.add( (t, RDF.type, class_terms[i]) )
        g.add( (t, SKOS.prefLabel, Literal(i, lang='fr')) )
        g.add ((clsfier_sp , monalia.conatainsClass, t))
		
    print(g.serialize(format='n3', encoding='utf-8').decode("utf-8"))
	
	
    #output scores for every image in the dataset
	
    # output all scores but it can be reduced reduce the number of top K stored predictions
    top_k = scores.shape[1] 

    for i, row in enumerate(data_set.samples):

        ref = row[2]

        classifier_bn = BNode()

        g.add( (notice[ref], monalia.imageClassifier, classifier_bn))
        g.add( (classifier_bn, RDF.type,   clsfier_sp ))

        pred_scores, pred_labels =  torch.topk(scores[i] , top_k ,0)
			   
        pred_score_dict = dict(zip( [data_set.classes[pl] for pl in pred_labels] , 
									pred_scores.numpy()))

        for r, label in enumerate(pred_score_dict):
            label_key_value = BNode()
			
            g.add( (classifier_bn, monalia.detected, label_key_value) )
	   #g.add( (label_key_value, monalia.label, Literal(label, lang='fr') ))#TODO: delete
            g.add( (label_key_value, RDF.type, class_terms[label] )) 
            g.add( (label_key_value, monalia.score, Literal(round(pred_score_dict[label], 4) , datatype=XSD.float)))

            if i % 1000 == 0:
                print (i, end=', ')
            elif i == 2:     
                sample_RDF = g.serialize(format='n3', encoding='utf-8').decode("utf-8")
        
    print('Done')
	
    print(sample_RDF)
	
    #save to TTL file.
	
    g.serialize(destination=filename, format='n3', encoding='utf-8')
    
    return g 

#%%
def get_term_URIs(classes):
    
    '''
    Get classes URIs from reprskos.rdf
    '''
   
    repr_thes_file_name = cfg.args.repr_thesaurus_file #os.path.join('C:\\Users\\abobashe\\Documents\\MonaLIA\\Joconde', 'reprskos.rdf')
    thes = metadata.create_graph()
    thes.parse(repr_thes_file_name, format='xml', encoding='utf-8')
    
    class_terms =  pd.Series(index=classes, dtype=object)

    for i, t in enumerate(class_terms.index):
        class_terms[t] = metadata.getJocondeTermByLabel_thesaurus_graph(thes, t)
    
    return class_terms

#%%
def binary_model_scoring():
    class_count = 2 #len(train_loader.dataset.classes)
    
    full_loader = preload_dataset()
    print(full_loader.dataset.classes)
    print(len(full_loader.dataset))
    
    net = train.load_net(model_name= cfg.args.arch ,
                         class_count=class_count)

    print('Model:', type(net))
    
    activation = train.set_activation('softmax')
    print('Activation:', activation)
   
    for one_class in full_loader.dataset.classes:
        print('*' * 120)
        print('    ', one_class)
        print('*' * 120)

        checkpoint_dir = cfg.args.output_dir
        checkpoint_prefix =  '%s_%s_%s.%s.%s' % (cfg.args.arch, cfg.args.database, cfg.args.dataset, one_class, cfg.args.param_file_suffix)
        checkpoint_file_name =  '%s.checkpoint.pth.tar' % checkpoint_prefix
        checkpoint_file = os.path.join(checkpoint_dir,checkpoint_file_name )
        
        score_file_name =  '%s.scores.pt' % checkpoint_prefix
        score_file = os.path.join(checkpoint_dir, score_file_name)
	
        rdf_file_name = '%s.ttl' %  checkpoint_prefix
        rdf_file = os.path.join(checkpoint_dir, rdf_file_name)

        checkpoint = torch.load(checkpoint_file)
        print(checkpoint.keys())
        
        net.load_state_dict(checkpoint['state_dict'])
        
        print(checkpoint['classes'])
        print(score_file)
        print(rdf_file)
    
        if not os.path.exists(score_file):
            scores = train.score(net, full_loader, activation, save_to_file= score_file)
        else:
            scores = torch.load(score_file).cpu()
   
        output_rdf_binary(full_loader.dataset, one_class, scores.cpu(), checkpoint_file, rdf_file)
#%%
def output_rdf_binary(data_set , theClass, scores, checkpoint_file, filename):

    g = metadata.create_graph()

    classifier_vocab = "REPR" #"DOMN"
    classifier_name = '%s_%s' % (data_set.name, theClass)
    classifier_descr = "Binary classifier for category '%s'@fr. Param file: %s " % (theClass , os.path.basename(checkpoint_file))
    classifier_type = monalia.classifierRepresentedSubjectBinary
    classifier_id = monalia['classifier_%s' % theClass.replace(' ','')]
    
    # output classifier class
    clsfier = URIRef(classifier_type)
    g.add((clsfier , RDF.type, RDFS.Class))
    g.add((clsfier , monalia.vocabID , Literal(classifier_vocab))) #Literal('DOMN')))
    
    
    clsfier_sp = URIRef(classifier_id) 
    g.add((clsfier_sp , RDFS.subClassOf, clsfier))
    g.add((clsfier_sp , RDFS.label , Literal(classifier_name))) 
    g.add((clsfier_sp , RDFS.comment , Literal(classifier_descr))) 
    
    checkpoint = torch.load(checkpoint_file)
    class_terms = get_term_URIs(checkpoint['classes'])
    print(class_terms)
    for i in class_terms.index:
        
        if class_terms[i] is not None:
            t = BNode()
            g.add( (t, RDF.type, class_terms[i]) )
            g.add( (t, SKOS.prefLabel, Literal(i, lang='fr')) )
            g.add ((clsfier_sp , monalia.conatainsClass, t))
		
    #print(g.serialize(format='n3', encoding='utf-8').decode("utf-8"))
	
	
    #output scores for every image in the dataset
	
    for i, row in enumerate(data_set.samples):

        ref = row[2]

        classifier_bn = BNode()

        g.add( (notice[ref], monalia.imageClassifier, classifier_bn))
        g.add( (classifier_bn, RDF.type,   clsfier_sp ))

        pred_scores, pred_labels =  torch.topk(scores[i] , 2 ,0)
			   
        pred_score_dict = dict(zip( [class_terms.index[pl] for pl in pred_labels] , 
									pred_scores.numpy()))

        for r, label in enumerate(pred_score_dict):
            if class_terms[label] is not None:
                label_key_value = BNode()
    			
                g.add( (classifier_bn, monalia.detected, label_key_value) )
                g.add( (label_key_value, RDF.type, class_terms[label] )) 
                g.add( (label_key_value, monalia.score, Literal(round(pred_score_dict[label], 4) , datatype=XSD.float)))

        if i % 10000 == 0:
            print (i, end=', ')
        elif i == 2:     
            sample_RDF = g.serialize(format='n3', encoding='utf-8').decode("utf-8")
            
        
    print('Done')
	
    print(sample_RDF)
	
    #save to TTL file.
	
    g.serialize(destination=filename, format='n3', encoding='utf-8')
    
    return g 
