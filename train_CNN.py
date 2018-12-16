#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""
from dataset import DataConverter
from data_generator import DataGenerator
from model import Model
import numpy as np
from utils import Utils
from sklearn.model_selection import StratifiedKFold

hyperparameters = Utils.get_hypes()

#input images to npy for generator
DataConverter(hyperparameters).convert_nrrd_to_npy()
labels = np.load(hyperparameters["dataset_dir"] + "labels" + '.npy').astype(int)
pids = np.load(hyperparameters["dataset_dir"] + "pids" + '.npy')

#stratification + crossvalidation
tst_split_index = 1
val_split_index = 1
skf_tr_tst = StratifiedKFold(n_splits=hyperparameters["kfold"][0],shuffle=hyperparameters["shuffle"])
skf_tr_val = StratifiedKFold(n_splits=hyperparameters["kfold"][1],shuffle=hyperparameters["shuffle"])
for trval_index, tst_index in skf_tr_tst.split(pids):
    for tr_index, val_index in skf_tr_val.split(pids[trval_index]):
        #network version
        version = str(tst_split_index)+"."+str(val_split_index)
        hyperparameters["version"] = "network_version:"+version+"/"
        
        #make dirs        
        Utils.make_dirs(version,hyperparameters)        
        
        #save patient id per network version
        Utils.save_skf_pids(version,pids[tr_index],pids[val_index],pids[tst_index],hyperparameters)        
        
        # Generators
        training_generator = DataGenerator(pids[tr_index], labels, hyperparameters, training=True)
        validation_generator = DataGenerator(pids[val_index], labels, hyperparameters, training=False)
        testing_generator = DataGenerator(pids[tst_index], labels, hyperparameters, training=False)
        
        #create network
        cnn = Model.get_model(hyperparameters)
        
        #fit network
        cnn = Model.train_model(cnn,hyperparameters,training_generator,validation_generator)
        
        #test set performance
        Model.test_model(cnn,hyperparameters,testing_generator)
        
        #save current hypes
        Utils.save_hypes(hyperparameters["chkp_dir"]+hyperparameters["version"], "hypes"+version, hyperparameters)
        
        #Update version indeces
        val_split_index+=1
    tst_split_index+=1