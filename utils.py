#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""
import json
import keras
import numpy as np
import os

class Utils:
    def __init__(self):
        self.data = []
    
    def make_dirs(version, hypes):
        if not os.path.exists(hypes["chkp_dir"]):
            os.makedirs(hypes["chkp_dir"])
        os.makedirs(hypes["chkp_dir"]+hypes["version"])
        os.makedirs(hypes["chkp_dir"]+hypes["version"]+hypes["log_dir"])
        
    def save_skf_pids(version,training,validation,testing,hypes):
        np.save(hypes["chkp_dir"]+hypes["version"]+"pids_tr"+version,training)
        np.save(hypes["chkp_dir"]+hypes["version"]+"pids_val"+version,validation)
        np.save(hypes["chkp_dir"]+hypes["version"]+"pids_test"+version,testing)
    
    def volume_to_image(mri,labels, roi_map):
        f_mri=[]
        f_l=[]
        for i in range(0, len(roi_map)):
            for k in range(0,len(roi_map[i])):
                if (1 in np.reshape(roi_map[i][k],-1)) is True: #if image with cancer region
                    f_mri.append(mri[i][k])
                    f_l.append(labels[i])                
        return f_mri, f_l

    #hyperparameters json file: read
    def get_hypes(path="hypes"):        
        with open(path, encoding='utf-8') as file:
            hypes = json.load(file)
        return hypes
    #hyperparameters json file: save
    def save_hypes(path, filename, hypes):        
        with open(path+filename,'w') as file:
            json.dump(hypes,file)

    #callbacks
    def get_callbacks(hyperparameters):
        #training console feedback
        class AccuracyHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.acc = []
        
            def on_epoch_end(self, batch, logs={}):
                self.acc.append(logs.get('acc'))
        history = AccuracyHistory()
        
        #saves metrics per epoch
        csv_logger = keras.callbacks.CSVLogger(hyperparameters["chkp_dir"]+hyperparameters["version"]+hyperparameters["log_dir"]+hyperparameters["log_name"])
        
        #tensorboard
        tb = keras.callbacks.TensorBoard(hyperparameters["chkp_dir"]+hyperparameters["version"]+hyperparameters["log_dir"]+hyperparameters["tb_dir"])
        
        #saves models per epoch
        filepath = hyperparameters["chkp_dir"]+hyperparameters["version"]+"weights-chpoint-{epoch:02d}-{val_accuracy:.2f}.h5"
        chkp = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               save_best_only=True,
                                               monitor='val_accuracy',
                                               mode="max")
        
        #early stop
        early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                   patience=hyperparameters["early_stop_patience"],
                                                   mode="max")
                                                   #,restore_best_weights=True)
        return [history,csv_logger,tb,chkp,early_stop]