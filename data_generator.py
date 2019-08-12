#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

@original source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""
from data_augmentation import DataAugmentation
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, hypes, training=True):
        'Initialization'
        self.hypes = hypes
        self.dim = hypes["input_shape"][0:(len(hypes["input_shape"])-1)] # [vol,x,y]
        self.batch_size = hypes["batch_size"]
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = hypes["input_shape"][(len(hypes["input_shape"])-1)] # color channels
        self.n_classes = hypes["num_classes"]
        self.shuffle = hypes["generator_shuffle"]
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(self.hypes["dataset_dir"] + ID + '.npy')       
            
            # Store class
            y[i] = self.labels[ID]
            
        #apply data augmentation
        if self.training == True and self.hypes["data_augmentation"] == True and self.hypes["offline_augmentation"] == False:
            X_augmented, y_augmented = DataAugmentation.apply_augmentation(list(X),list(y),self.hypes)
            X = np.concatenate((X,X_augmented))
            y = np.concatenate((y,y_augmented))
            
        if self.hypes["loss"] != "scc":
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y