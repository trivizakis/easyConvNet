#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, BatchNormalization, Activation
from keras.layers import Flatten, Dense, Dropout
from keras import regularizers
from utils import Utils

#create a convolutional layer
def get_conv_layer(hyperparameters, index):
    if hyperparameters["conv_type"]=="1d":
        print("Not Supported: " + hyperparameters["conv_type"] + " ...Yet!")
    elif hyperparameters["conv_type"]=="2d":
        if index == 0: #first layer
            conv_layer = Conv2D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=hyperparameters["kernel_size"][index],
                    strides=hyperparameters["strides"][index],
                    padding=hyperparameters["padding"],
                    input_shape=hyperparameters["input_shape"], #only at first layer
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv"+str(index))
        else: #every other layer
            conv_layer = Conv2D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=hyperparameters["kernel_size"][index],
                    strides=hyperparameters["strides"][index],
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv"+str(index))              
    elif hyperparameters["conv_type"]=="3d":
        if index == 0: #first layer
            conv_layer = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=hyperparameters["kernel_size"][index],
                    strides=hyperparameters["strides"][index],
                    padding=hyperparameters["padding"],
                    input_shape=hyperparameters["input_shape"], #only at first layer
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv"+str(index))
        else: #every other layer
            conv_layer = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=hyperparameters["kernel_size"][index],
                    strides=hyperparameters["strides"][index],
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv"+str(index))
    else:
        print("Not Supported: " + hyperparameters["conv_type"])        
    return conv_layer

def get_fc_layer(hyperparameters, index):
    fc = Dense(units=hyperparameters["neurons"][index], activation=hyperparameters["activation"], name="fc"+str(index))
    return fc

class Model:
    def __init__(self):
        self.data = []
    #test model
    def test_model(model,hyperparameters,testing_generator):
        score = model.fit_evaluate(generator=testing_generator,
                            use_multiprocessing=hyperparameters["use_multiprocessing"],
                            workers=hyperparameters["workers"])
        #save testing.txt
        with open(hyperparameters["chkp_dir"]+hyperparameters["version"]+hyperparameters["log_dir"]+"test_score.txt", "w") as text_file:
            text_file.write('Test loss:'+str(score[0])+' Test accuracy:'+str(score[1]))
    
    #train model
    def train_model(model,hyperparameters,training_generator,validation_generator):
        model.fit_generator(generator=training_generator,
                            epochs=hyperparameters["epochs"],
                            callbacks=Utils.get_callbacks(hyperparameters),
                            validation_data=validation_generator,
                            use_multiprocessing=hyperparameters["use_multiprocessing"],
                            workers=hyperparameters["workers"],
                            shuffle=hyperparameters["shuffle"])
        return model
        
    #create model from hypes dict
    def get_model(hyperparameters):
        convnet = Sequential()
        #add convolutional layers
        for conv_layer_num in range(0,len(hyperparameters["filters"])): #infer number of layers
            convnet.add(get_conv_layer(hyperparameters,conv_layer_num))
            if hyperparameters["batch_normalization"]==True:
                convnet.add(BatchNormalization(name="conv_bn"+str(conv_layer_num)))
            convnet.add(Activation(hyperparameters["activation"], name="conv"+str(conv_layer_num)+"_out"))
        #add flatten
        convnet.add(Flatten(name="flat"))        
        #add fc layers
        for fc_layer_num in range(0,len(hyperparameters["neurons"])):
            convnet.add(get_fc_layer(hyperparameters, fc_layer_num))
            convnet.add(Dropout(rate=hyperparameters["dropout"],name="drop"+str(fc_layer_num)))
            if hyperparameters["fc_bn"]==True:
                convnet.add(BatchNormalization(name="fc_bn"+str(fc_layer_num)))
        #add classification layer
        convnet.add(Dense(units=hyperparameters["num_classes"], activation=hyperparameters["classifier"], name=hyperparameters["classifier"]+"_layer"))
        
        #keras loss functions by initials
        if hyperparameters["loss"] == "scc":
            loss=keras.losses.sparse_categorical_crossentropy
        elif hyperparameters["loss"] == "cc":
            loss=keras.losses.categorical_crossentropy
        elif hyperparameters["loss"] == "bc":
            loss=keras.losses.binary_crossentropy
        else:
            print("Not Supported: " + hyperparameters["loss"])
        
        #keras optimizers by name
        if hyperparameters["optimizer"] == "Adam":
            opt=keras.optimizers.Adam(lr=hyperparameters["learning_rate"],decay=hyperparameters["weight_decay"])
        elif hyperparameters["optimizer"] == "SGD":
            opt=keras.optimizers.SGD(lr=hyperparameters["learning_rate"])
        elif hyperparameters["optimizer"] == "Adagrad":
            opt=keras.optimizers.Adagrad(lr=hyperparameters["learning_rate"],decay=hyperparameters["weight_decay"])
        elif hyperparameters["optimizer"] == "Adadelta":
            opt=keras.optimizers.Adadelta(lr=hyperparameters["learning_rate"],decay=hyperparameters["weight_decay"])
        elif hyperparameters["optimizer"] == "Adamax":
            opt=keras.optimizers.Adamax(lr=hyperparameters["learning_rate"],decay=hyperparameters["weight_decay"])
        else:
            print("Not Supported: " + hyperparameters["optimizer"])
            
        convnet.compile(loss=loss,
                        optimizer=opt,
                        metrics=[hyperparameters["metric"]])
        
        return convnet
