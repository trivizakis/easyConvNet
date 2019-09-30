#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""

import keras
from keras.models import Sequential,Model
from keras.layers import LSTM, TimeDistributed,Conv1D, Conv2D, Conv3D, BatchNormalization, Activation
from keras.layers import Flatten, Dense, Dropout, Input, AveragePooling1D, AveragePooling2D,AveragePooling3D
from keras import regularizers
from utils import Utils
import numpy as np
from sklearn.metrics import confusion_matrix,auc,classification_report,accuracy_score,roc_auc_score,roc_curve
from medinception import Medinception as MI
from widenet1d import WideNet as WN
from deep_wide_analysis import DWNet as DW
from DWNet3D import DWNet as DW3D
import matplotlib.pyplot as plt

#create a convolutional layer
def get_conv_layer(hyperparameters, index):
    if hyperparameters["conv_type"]=="1d":
        if index == 0: #first layer
            conv_layer = Conv1D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=hyperparameters["kernel_size"][index],
                    strides=hyperparameters["strides"][index],
                    padding=hyperparameters["padding"],
                    input_shape=hyperparameters["input_shape"], #only at first layer
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv"+str(index))
        else: #every other layer
            conv_layer = Conv1D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=hyperparameters["kernel_size"][index],
                    strides=hyperparameters["strides"][index],
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv"+str(index))   
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

class CustomModel:
    def __init__(self):
        self.data = []
    def test_fitted_model(model,chckp_dir,hyperparameters,testing_set,testing_labels):
        if hyperparameters['loss']!="scc":
            y_true = np.argmax(testing_labels,axis=-1)
        else:            
            y_true = testing_labels
        confidence = model.predict(x=testing_set)
        y_pred = np.argmax(confidence,axis=-1)
        acc = accuracy_score(y_true,y_pred)
        report = classification_report(y_true,y_pred, target_names=hyperparameters["class_names"])
        cm = confusion_matrix(y_true,y_pred)
            
        roc = roc_auc_score(y_true,y_pred)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
        auc_score = auc(fpr, tpr)

        #print metrics
        print("Test Accuracy: {0:.2f}%".format(acc*100))
        print("Test roc_auc_score: {0:.2f}%".format(roc*100))
        print(report)
        print("__Confusion matrix__\n")
        print(cm)
        
        #save testing.txt
        with open(chckp_dir+hyperparameters["log_dir"]+"test_metrics.txt", "w") as text_file:
            text_file.write('Test accuracy: '+str(acc))
            text_file.write('\n')
            text_file.write('Test roc_score: '+str(roc))
            text_file.write('\n')
            text_file.write('Test auc_score: '+str(auc_score))
            text_file.write('\n')
            text_file.write(str(report))
            text_file.write('\n')
            text_file.write('Confusion matrix:\n')
            text_file.write(str(cm))
            text_file.write('\n')
            text_file.write('Confidence:\n')
            text_file.write(str(confidence))
            text_file.write('\n')
            text_file.write('Predictions:\n')
            text_file.write(str(y_pred))
            
        return y_true, confidence
            
    #test model
    def test_model(model,hyperparameters,testing_generator):
        confidence = model.predict_generator(generator=testing_generator,
                            use_multiprocessing=hyperparameters["use_multiprocessing"],
                            workers=hyperparameters["workers"])
        y_pred = np.argmax(confidence,axis=1)
        y_true = np.empty((len(y_pred)), dtype=int)
        for i, ID in enumerate(testing_generator.list_IDs):
#            if len(y_pred) == i:
#                break
#            print(len(testing_generator.labels))
#            print(len(y_true))
#            print(len(y_pred))
#            print(len(testing_generator.list_IDs))
            y_true[i]=testing_generator.labels[ID]
        acc = accuracy_score(y_true,y_pred)
        report = classification_report(y_true,y_pred, target_names=hyperparameters["class_names"])# or instead y_pred --> confidence
        cm = confusion_matrix(y_true,y_pred)
        score = model.evaluate_generator(testing_generator)

        #print metrics
        print("Test Accuracy: {0:.2f}%".format(acc*100))
        print(model.metrics_names)
        print(score)
        print(report)
        print("_Confusion matrix_\n")
        print(cm)        
        
        #if binary classification
        if confidence.shape[1]==2:
            roc = roc_auc_score(y_true,confidence[:,1],average='weighted')        
            fpr, tpr, thresholds = roc_curve(y_true,confidence[:,1])#, pos_label=1)
            auc_score = auc(fpr, tpr)
        
        #if binary classification
        if confidence.shape[1]==2:
            print("Test roc_auc_score: {0:.2f}%".format(roc*100))
            print("Test auc_score: {0:.2f}%".format(auc_score*100))
        
        #save testing.txt
        with open(hyperparameters["chkp_dir"]+hyperparameters["version"]+hyperparameters["log_dir"]+"test_metrics.txt", "w") as text_file:
            text_file.write('Test accuracy: '+str(acc))
            text_file.write('\n')
            #if binary classification
            if confidence.shape[1]==2:
                text_file.write('Test roc_score: '+str(roc))
                text_file.write('\n')
                text_file.write('Test auc_score: '+str(auc_score))
                text_file.write('\n')
            text_file.write(str(report))
            text_file.write('\n')
            text_file.write('Confusion matrix:\n')
            text_file.write(str(cm))
            text_file.write('\n')
            text_file.write('Confidence:\n')
            text_file.write(str(confidence))
            text_file.write('\n')
            text_file.write('Predictions:\n')
            text_file.write(str(y_pred))
            
        return y_true, confidence
            
    
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
        
    #fit model
    def fit_model(model,hyperparameters,training_set,training_labels,validation_set,validation_labels):
        model.fit(x=training_set,
                  y=training_labels,
                  epochs=hyperparameters["epochs"],
                  callbacks=Utils.get_callbacks(hyperparameters),
                  validation_data=[validation_set,validation_labels],
                  shuffle=hyperparameters["shuffle"])
        return model
    #create model from hypes dict
    def get_model(hyperparameters, visual=False):
        
        if hyperparameters["MedInception"]:
            #custom inception-ResNet-v2
            input_layer = Input(shape=tuple(hyperparameters["input_shape"]))
            output_layer = MI.get_model(input_layer,hyperparameters)
            convnet = Model(inputs=input_layer, outputs=output_layer)  
        elif hyperparameters["DWNet"] and hyperparameters["conv_type"]=='1d':
            #widenet1d
            input_layer = Input(shape=tuple(hyperparameters["input_shape"]))
            output_layer = WN.get_model(input_layer,hyperparameters)
            convnet = Model(inputs=input_layer, outputs=output_layer)    
        elif hyperparameters["DWNet"] and hyperparameters["conv_type"]=='2d':
            #deep & wide net
            input_layer = Input(shape=tuple(hyperparameters["input_shape"]))
            output_layer = DW.get_model(input_layer,hyperparameters)
            convnet = Model(inputs=input_layer, outputs=output_layer)        
        elif hyperparameters["DWNet"] and hyperparameters["conv_type"]=='3d':
            #deep & wide net
            input_layer = Input(shape=tuple(hyperparameters["input_shape"]))
            output_layer = DW3D.get_model(input_layer,hyperparameters)
            convnet = Model(inputs=input_layer, outputs=output_layer)    
        else:
            #new network
            convnet = Sequential()
            #add convolutional layers
            for conv_layer_num in range(0,len(hyperparameters["filters"])): #infer number of layers
                convnet.add(get_conv_layer(hyperparameters,conv_layer_num))
                if hyperparameters["batch_normalization"]:
                    convnet.add(BatchNormalization(name="conv_bn"+str(conv_layer_num)))
                convnet.add(Activation(hyperparameters["activation"], name="conv"+str(conv_layer_num)+"_out"))
            
            if visual is False:
                #Average Pooling
                if hyperparameters["avgpool"]:
                    if hyperparameters["conv_type"]=="3d":
        #                print("no AvgPooling")                
                        convnet.add(AveragePooling3D(name="AvgPooling3D"))
                    elif hyperparameters["conv_type"]=="2d":
        #                print("no AvgPooling")                
                        convnet.add(AveragePooling2D(name="AvgPooling2D"))
                    elif hyperparameters["conv_type"]=="1d":
        #                print("no AvgPooling")                
                        convnet.add(AveragePooling1D(name="AvgPooling1D"))
                
                if hyperparameters["Recurrent_Layer"]:
                    for index, num_rnn_filters in enumerate(hyperparameters["rnn_filters"]):
                        convnet.add(LSTM(units=num_rnn_filters,activation='tanh',recurrent_activation='hard_sigmoid', dropout=hyperparameters["dropout"], return_sequences=True))
                        if hyperparameters["rnn_bn"]:
                            convnet.add(BatchNormalization(name="rnn_bn"+str(index)))
                    for index, fc_num in enumerate(hyperparameters["neurons"]):
                        convnet.add(TimeDistributed(get_fc_layer(hyperparameters, index)))
                        if hyperparameters["fc_bn"]:
                            convnet.add(BatchNormalization(name="fc_bn"+str(index)))
                    #add flatten
                    convnet.add(Flatten(name="flatten_layer"))  
                else:
                    #add flatten
                    convnet.add(Flatten(name="flatten_layer"))        
                    #add fc layers
                    for fc_layer_num in range(0,len(hyperparameters["neurons"])):
                        convnet.add(get_fc_layer(hyperparameters, fc_layer_num))
                        convnet.add(Dropout(rate=hyperparameters["dropout"],name="drop"+str(fc_layer_num)))
                        if hyperparameters["fc_bn"]:
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
        elif hyperparameters["loss"] == "hinge":
            loss=keras.losses.hinge
        elif hyperparameters["loss"] == "squared_hinge":
            loss=keras.losses.squared_hinge
        elif hyperparameters["loss"] == "categorical_hinge":
            loss=keras.losses.categorical_hinge     
        elif hyperparameters["loss"] == "logcosh":
            loss=keras.losses.logcosh          
        elif hyperparameters["loss"] == "huber_loss":
            loss=keras.losses.huber_loss                 
        elif hyperparameters["loss"] == "kullback_leibler_divergence":
            loss=keras.losses.kullback_leibler_divergence                       
        elif hyperparameters["loss"] == "poisson":
            loss=keras.losses.poisson                   
        elif hyperparameters["loss"] == "cosine_proximity":
            loss=keras.losses.cosine_proximity    
        elif hyperparameters["loss"] == "mean_squared_error":
            loss=keras.losses.mean_squared_error   
        elif hyperparameters["loss"] == "mean_squared_logarithmic_error":
            loss=keras.losses.mean_squared_logarithmic_error   
        elif hyperparameters["loss"] == "mean_absolute_error":
            loss=keras.losses.mean_absolute_error   
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
        print(convnet.summary())
        return convnet
