#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""
from keras.layers import Conv3D,BatchNormalization,Activation,concatenate
from keras.layers import MaxPooling3D,AveragePooling3D,Flatten,Dense,Dropout
from keras import regularizers

def zita(input_layer,hyperparameters):
    index=5
    c_1 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,1,1),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_1_"+str(index))(input_layer)
    c_3 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,3,3),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_3_"+str(index))(input_layer)
    c_5 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,5,5),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_5_"+str(index))(input_layer)
    c_7 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,7,7),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_7_"+str(index))(input_layer)
    c_9 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,9,9),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_9_"+str(index))(input_layer)
    c_11 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,11,11),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_11_"+str(index))(input_layer)
    c_13 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,13,13),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_13_"+str(index))(input_layer)
    
    concatenation_layer = concatenate([c_1,c_3,c_5,c_7,c_9,c_11,c_13],axis=-1)
    max_pool = MaxPooling3D(pool_size=(2,2,2))(concatenation_layer)
    
    if hyperparameters["batch_normalization"]:
        bn=BatchNormalization(name="bn_"+str(index))(max_pool)
        output=Activation(hyperparameters["activation"], name="first_layer_out")(bn)
    else:
        output=Activation(hyperparameters["activation"], name="first_layer_out")(max_pool)
    
    return output

def epsilon(input_layer,hyperparameters):
    index=4
    c_1 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,1,1),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_1_"+str(index))(input_layer)
    c_3 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,3,3),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_3_"+str(index))(input_layer)
    c_5 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,5,5),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_5_"+str(index))(input_layer)
    c_7 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,7,7),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_7_"+str(index))(input_layer)
    c_9 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,9,9),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_9_"+str(index))(input_layer)
    c_11 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,11,11),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_11_"+str(index))(input_layer)
       
    concatenation_layer = concatenate([c_1,c_3,c_5,c_7,c_9,c_11],axis=-1)
    max_pool = MaxPooling3D(pool_size=(2,2,2))(concatenation_layer)
    
    if hyperparameters["batch_normalization"]:
        bn=BatchNormalization(name="bn_"+str(index))(max_pool)
        output=Activation(hyperparameters["activation"], name="second_layer_out")(bn)
    else:
        output=Activation(hyperparameters["activation"], name="second_layer_out")(max_pool)
    
    return output

def delta(input_layer,hyperparameters):
    index=3
    c_1 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,1,1),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_1_"+str(index))(input_layer)
    c_3 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,3,3),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_3_"+str(index))(input_layer)
    c_5 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,5,5),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_5_"+str(index))(input_layer)
    c_7 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,7,7),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_7_"+str(index))(input_layer)
    c_9 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,9,9),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_9y_"+str(index))(input_layer)
    
    concatenation_layer = concatenate([c_1,c_3,c_5,c_7,c_9],axis=-1)
    max_pool = MaxPooling3D(pool_size=(2,2,2))(concatenation_layer)
    
    if hyperparameters["batch_normalization"]:
        bn=BatchNormalization(name="bn_"+str(index))(max_pool)
        output=Activation(hyperparameters["activation"], name="third_layer_out")(bn)
    else:
        output=Activation(hyperparameters["activation"], name="third_layer_out")(max_pool)
    
    return output
def gama(input_layer,hyperparameters):
    index=2
    c_1 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,1,1),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_1_"+str(index))(input_layer)
    c_3 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,3,3),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_3_"+str(index))(input_layer)
    c_5 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,5,5),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_5_"+str(index))(input_layer)
    c_7 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,7,7),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_7_"+str(index))(input_layer)
    
    concatenation_layer = concatenate([c_1,c_3,c_5,c_7],axis=-1)
    max_pool = MaxPooling3D(pool_size=(2,2,2))(concatenation_layer)
    
    if hyperparameters["batch_normalization"]:
        bn=BatchNormalization(name="bn_"+str(index))(max_pool)
        output=Activation(hyperparameters["activation"], name="fourth_layer_out")(bn)
    else:
        output=Activation(hyperparameters["activation"], name="fourth_layer_out")(max_pool)
    
    return output
def beta(input_layer,hyperparameters):
    index=1
    c_1 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,1,1),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_1_"+str(index))(input_layer)
    c_3 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,3,3),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_3_"+str(index))(input_layer)
    c_5 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,5,5),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_5_"+str(index))(input_layer)
    
    concatenation_layer = concatenate([c_1,c_3,c_5],axis=-1)    
    max_pool = MaxPooling3D(pool_size=(2,2,2))(concatenation_layer)
    
    if hyperparameters["batch_normalization"]:
        bn=BatchNormalization(name="bn_"+str(index))(max_pool)
        output=Activation(hyperparameters["activation"], name="fifth_layer_out")(bn)
    else:
        output=Activation(hyperparameters["activation"], name="fifth_layer_out")(max_pool)
    
    return output
def alpha(input_layer,hyperparameters):
    index=0
    c_1 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,1,1),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_1_"+str(index))(input_layer)
    c_3 = Conv3D(
                    filters=hyperparameters["filters"][index],
                    kernel_size=(3,3,3),
                    strides=(1,1,1),
                    padding=hyperparameters["padding"],
                    kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                    name="conv_3_"+str(index))(input_layer)
    
    concatenation_layer = concatenate([c_1,c_3],axis=-1)
    max_pool = MaxPooling3D(pool_size=(2,2,2))(concatenation_layer)
    
    if hyperparameters["batch_normalization"]:
        bn=BatchNormalization(name="bn_"+str(index))(max_pool)
        output=Activation(hyperparameters["activation"], name="sixth_layer_out")(bn)
    else:
        output=Activation(hyperparameters["activation"], name="sixth_layer_out")(max_pool)
    
    return output
class DWNet:
    def __init__(self):
        self.data = []
    def get_features(cnn,test_set):        
        return cnn.predict(test_set, batch_size=1)
        
    #create model from hypes dict
    def get_model(input_layer,hyperparameters, visual=False):
        
        alpha_out = alpha(input_layer,hyperparameters)
        beta_out = beta(alpha_out,hyperparameters)
        gama_out = gama(beta_out,hyperparameters)
#        delta_out = delta(gama_out,hyperparameters)
#        epsilon_out = epsilon(delta_out,hyperparameters)
#        zita_out = zita(epsilon_out,hyperparameters)
        conv_out = gama_out
        
        if hyperparameters["avgpool"]:
            avg = AveragePooling3D(name="AvgPooling3D")(conv_out)
            flatten = Flatten()(avg)
        else:
            #flatten
            flatten = Flatten()(conv_out)
#            flatten = Flatten()(sixth_out)
        
            
        #Neural Network with Dropout
        nn = Dense(units=hyperparameters["neurons"][0], activation=hyperparameters["activation"], name="nn_layer")(flatten)
        do = Dropout(rate=hyperparameters["dropout"],name="drop")(nn)
        if not visual:
            #Classification Layer
            final_layer = Dense(units=hyperparameters["num_classes"], activation=hyperparameters["classifier"], name=hyperparameters["classifier"]+"_layer")(do)
        else:
            final_layer = do
        return final_layer
