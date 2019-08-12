#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:42:22 2019

@author: eleftherios
"""

from keras.layers import Conv2D, BatchNormalization, Activation, Dense
from keras.layers import Flatten, MaxPooling2D, Dropout, Add, concatenate, AveragePooling2D
from keras import regularizers

def stem(previous_layer, filters, hyperparameters):
    filters_factor = filters/256
    conv1 = Conv2D(
                filters=int(32*filters_factor),
                kernel_size=[3,3],
                strides=[2,2],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="stem_conv_1")(previous_layer)
    conv2 = Conv2D(
                filters=int(32*filters_factor),
                kernel_size=[3,3],
                strides=[1,1],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="stem_conv_2")(conv1)
    conv3 = Conv2D(
                filters=int(64*filters_factor),
                kernel_size=[3,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="stem_conv_3")(conv2)
    mp=MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding="valid")(conv3)
    conv4 = Conv2D(
                filters=int(80*filters_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="stem_conv_4")(mp)
    conv5 = Conv2D(
                filters=int(192*filters_factor),
                kernel_size=[3,3],
                strides=[1,1],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="stem_conv_5")(conv4)
    conv6 = Conv2D(
                filters=int(256*filters_factor),
                kernel_size=[3,3],
                strides=[2,2],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="stem_conv_6")(conv5)
    if hyperparameters["batch_normalization"]:
        bn = BatchNormalization(name="stem_bn")(conv6)
        output_layer = Activation(hyperparameters["activation"], name="stem_out")(bn)
    else:
        output_layer = Activation(hyperparameters["activation"], name="stem_out")(conv6)
    
    return output_layer

def inception_a(previous_layer, index, hyperparameters):
    column_one = Conv2D(
                filters=32,
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_one_"+str(index))(previous_layer)
    
    column_two = Conv2D(
                filters=32,
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.1_"+str(index))(previous_layer)
    column_two_b = Conv2D(
                filters=32,
                kernel_size=[3,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.2_"+str(index))(column_two)
    
    column_three = Conv2D(
                filters=32,
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.1_"+str(index))(previous_layer)
    column_three_b = Conv2D(
                filters=48,
                kernel_size=[3,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.2_"+str(index)) (column_three)    
    column_three_c = Conv2D(
                filters=64,
                kernel_size=[3,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.3_"+str(index)) (column_three_b)
    
    concat_layer = concatenate([column_one,column_two_b,column_three_c],axis=-1)
    
    conv_linear = Conv2D(
                filters=384,
                kernel_size=[1,1],
                strides=[1,1],
                activation="linear",
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="output_1x1_"+str(index))(concat_layer)
    
    concat_layer2 = concatenate([previous_layer,conv_linear],axis=-1)
    
    residual_connection = Add()(concat_layer2)
    
    if hyperparameters["batch_normalization"]:
        bn = BatchNormalization(name="stem_bn")(residual_connection)
        output_layer = Activation(hyperparameters["activation"], name="inca_out")(bn)
    else:
        output_layer = Activation(hyperparameters["activation"], name="inca_out")(residual_connection)
    return output_layer

def inception_b(previous_layer, index, hyperparameters):
    column_one = Conv2D(
                filters=192,
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_one_"+str(index))(previous_layer)
        
    column_two = Conv2D(
                filters=128,
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.1_"+str(index))(previous_layer)
    column_two_b = Conv2D(
                filters=160,
                kernel_size=[1,7],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.2_"+str(index)) (column_two)    
    column_two_c = Conv2D(
                filters=192,
                kernel_size=[7,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.3_"+str(index)) (column_two_b)
    
    concat_layer = concatenate([column_one,column_two_c],axis=-1)
    
    conv_linear = Conv2D(
                filters=1154,
                kernel_size=[1,1],
                strides=[1,1],
                activation="linear",
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="output_1x1_"+str(index))(concat_layer)
    
    concat_layer2 = concatenate([previous_layer,conv_linear],axis=-1)
    
    residual_connection = Add()(concat_layer2)
    
    if hyperparameters["batch_normalization"]:
        bn = BatchNormalization(name="stem_bn")(residual_connection)
        output_layer = Activation(hyperparameters["activation"], name="incb_out")(bn)
    else:
        output_layer = Activation(hyperparameters["activation"], name="incb_out")(residual_connection)
    return output_layer

def inception_c(previous_layer, index, hyperparameters):
    column_one = Conv2D(
                filters=192,
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_one_"+str(index))(previous_layer)
        
    column_two = Conv2D(
                filters=192,
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.1_"+str(index))(previous_layer)
    column_two_b = Conv2D(
                filters=224,
                kernel_size=[1,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.2_"+str(index)) (column_two)    
    column_two_c = Conv2D(
                filters=256,
                kernel_size=[3,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.3_"+str(index)) (column_two_b)
    
    concat_layer = concatenate([column_one,column_two_c],axis=-1)
    
    conv_linear = Conv2D(
                filters=2048,
                kernel_size=[1,1],
                strides=[1,1],
                activation="linear",
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="output_1x1_"+str(index))(concat_layer)
    
    concat_layer2 = concatenate([previous_layer,conv_linear],axis=-1)
    
    residual_connection = Add()(concat_layer2)
    
    if hyperparameters["batch_normalization"]:
        bn = BatchNormalization(name="stem_bn")(residual_connection)
        output_layer = Activation(hyperparameters["activation"], name="incc_out")(bn)
    else:
        output_layer = Activation(hyperparameters["activation"], name="incc_out")(residual_connection)
    return output_layer

def reduction_a(previous_layer, filters, hyperparameters):
    filters_factor = filters/384    
    column_one=MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding="valid")(previous_layer)
    
    column_two = Conv2D(
                filters=int(384*filters_factor),
                kernel_size=[3,3],
                strides=[2,2],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two_reda")(previous_layer)
    
    column_three = Conv2D(
                filters=int(256*filters_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three_1_reda")(previous_layer)
    column_three_b = Conv2D(
                filters=int(256*filters_factor),
                kernel_size=[3,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three_2_reda") (column_three)    
    column_three_c = Conv2D(
                filters=int(384*filters_factor),
                kernel_size=[3,3],
                strides=[2,2],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three_3_reda") (column_three_b)
    
    output_layer = concatenate([column_one,column_two,column_three_c],axis=-1)
    
    return output_layer

def reduction_b(previous_layer, filters, hyperparameters):
    filters_factor = filters/320    
    
    column_one=MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding="valid")(previous_layer)
    
    column_two = Conv2D(
                filters=int(256*filters_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two_1_redb")(previous_layer)
    column_two_b = Conv2D(
                filters=int(384*filters_factor),
                kernel_size=[3,3],
                strides=[2,2],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two_2_redb")(column_two)
    
    column_three = Conv2D(
                filters=int(256*filters_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two_3_redb")(previous_layer)
    column_three_b = Conv2D(
                filters=int(288*filters_factor),
                kernel_size=[3,3],
                strides=[2,2],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two_4_redb")(column_three)
    
    column_four = Conv2D(
                filters=int(256*filters_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three_1_redb")(previous_layer)
    column_four_b = Conv2D(
                filters=int(288*filters_factor),
                kernel_size=[3,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three_2_redb") (column_four)    
    column_four_c = Conv2D(
                filters=int(320*filters_factor),
                kernel_size=[3,3],
                strides=[2,2],
                padding="valid",
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three_3_redb") (column_four_b)
    
    concat_layer = concatenate([column_one,column_two_b,column_three_b,column_four_c],axis=-1)
    
    return concat_layer

class Medinception:
    def __init__(self):
        self.data = []
    #test model
    def get_model(input_layer, hyperparameters):
        #Custom Inception model
        #Stem module
        stem_module = stem(input_layer,hyperparameters["filters"][0], hyperparameters)
        
        #Inception-A module
        inca_module = inception_a(stem_module,0, hyperparameters)
        for index in range (1,hyperparameters["filters"][1]):
            inca_module = inception_a(inca_module, index+1, hyperparameters)
        
        #Reduction-A module
        reda_module = reduction_a(inca_module,hyperparameters["filters"][2], hyperparameters)
        
        #Inception-B module
        incb_module = inception_b(reda_module,0, hyperparameters)
        for index in range(1,hyperparameters["filters"][3]):
            incb_module = inception_b(incb_module,index+1, hyperparameters)
        
        #Reduction-B module
        redb_module = reduction_b(incb_module,hyperparameters["filters"][4], hyperparameters)
        
        #Inception-C module
        incc_module = inception_c(redb_module,0, hyperparameters)
        for index in range(1,hyperparameters["filters"][5]):
            incc_module = inception_c(incc_module,index+1, hyperparameters)
        
        #Average Pooling
        avg_pooling_layer = AveragePooling2D()(incc_module)
        flatten_layer = Flatten()(avg_pooling_layer)
        
        #Neural Network with Dropout
        nn = Dense(units=hyperparameters["neurons"], activation=hyperparameters["activation"], name="hidden_layer")(flatten_layer)
        dropout_layer = Dropout(rate=hyperparameters["dropout"],name="drop")(nn)
        
        #Classification Layer
        classification_layer = Dense(units=hyperparameters["num_classes"], activation=hyperparameters["classifier"], name=hyperparameters["classifier"]+"_layer")(dropout_layer)
        return classification_layer