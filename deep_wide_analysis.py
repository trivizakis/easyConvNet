#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:43:50 2019

@author: eleftherios
"""

from keras.layers import Conv2D, BatchNormalization, Activation, Dense
from keras.layers import Reshape, Flatten, MaxPooling2D, Dropout, Add, concatenate, AveragePooling2D
from keras import regularizers
import numpy as np

def deep_wide_top(previous_layer, filters, index, hyperparameters):
    filter_factor=filters/200
    
    #first column
    column_one = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_one_top_"+str(index))(previous_layer)
    
    #second column
    column_two = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.1_top_"+str(index))(previous_layer)
    column_two = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[3,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.2_top_"+str(index))(column_two)
    column_two = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[1,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.3_top_"+str(index))(column_two)
    
    #third column
    column_three = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.1_top_"+str(index))(previous_layer)
    column_three = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[5,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.2_top_"+str(index)) (column_three)    
    column_three = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[1,5],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.3_top_"+str(index)) (column_three)
    
    #fourth column
    column_four = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_four.1_"+str(index))(previous_layer)
    column_four = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[7,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_four.2_"+str(index)) (column_four)    
    column_four = Conv2D(
                filters=int(20*filter_factor),
                kernel_size=[1,7],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_four.3_"+str(index)) (column_four)
    
    #concatenation of feature maps
    concat_layer = concatenate([column_one,column_two,column_three,column_four],axis=-1)
    
    x = Conv2D(
                filters=int(200*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                activation="linear",
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="output_1x1_top_"+str(index))(concat_layer)
    
    
    if hyperparameters["residual"]:
        x = concatenate([previous_layer,x],axis=-1)
        x = Add()(x)#+previous_layer
    
    if hyperparameters["batch_normalization"]:
        bn = BatchNormalization(name="stem_bn")(x)
        activ = Activation(hyperparameters["activation"], name="activ"+str(index))(bn)
    else:
        activ = Activation(hyperparameters["activation"], name="activ"+str(index))(x)
    
    if hyperparameters["avgpool"]:
        output_layer = AveragePooling2D(pool_size=hyperparameters["strides"][0][index], name="deep_wide_top_out"+str(index))(activ)
    else:
        output_layer = MaxPooling2D(pool_size=hyperparameters["strides"][0][index], name="deep_wide_top_out"+str(index))(activ)
        
    return output_layer

def deep_wide_mid(previous_layer, filters, index, hyperparameters):
    filter_factor=filters/300
    
    #first column
    column_one = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_one_mid_"+str(index))(previous_layer)
    
    #second column
    column_two = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.1_mid_"+str(index))(previous_layer)
    column_two = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[3,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.2_mid_"+str(index))(column_two)
    column_two = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[1,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.3_mid_"+str(index))(column_two)
    
    #third column
    column_three = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.1_mid_"+str(index))(previous_layer)
    column_three = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[5,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.2_mid_"+str(index)) (column_three)    
    column_three = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[1,5],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.3_mid_"+str(index)) (column_three)
    
    #fourth column
    column_four = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_four.1_mid_"+str(index))(previous_layer)
    column_four = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[7,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_four.2_mid_"+str(index)) (column_four)    
    column_four = Conv2D(
                filters=int(30*filter_factor),
                kernel_size=[1,7],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_four.3_mid_"+str(index)) (column_four)
    
    #concatenation of feature maps
    concat_layer = concatenate([column_one,column_two,column_three,column_four],axis=-1)
    
    x = Conv2D(
                filters=int(300*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                activation="linear",
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="output_1x1_mid_"+str(index))(concat_layer)
    
    
    if hyperparameters["residual"]:
        x = concatenate([previous_layer,x],axis=-1)
        x = Add()(x)#+previous_layer
    
    if hyperparameters["batch_normalization"]:
        bn = BatchNormalization(name="stem_bn")(x)
        activ = Activation(hyperparameters["activation"], name="activ_mid_"+str(index))(bn)
    else:
        activ = Activation(hyperparameters["activation"], name="activ_mid_"+str(index))(x)
    
    if hyperparameters["avgpool"]:
        output_layer = AveragePooling2D(pool_size=hyperparameters["strides"][1][index], name="deep_wide_mid_out"+str(index))(activ)
    else:
        output_layer = MaxPooling2D(pool_size=hyperparameters["strides"][1][index], name="deep_wide_mid_out"+str(index))(activ)
        
    return output_layer

def deep_wide_low(previous_layer, filters, index, hyperparameters):
    filter_factor=filters/800
    
    #first column
    column_one = Conv2D(
                filters=int(80*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_one_low_"+str(index))(previous_layer)
    
    #second column
    column_two = Conv2D(
                filters=int(80*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.1_low_"+str(index))(previous_layer)
    column_two = Conv2D(
                filters=int(80*filter_factor),
                kernel_size=[3,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.2_low_"+str(index))(column_two)
    column_two = Conv2D(
                filters=int(80*filter_factor),
                kernel_size=[1,3],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_two.3_low_"+str(index))(column_two)
    
    #third column
    column_three = Conv2D(
                filters=int(80*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.1_low_"+str(index))(previous_layer)
    column_three = Conv2D(
                filters=int(80*filter_factor),
                kernel_size=[5,1],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.2_low_"+str(index)) (column_three)    
    column_three = Conv2D(
                filters=int(80*filter_factor),
                kernel_size=[1,5],
                strides=[1,1],
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="column_three.3_low_"+str(index)) (column_three)
    
    #concatenation of feature maps
    concat_layer = concatenate([column_one,column_two,column_three],axis=-1)
    x = Conv2D(
                filters=int(800*filter_factor),
                kernel_size=[1,1],
                strides=[1,1],
                activation="linear",
                padding=hyperparameters["padding"],
                kernel_regularizer=regularizers.l2(hyperparameters["kernel_regularizer"]),
                name="output_1x1_low_"+str(index))(concat_layer)
    
    
    if hyperparameters["residual"]:
        x = concatenate([previous_layer,x],axis=-1)
        x = Add()(x)#+previous_layer
                    
    
    if hyperparameters["batch_normalization"]:
        bn = BatchNormalization(name="stem_bn")(x)
        activ = Activation(hyperparameters["activation"], name="activ_low_"+str(index))(bn)
    else:
        activ = Activation(hyperparameters["activation"], name="activ_low_"+str(index))(x)
    
    if hyperparameters["avgpool"]:
        output_layer = AveragePooling2D(pool_size=hyperparameters["strides"][2][index], name="deep_wide_low_out"+str(index))(activ)
    else:
        output_layer = MaxPooling2D(pool_size=hyperparameters["strides"][2][index], name="deep_wide_low_out"+str(index))(activ)
        
    return output_layer

class DWNet:
    def __init__(self):
        self.data = []
    #test model
    def get_model(input_layer, hyperparameters):
        #deep-wide top module
        dw_top = deep_wide_top(input_layer,hyperparameters["filters"][0][0], 0, hyperparameters)
        for index in range(1,len(hyperparameters["filters"][0])):
            dw_top = deep_wide_top(dw_top,hyperparameters["filters"][0][index], index, hyperparameters)
        
        #deep-wide mid module
        dw_mid = deep_wide_mid(dw_top,hyperparameters["filters"][1][0], 0, hyperparameters)
        for index in range(1,len(hyperparameters["filters"][1])):
            dw_mid = deep_wide_mid(dw_mid,hyperparameters["filters"][1][index], index, hyperparameters)
        
        #deep-wide low module
        dw_low = deep_wide_low(dw_mid,hyperparameters["filters"][2][0], 0, hyperparameters)
        for index in range(1,len(hyperparameters["filters"][2])):
            dw_low = deep_wide_low(dw_low,hyperparameters["filters"][2][index], index, hyperparameters)
        
        shape = dw_low.shape
        print(shape)
#        print(type(dw_low))
        #flatten feature maps
#        flatten_layer = Flatten()(dw_low)
        
        dim = shape[1]*shape[2]*shape[3]
        print(dim)
        print(dim.value)
#        print(type(dim))
#        print(type(dim.value))
#        flatten_layer = Flatten()(dw_low)
        flatten_layer = Reshape((dim.value,))(dw_low)
        shape_before_flatten = dw_low.shape.as_list()[1:]# [1:] to skip none
        print(shape_before_flatten)
#        flatten_layer = np.prod(shape_before_flatten)
        print(flatten_layer.shape)
#        print(type(flatten_layer))
        
        #Neural Network with Dropout
        nn = Dense(units=hyperparameters["neurons"], activation=hyperparameters["activation"], name="hidden_layer")(flatten_layer)
        dropout_layer = Dropout(rate=hyperparameters["dropout"],name="drop")(nn)
        
        #Classification Layer
        final_output = Dense(units=hyperparameters["num_classes"], activation=hyperparameters["classifier"], name=hyperparameters["classifier"]+"_layer")(dropout_layer)
        return final_output