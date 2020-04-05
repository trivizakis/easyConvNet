#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from math import floor
from sklearn.utils import shuffle

#3D
def get_aug_data_volume(volume,dict_params):
    #volume: volume of images
    volume=list(volume) # eg: from array:(30,256,256) to list:(30)(256,256)
    elastic =[]
    mirroring=[]    
    rotated=[]
    
    #rotate image
    for i in range(0,len(dict_params["degree"])):
        rotated.append(np.stack(rotation(volume,dict_params["degree"][i])))
    rotated = np.stack(rotated)
#    print("rotated: "+str(rotated.shape))
    
    #up-down
    flippedud = flipping(volume,vertical=True)
    flippedud = np.stack(flippedud).reshape(-1,dict_params["input_shape"][0],dict_params["input_shape"][1],dict_params["input_shape"][2],dict_params["input_shape"][3])
#    print("Flip1: "+str(flippedud.shape))
    #left_right
    flippedlr = flipping(volume,vertical=False)
    flippedlr=np.stack(flippedlr).reshape(-1,dict_params["input_shape"][0],dict_params["input_shape"][1],dict_params["input_shape"][2],dict_params["input_shape"][3])
#    print("Flip2: "+str(flippedlr.shape))
    
    #elastic transformations
    for i in range(0,len(dict_params["alpha"])):
        tmp_v=[]
        for j in range(0,len(volume)): #same tansformation hypes per volume
#            tmp_v.append(np.reshape(elastic_transform(volume[j],alpha=dict_params["input_shape"][1]*dict_params["alpha"][i],sigma=dict_params["input_shape"][1]*dict_params["sigma"][i],alpha_affine=dict_params["input_shape"][1]*dict_params["alpha_affine"][i]),(dict_params["input_shape"][1],dict_params["input_shape"][2])))
#            print(volume[j].shape)
#            print(str((dict_params["input_shape"][1],dict_params["input_shape"][2],dict_params["input_shape"][3])))
            tmp_v.append(np.reshape(elastic_transform(np.reshape(volume[j],(dict_params["input_shape"][1],dict_params["input_shape"][2],dict_params["input_shape"][3])),alpha=dict_params["input_shape"][1]*dict_params["alpha"][i],sigma=dict_params["input_shape"][1]*dict_params["sigma"][i],alpha_affine=dict_params["input_shape"][1]*dict_params["alpha_affine"][i]),(dict_params["input_shape"][1],dict_params["input_shape"][2],dict_params["input_shape"][3])))
        elastic.append(np.stack(tmp_v))
    elastic=np.stack(elastic).reshape(-1,dict_params["input_shape"][0],dict_params["input_shape"][1],dict_params["input_shape"][2],dict_params["input_shape"][3])
#    print("Elastic list:"+str(elastic.shape))
    
    #mirroring
    for j in range(0,len(dict_params["mirror_x"])):
        tmp=[]
        mirror_tmp=[]
        for i in range(0,len(volume)):
            tmp=np.asarray(mirror([mirror(sublist) for sublist in volume[i]]))
            mirror_tmp.append(tmp[floor(dict_params["input_shape"][1]*dict_params["mirror_x"][j]):dict_params["input_shape"][1]+floor(dict_params["input_shape"][1]*dict_params["mirror_x"][j]),floor(dict_params["input_shape"][2]*dict_params["mirror_y"][j]):dict_params["input_shape"][2]+floor(dict_params["input_shape"][2]*dict_params["mirror_y"][j])])
        mirroring.append(np.stack(mirror_tmp))
    mirroring=np.stack(mirroring)
#    print("Mirroring list:"+str(mirroring.shape))
    return np.concatenate((elastic, mirroring, rotated, flippedud, flippedlr),axis=0)

#2D
def get_aug_data(images,labels,dict_params):
#    images: list of images
#    labels: list of labels
    augmented_labels=[]
    elastic =[]
    mirroring=[]
    rotated=[]
    
    #rotate images
    for i in range(0,len(dict_params["degree"])):
        rotated+=rotation(images,dict_params["degree"][i])
        augmented_labels+=labels
    
    #up-down     
    flippedud = flipping(images,vertical=True)
    augmented_labels+=labels
    #left_right
    flippedlr = flipping(images,vertical=False)
    augmented_labels+=labels
    
    #elastic transformations
    for i in range(0,len(dict_params["alpha"])):
        tmp_im=[]
        for j in range(0,len(images)):
            #reshape
            tmp_im.append(elastic_transform(images[j],alpha=dict_params["input_shape"][0]*dict_params["alpha"][i],sigma=dict_params["input_shape"][0]*dict_params["sigma"][i],alpha_affine=dict_params["input_shape"][0]*dict_params["alpha_affine"][i]))
            augmented_labels.append(labels[j])
        elastic+=tmp_im
        
    #mirroring
    for j in range(0,len(dict_params["mirror_x"])):
        tmp=[]
        mirror_tmp=[]
        for i in range(0,len(images)):
            tmp=np.asarray(mirror([mirror(sublist) for sublist in images[i]]))
            mirror_tmp.append(tmp[floor(dict_params["input_shape"][0]*dict_params["mirror_x"][j]):dict_params["input_shape"][0]+floor(dict_params["input_shape"][0]*dict_params["mirror_x"][j]),floor(dict_params["input_shape"][1]*dict_params["mirror_y"][j]):dict_params["input_shape"][1]+floor(dict_params["input_shape"][1]*dict_params["mirror_y"][j])])
            augmented_labels.append(labels[i])
        mirroring+=mirror_tmp
#    print("What's the final shape")
#    print(images.shape)
#    print(len(elastic))
#    print(elastic[0].shape)
#    print(len(mirroring))
#    print(mirroring[0].shape)
#    print(len(rotated))
#    print(rotated[0].shape)
#    print(len(flippedud))
#    print(flippedud[0].shape)
#    print(len(flippedlr))
#    print(flippedlr[0].shape)
    return np.stack(elastic + mirroring + rotated + flippedud + flippedlr), augmented_labels

def mirror(seq):
    output = list(seq[::-1])
    output.extend(seq[1:])
    return output

def flipping(image,vertical=True):
    flipped=[]
    for i in range(0,len(image)):
        if vertical == False:
            flipped.append(np.fliplr(image[i]))#left to right flip
        else:
            flipped.append(np.flipud(image[i]))#up to down flip   
    return flipped

def rotation(image,degree=90):    
    rotated =[]
    for i in range(0,len(image)):
        if degree >= 90:
            times = int(degree/90)
            rotated.append(np.rot90(image[i],k=times))
        else:
            print("Not less than 90o!")
            
    return rotated

# Function to transform image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
#    print("What's the image shape")
#    print(shape)
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant",cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant",cval=0) * alpha
#    dz = np.zeros_like(dx)
    
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
#    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def get_aug_3d_data(data, labels, hyperparameters):
    aug_data=[]
    aug_labels=[]
    for i in range(0,len(data)):
        tmp = get_aug_data_volume(data[i],hyperparameters) #returns augmented volumes
        for j in range(0,len(tmp)):
            aug_labels.append(labels[i])
        aug_data.append(tmp)
    aug_data=np.stack(aug_data).reshape(-1,hyperparameters["input_shape"][0],hyperparameters["input_shape"][1],hyperparameters["input_shape"][2],hyperparameters["input_shape"][3])
    return aug_data, aug_labels

class DataAugmentation:
        def apply_augmentation(data,labels,hyperparameters):
            """
                data: list of images or volumes
                labels: list of labels
                hyperparameters : dictionary of hypes
            """
            if hyperparameters["offline_augmentation"] == True:
                print("Augmenting...")
                
            if hyperparameters["conv_type"] == "2d":
                aug_data, aug_labels = get_aug_data(data, labels, hyperparameters)
            elif hyperparameters["conv_type"] == "3d":
                aug_data, aug_labels = get_aug_3d_data(data, labels, hyperparameters)
            else:
                print("Not Supported: " + hyperparameters["conv_type"]) 
                
            aug_data, aug_labels = shuffle(aug_data, aug_labels)
            return np.array(aug_data),np.array(aug_labels)