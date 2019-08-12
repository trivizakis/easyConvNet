#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""
import numpy as np
import pickle as pkl
#import nrrd
#import SimpleITK as sitk
import cv2
import os
        
def centering_volume(mri,roi,labels,pid,hypes):
    f_mri=[]
    f_labels=[]
    f_pids=[]
    for i in range(0, len(mri)):
        v_length = len(mri[i])
        if v_length>hypes["input_shape"][0]-1:
            head=0
            tail=0
            tmp=0
            tmp2=0
            for k in range(0,len(roi[i])):
                if any(e==1 for j in range(0,len(roi[i][k])) for e in roi[i][k][j]) is True:                   
                    head=k
                    break
            diff = v_length-hypes["input_shape"][0]
            if diff < head:
                head=diff
                tmp = np.reshape(np.stack(mri[i][head:]),(-1,hypes["input_shape"][1],hypes["input_shape"][2],hypes["input_shape"][3]))#(vol,x,y,ch)
            else:
                tail = v_length - (diff-head)
                tmp = np.reshape(np.stack(mri[i][head:tail]),(-1,hypes["input_shape"][1],hypes["input_shape"][2],hypes["input_shape"][3]))#(vol,x,y,ch)
            f_mri.append(tmp)
            f_labels.append(labels[i])                
            f_pids.append(pid[i])
        else:
            diff = hypes["input_shape"][0]-v_length
            tmp = np.zeros((diff,hypes["input_shape"][1],hypes["input_shape"][2],hypes["input_shape"][3]))#(vol,x,y,ch)
            tmp2 = np.reshape(np.stack(mri[i]),(-1,hypes["input_shape"][1],hypes["input_shape"][2],hypes["input_shape"][3]))#(vol,x,y,ch)
            f_mri.append(np.concatenate((tmp2,tmp),axis=0))
            f_labels.append(labels[i])                
            f_pids.append(pid[i])                
    return f_mri, f_labels, f_pids

def pad2d_image(image,desired_shape):
    shape = image.shape
#    print("Desired: "+str(desired_shape)+" Original: "+str(shape))
    xDiff = desired_shape[0]-shape[0]
    xLeft = xDiff//2
    xRight = xDiff-xLeft
        
    yDiff = desired_shape[1]-shape[1]
    yLeft = yDiff//2
    yRight = yDiff-yLeft
    return np.reshape(np.pad(image,((xLeft,xRight),(yLeft,yRight)),mode='constant',constant_values=(0,0)),tuple(desired_shape))
    
def dataset_loader(file_path,pid,suffix):
    """
       loads images in a list of lists
    
       file_path: str - path of dataset
       pid: list - patient ID
       suffix: str - file name suffix
   """
    dataset=[]
    for j in range (0,len(pid)):
#        data, options = nrrd.read(file_path + pid[j]  + suffix)
        data = sitk.ReadImage(file_path + pid[j]  + suffix)
        data = np.transpose(data,(2,0,1))#first number of slices
        
        #fix rendering errors
        data = np.nan_to_num(data)
        data[data<0]=0        
        
        dataset.append(data)
    return dataset

def image_normalization(X, hypes): 
    print("Normalizing...")          
    if hypes["image_normalization"] == "01":
        x_min = np.amin(X)
        x_max = np.amax(X)
        X = (X - x_min)/(x_max-x_min)
    elif hypes["image_normalization"] == "-11":
        std = np.std(np.array(X).ravel())
        mean = np.mean(np.array(X).ravel())
        X = (X - mean)/std
    return X.astype("float16")

class DataConverter():
    'Loads data'
    def __init__(self, hypes):
        'Initialization'
        self.hypes = hypes
    def load_npy_from_hdd(self, pids, labels):
        # Initialization
        dim = self.hypes["input_shape"][0:(len(self.hypes["input_shape"])-1)] # [vol,x,y]
        n_channels = self.hypes["input_shape"][(len(self.hypes["input_shape"])-1)] # color channels
        X = np.empty((len(pids), *dim, n_channels))
        y = np.empty((len(pids)), dtype=int)
        #load selected data from hdd
        for i, ID in enumerate(pids):
            # Store sample
            X[i,] = np.load(self.hypes["dataset_dir"] + ID + '.npy')       
            
            # Store class
            y[i] = labels[ID]
        return X, list(y)
        
    def save_augmented_samples(self, augmented_samples, augmented_labels,pids_tr,labels_tr):
        #for offline_augmentation
        print("Saving Augmented Samples...")     
        #saving augmented labels     
#        if not os.path.exists(self.hypes["dataset_dir"]+"labels_original.npy"):
#            labels = np.load(self.hypes["dataset_dir"] + 'labels.npy')
#            np.save(self.hypes["dataset_dir"]+"labels_original.npy", labels)
#        else:
#            labels = np.load(self.hypes["dataset_dir"] + 'labels_original.npy')
#        labels = np.load(self.hypes["dataset_dir"] + 'labels.npy')
#        tr_labels = np.concatenate((labels,augmented_labels))
#        np.save(self.hypes["dataset_dir"]+"labels",f_labels)
        
        #saving augmented pids
        augm_pids=[]
#        if not os.path.exists(self.hypes["dataset_dir"]+"pids_original.npy"):
#            pids = np.load(self.hypes["dataset_dir"] + 'pids.npy')
#            np.save(self.hypes["dataset_dir"]+"pids_original.npy", labels)
#        else:
#            pids = np.load(self.hypes["dataset_dir"] + 'pids_original.npy')
        #create augmented images pids
        for index, label in enumerate(augmented_labels):
            augm_pids.append("augm_p"+str(index+1))
        augmented_pids=np.array(augm_pids)
        tr_pids = np.concatenate((pids_tr,augmented_pids))
        tr_labels = np.concatenate((labels_tr,augmented_labels))
#        np.save(self.hypes["dataset_dir"]+"pids",f_pids)
        
        #open pkl with dict classes[pids]
#        if not os.path.exists(self.hypes["dataset_dir"]+"labels_original.pkl"):
#            with open(self.hypes["dataset_dir"]+"labels.pkl", "rb") as file:        
#                classes=pkl.load(file)
#            with open(self.hypes["dataset_dir"]+"labels_original.pkl", "wb") as file:
#                pkl.dump(classes, file, protocol=pkl.HIGHEST_PROTOCOL)
#        else:
#            with open(self.hypes["dataset_dir"]+"labels_original.pkl", "rb") as file:        
#                classes=pkl.load(file)
#        with open(self.hypes["dataset_dir"]+"labels.pkl", "rb") as file:
#            classes=pkl.load(file)
        tr_classes={}
        for index, label in enumerate(tr_labels):
            tr_classes[tr_pids[index]]=label        
#        with open(self.hypes["dataset_dir"]+"labels.pkl", "wb") as file:
#            pkl.dump(classes, file, protocol=pkl.HIGHEST_PROTOCOL)
        
        #saving augmented images
        for index , image in enumerate(augmented_samples):
            np.save(self.hypes["dataset_dir"]+augmented_pids[index],image)
            
        return tr_pids, tr_classes
        
    def convert_nrrd_to_npy(self):
        #open pkl with classes and pids
        with open(self.hypes["dataset_dir"]+"labels.pkl", "rb") as file:        
            classes=pkl.load(file)            
        labels = list(classes.values())
        pid = list(classes.keys())
        
        print("Loading from HDD...")
        mri = dataset_loader(self.hypes["dataset_dir"]+"mri/",pid,self.hypes["mri_file_suffix"]) #pid
        roi = dataset_loader(self.hypes["dataset_dir"]+"roi/",pid,self.hypes["roi_file_suffix"])                

        print("Centering...")
        f_data, f_labels, f_pids = centering_volume(mri,roi,labels,pid, self.hypes)
        f_data = image_normalization(f_data, self.hypes)                                
        
        print("Saving...")
        for index , image in enumerate(f_data):
            np.save(self.hypes["dataset_dir"]+f_pids[index],image)
            
        np.save(self.hypes["dataset_dir"]+"labels",f_labels)
        np.save(self.hypes["dataset_dir"]+"pids",f_pids)

    def convert_png_to_npy(self):
        #images are separated in folders by class
        label=0
        pid=0
        images=[]
        labels={}
        pids=[]
        for subpath in os.listdir(self.hypes["dataset_dir"]):
            for filename in os.listdir(self.hypes["dataset_dir"]+subpath):
                #constant shape images
                images.append(np.reshape(cv2.imread(self.hypes["dataset_dir"]+subpath+"/"+filename,0),tuple(self.hypes["input_shape"])))
                #high-res images with variable shape
#                images.append(pad2d_image(cv2.imread(self.hypes["dataset_dir"]+subpath+"/"+filename,0),self.hypes["input_shape"]))
                pids.append("P"+str(pid))
                labels["P"+str(pid)]=label
                pid+=1
            label+=1
            
        images = image_normalization(images, self.hypes) 
        np.save(self.hypes["dataset_dir"]+"pids",pids)
        
        with open(self.hypes["dataset_dir"]+"labels.pkl", "wb") as file:
            pkl.dump(labels, file, protocol=pkl.HIGHEST_PROTOCOL)
        label_values = list(labels.values())
        np.save(self.hypes["dataset_dir"]+"labels", label_values)
        
        for index , image in enumerate(images):
            np.save(self.hypes["dataset_dir"]+pids[index],image)
    
    def convert_png_to_npy_with_dict(self,pids,label_values):
        #images are read with dictionary
        images=[]
        image_dir_path=os.path.dirname(os.path.dirname(self.hypes["dataset_dir"]))+"/"
        print(image_dir_path)
        for filename in pids:
            #constant shape images
            images.append(np.reshape(cv2.imread(image_dir_path+filename,0),tuple(self.hypes["input_shape"])))
            #high-res images with variable shape
#                images.append(pad2d_image(cv2.imread(self.hypes["dataset_dir"]+subpath+"/"+filename,0),self.hypes["input_shape"]))
        
        images = image_normalization(images, self.hypes) 
        np.save(self.hypes["dataset_dir"]+"pids",pids)
        np.save(self.hypes["dataset_dir"]+"labels", label_values)
        
        for index , image in enumerate(images):
            np.save(self.hypes["dataset_dir"]+pids[index],image)