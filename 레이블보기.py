# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:05:53 2019

@author: lawle
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import keras

data_name = 'CIFAR10'
attack_name = 'DeepFool'

keras.backend.set_learning_phase(0)
if data_name == 'CIFAR10': kmodel = load_model('saved_models/cifar10_ResNet32v1_model.h5')
elif data_name == 'MNIST': kmodel = load_model('saved_models/MNIST_model.h5')

origin =[]
adver=[]

for i in range(10):
    
    a = np.load(os.getcwd() + '/Data/'+data_name+'/'+attack_name+'/'+str(i)+'/8_origin.npy')
    b = np.load(os.getcwd() + '/Data/'+data_name+'/'+attack_name+'/'+str(i)+'/8_adv.npy')
    
    #plt.imshow(a)
    #plt.imshow(b)
    
    orig=np.argmax(kmodel.predict(a[np.newaxis,:,:,:]/255), axis=1)
    adv=np.argmax(kmodel.predict(b[np.newaxis,:,:,:]/255), axis=1)
    
    origin.append(orig[0])
    adver.append(adv[0])

print(origin)
print(adver)