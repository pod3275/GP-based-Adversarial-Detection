# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:05:53 2019

@author: lawle
"""

import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import keras

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'CIFAR10', 'target dataset of the trained model')
flags.DEFINE_string('attack', 'DeepFool', 'attack to check the label')

keras.backend.set_learning_phase(0)

if FLAGS.dataset == 'CIFAR10': kmodel = load_model('saved_models/cifar10_ResNet32v1_model.h5')
elif FLAGS.attack == 'MNIST': kmodel = load_model('saved_models/MNIST_model.h5')

origin =[]
adver=[]

for i in range(10):
    print("%d th class checking label..." % i)
    a = np.load(os.getcwd() + '/Data/'+FLAGS.dataset+'/'+FLAGS.attack+'/'+str(i)+'/8_origin.npy')
    b = np.load(os.getcwd() + '/Data/'+FLAGS.dataset+'/'+FLAGS.attack+'/'+str(i)+'/8_adv.npy')
    
    # To see the image
    #plt.imshow(a/255)
    #plt.imshow(b/255)
    
    orig=np.argmax(kmodel.predict(a[np.newaxis,:,:,:]/255), axis=1)
    adv=np.argmax(kmodel.predict(b[np.newaxis,:,:,:]/255), axis=1)
    
    origin.append(orig[0])
    adver.append(adv[0])

print("Original data label:", origin)
print("Adversarial example label:", adver)