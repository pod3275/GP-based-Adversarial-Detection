# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:08:46 2020

@author: lawle
"""

import os
import keras
import numpy as np
from tqdm import tqdm
from keras.models import load_model


def gray2rgb(img):
    w, h, _ = img.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = img[:,:,0]
    ret[:, :, 1] = img[:,:,0]
    ret[:, :, 2] = img[:,:,0]
    return ret


def load_img(dataset, attack):
    location = os.getcwd() + '/Data/' + dataset + '/' + attack
    file_list = os.listdir(location)
        
    adv_img = []
    clean_img = []
    total = 0
    
    for i in tqdm(file_list):
        path_dir = location + '/' + i
        img_list = os.listdir(path_dir)
        img_list.sort()

        for file_name in img_list:
            img = np.load(path_dir + '/' + file_name)
            img_plx1 = np.asarray(img, dtype='float32')
        
            if 'adv' in file_name:
                adv_img.append(img_plx1)
                            
            if 'origin' in file_name:
                clean_img.append(img_plx1)
                    
            total = total + 1
                
    adv_img = np.asarray(adv_img, dtype='float32')
    clean_img = np.asarray(clean_img, dtype='float32')
                                                  
    return adv_img, clean_img, total


def l2_perturb(dataset, attack):
    f = open(dataset + '_' + attack + '_L2perturb.txt', 'w')
    f.write(dataset + '_' + attack + '_L2 perturb\n')
    print(dataset + '_' + attack + '_L2 perturb\n')
    
    adv, clean, count = load_img(dataset, attack)
    ### adv = adv*255
    noise = adv-clean
    
    avg_l2_norm = 0
    for leng in range(len(noise)):
        ka = noise[leng]
        ka = ka.ravel() / 255
        avg_l2_norm = avg_l2_norm + np.linalg.norm(ka)
        
    avg_l2_norm = avg_l2_norm / len(noise)
    f.write('Total data : ' + str(count) + '\n')
    f.write('Average L2 Norm of Noise : ' + str(avg_l2_norm) + '\n\n')
    print('Average L2 Norm of Noise :', avg_l2_norm)
    
    f.close()
    

def check_image_label(dataset, attack, file_idx=8):
    keras.backend.set_learning_phase(0)
    
    if dataset == 'CIFAR10': kmodel = load_model('saved_models/cifar10_ResNet32v1_model.h5')
    elif dataset == 'MNIST': kmodel = load_model('saved_models/MNIST_model.h5')
    
    origin_img =[]
    adv_img=[]
    
    for i in tqdm(range(10), desc="%d th class checking label..."):
        a = np.load(os.getcwd() + f'/Data/{dataset}/{attack}/{str(i)}/{str(file_idx)}_origin.npy')
        b = np.load(os.getcwd() + f'/Data/{dataset}/{attack}/{str(i)}/{str(file_idx)}_adv.npy')
        
        origin_label=np.argmax(kmodel.predict(a[np.newaxis,:,:,:]/255), axis=1)
        adv_label=np.argmax(kmodel.predict(b[np.newaxis,:,:,:]/255), axis=1)
        
        origin_img.append(origin_label[0])
        adv_img.append(adv_label[0])
        
        # To see the image
        #plt.imshow(a/255)
        #plt.imshow(b/255)
    
    print("Original data label:", origin_img)
    print("Adversarial example label:", adv_img)