# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:10:17 2018

@author: lawle
"""

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'CIFAR10', 'target dataset of the trained model')
flags.DEFINE_string('attack', 'BIM_e9', 'attack to check the label')


f = open(FLAGS.dataset + '_' + FLAGS.attack + '_L2size.txt', 'w')
f.write(FLAGS.dataset + '_' + FLAGS.attack + '_L2 size.txt\n')
print(FLAGS.dataset + '_' + FLAGS.attack + '_L2 size\n')


def gray2rgb(img):
    w, h, _ = img.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = img[:,:,0]
    ret[:, :, 1] = img[:,:,0]
    ret[:, :, 2] = img[:,:,0]
    return ret


def load_img():
    location = os.getcwd() + '/Data/' + FLAGS.dataset + '/' + FLAGS.attack
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


adv, clean, count = load_img()
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

