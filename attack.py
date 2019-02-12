# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:04:06 2018

@author: lawle
"""

import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.datasets import mnist, cifar10
from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, SaliencyMapMethod, FastGradientMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt


data_name = 'CIFAR10'
attack_name = 'DeepFool'
save_loc = '/Data/' + data_name + '/' + attack_name


f = open(data_name + '_' + attack_name + '_Classifier.txt', 'w')
f.write(data_name + '_' + attack_name + '_Classifier\n')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
keras.backend.set_session(sess)

def save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc_add):
    count_class_save = np.zeros(10)
    count_class_save = count_class_save.astype('int32')
    origin_count=0
    adv_count=0
    adv_result = []
    buffer = []
    buffer2 = []
    # i=0
    for i in tqdm(range(len(x_test))):
        x_img_input = x_test[i][np.newaxis,:,:,:]
        y_input=[]
        y_input.append(preds1[i])

        if attack_name == 'BIM':
            y_input = keras.utils.to_categorical(y_input, 10)

        adv_x_eval = adv_x.eval(session = sess, feed_dict={x:x_img_input/255, y:y_input})
        adv_x_eval = adv_x_eval*255
        preds2 = np.argmax(kmodel.predict(adv_x_eval/255), axis=1)
        
        '''
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(adv_x_eval[0] / 255)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')
        '''
        
        adv_result.append(preds2[0])
        
        if not(os.path.isdir(os.getcwd() + save_loc_add + '/' + str(y_test[i][0]) + '/')):
                os.makedirs(os.path.join(os.getcwd() + save_loc_add + '/' + str(y_test[i][0]) + '/'))
                             
        if preds1[i][0] == y_test[i][0] and preds2[0] != y_test[i][0]:
            np.save(os.getcwd() + save_loc_add + '/' + str(y_test[i][0]) + '/' 
                    + str(count_class_save[y_test[i][0]])
                    + '_origin.npy', x_test[i])
            
            np.save(os.getcwd() + save_loc_add + '/' + str(y_test[i][0]) + '/'
                    + str(count_class_save[y_test[i][0]])
                    + '_adv.npy', adv_x_eval[0])

            count_class_save[y_test[i][0]] = count_class_save[y_test[i][0]] + 1
        
        if preds1[i][0] == y_test[i][0]:
            origin_count +=1
            
        if preds2[0] == y_test[i][0]:
            adv_count +=1
        
        buffer.append(preds2[0])
        buffer2.append(preds1[i][0])
            
    f.write('Count_Class_Save: ' + str(count_class_save) + '\n')
    f.write('Origin Accuracy: ' + str(origin_count/len(x_test) * 100) + '%\n')
    f.write('Adv Accuracy: ' + str(adv_count/len(x_test) * 100) + '%\n\n')
    #f.write('Origin class: ' + str(buffer2[0:100]) + '\n')
    #f.write('Adv class: ' + str(buffer[0:100]) + '\n')


if data_name == 'MNIST':
    img_rows, img_cols = 28, 28
    num_classes = 10
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    # p2 = np.argmax(kmodel.predict(x_test/255), axis=1)
    
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]
    
    keras.backend.set_learning_phase(0)
    kmodel = load_model('saved_models/MNIST_model.h5')
    wrap = KerasModelWrapper(kmodel)

elif data_name == 'CIFAR10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]
    
    keras.backend.set_learning_phase(0)
    kmodel = load_model('saved_models/cifar10_ResNet32v1_model.h5')
    wrap = KerasModelWrapper(kmodel)


#Placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

if attack_name == 'BIM':
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
preds1 = np.argmax(kmodel.predict(x_test/255), axis=1)
preds1 = preds1.reshape(preds1.shape[0], 1)


#Attack
if attack_name == 'FGSM':
    attack = FastGradientMethod(wrap, sess=sess)
    eps = 0
    
    if data_name == 'MNIST':
        for ep in range(5):
            eps = eps + 0.1
            params = {'eps': eps,
                      'clip_min': 0.,
                      'clip_max': 1.}
            adv_x = attack.generate(x, **params)
            
            print('Epsilon:', eps)
            f.write('Epsilon: ' + str(eps) + '\n')
                        
            save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc + '_e' + str(eps))
            
    if data_name == 'CIFAR10':
        for ep in range(10):
            eps = eps + 1
            params = {'eps': eps/255,
                      'clip_min': 0.,
                      'clip_max': 1.}
            adv_x = attack.generate(x, **params)
            
            print('Epsilon:', eps)
            f.write('Epsilon: ' + str(eps) + '\n')
                        
            save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc + '_e' + str(eps))
           
            
#attack_JSMA
if attack_name == 'JSMA':
    attack = SaliencyMapMethod(wrap, sess=sess)
    params = {'clip_min': 0., 'clip_max': 1.}
    adv_x = attack.generate(x, **params)
    save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc)


#attack_BIM
if attack_name == 'BIM':
    attack = BasicIterativeMethod(wrap, sess=sess)
    eps = 0
    
    if data_name == 'MNIST':
        for ep in range(5):
            eps = eps + 0.1
            params = {'eps': eps,
                      'eps_iter': eps/10,
                      'nb_iter': 10,
                      'y': y,
                      'clip_min': 0.,
                      'clip_max': 1.}
            adv_x = attack.generate(x, **params)
            adv_x = tf.stop_gradient(adv_x)
            
            print('Epsilon:', eps)
            f.write('Epsilon: ' + str(eps) + '\n')
                        
            save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc + '_e' + str(eps))
            
    if data_name == 'CIFAR10':
        for ep in range(10):
            eps = eps + 1
            params = {'eps': eps/255,
                      'eps_iter': eps/255/10,
                      'nb_iter': 10,
                      'y': y,
                      'clip_min': 0.,
                      'clip_max': 1.}
            adv_x = attack.generate(x, **params)
            adv_x = tf.stop_gradient(adv_x)
            
            print('Epsilon:', eps)
            f.write('Epsilon: ' + str(eps) + '\n')
                 
            save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc + '_e' + str(eps))


#attack_DeepFool
if attack_name == 'DeepFool':
    attack = DeepFool(wrap, sess=sess)
    params = {'nb_candidate': 10,
              'max_iter': 100,
              'clip_min': 0.,
              'clip_max': 1.}
    adv_x = attack.generate(x, **params)
    save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc)


#attack_CW
if attack_name == 'CW':
    attack = CarliniWagnerL2(wrap, sess=sess)
    params = {'batch_size':1,
              'max_iterations':1000,
              'binary_search_steps':9,
              'initial_const':1e-3,
              'learning_rate':5e-3,
              'clip_min': 0.,
              'clip_max': 1.}
    adv_x = attack.generate(x, **params)
    save_images(kmodel, adv_x, x_test, y_test, preds1, save_loc)

f.close()
