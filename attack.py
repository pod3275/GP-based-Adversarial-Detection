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

from keras.datasets import mnist
from keras.datasets import cifar10
from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, SaliencyMapMethod, FastGradientMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from keras.models import load_model
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
keras.backend.set_session(sess)

data_name = 'MNIST'
attack_name = 'FGSM'


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
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                      nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))


if attack_name == 'FGSM':
    attack = FastGradientMethod(wrap, sess=sess)
    eps = 0
    batch_size = 128
    eval_par = {'batch_size': batch_size}
    
    preds1 = np.argmax(kmodel.predict(x_test/255), axis=1)
    preds1 = preds1.reshape(preds1.shape[0], 1)
    
    f = open(data_name + '_' + attack_name + '_Classifier.txt', 'w')
    
    if data_name == 'MNIST': 
        for ep in tqdm(range(5)):
            eps = eps + 0.1
            params = {'eps': eps,
                      'clip_min': 0.,
                      'clip_max': 1.}
            adv_x = attack.generate(x, **params)
            
            count_class_save = np.zeros(10)
            count_class_save = count_class_save.astype('int32')
            origin_count=0
            adv_count=0
            
            adv_result = []
        
            print('Epsilon:', eps)
            
            for i in tqdm(range(len(x_test))):
                x_img_input = x_test[i][np.newaxis,:,:,:]
                
                y_input=[]
                y_input.append(preds1[i])
                y_input = keras.utils.to_categorical(y_input, 10)
                
                adv_x_eval = adv_x.eval(session = sess, feed_dict={x:x_img_input/255, y:y_input})
                adv_x_eval = adv_x_eval*255
                preds2 = np.argmax(kmodel.predict(adv_x_eval/255), axis=1)
                
                adv_result.append(preds2[0])
                
                if not(os.path.isdir(os.getcwd() + '/Data/MNIST/FGSM_e' + str(eps) + '/' + str(y_test[i][0]) + '/')):
                        os.makedirs(os.path.join(os.getcwd() + '/Data/MNIST/FGSM_e' + str(eps) + '/' + str(y_test[i][0]) + '/'))
                                     
                if preds1[i][0] == y_test[i][0] and preds2[0] != y_test[i][0]:
                    np.save(os.getcwd() + '/Data/MNIST/FGSM_e' + str(eps) + '/' + str(y_test[i][0]) + '/' 
                            + str(count_class_save[y_test[i][0]])
                            + '_origin.npy', x_test[i])
                    np.save(os.getcwd() + '/Data/MNIST/FGSM_e' + str(eps) + '/' + str(y_test[i][0]) + '/'
                            + str(count_class_save[y_test[i][0]])
                            + '_adv.npy', adv_x_eval[0])
        
                    count_class_save[y_test[i][0]] = count_class_save[y_test[i][0]] + 1
                
                if preds1[i][0] == y_test[i][0]:
                    origin_count+=1
                if preds2[0] == y_test[i][0]:
                    adv_count+=1
                    
            f.write('Epsilon: ' + str(eps) + '\n')
            f.write('Origin Accuracy: ' + str(origin_count/len(x_test) * 100) + '%\n')
            f.write('Adv Accuracy: ' + str(adv_count/len(x_test) * 100) + '%\n\n')
            

#attack_JSMA
attack = SaliencyMapMethod(wrap, sess=sess)

#attack_BIM
attack = BasicIterativeMethod(wrap, sess=sess)

#attack_DeepFool
attack = DeepFool(wrap, sess=sess)

#attack_CW
attack = CarliniWagnerL2(wrap, sess=sess)