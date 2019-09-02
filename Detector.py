# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:44:28 2018

@author: lawle
"""

import keras
import tensorflow as tf
import numpy as np
import os
import GPy
import timeit

from tqdm import tqdm
from keras.models import load_model
# ----------------------------------------------------------------------
# Hyperparameter Setting
iteration = [30]
# total number of data = 위 * 10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'MNIST', 'Training dataset name')
flags.DEFINE_string('attack', 'FGSM', 'Adversarial attack name')
flags.DEFINE_integer('num_data_in_class', 30, 'Number of adversarial example per one class used for detector')

print("Detect", FLAGS.attack, "attack on", FLAGS.dataset, "CNN model with", str(FLAGS.num_data_in_class*10), "adversarial examples.")

train_dataset = FLAGS.dataset + '/' + FLAGS.attack

# Detection result will be written on .txt file
f = open(FLAGS.dataset + '_' + FLAGS.attack + '_' + str(FLAGS.num_data_in_class*10) + '_Detector.txt', 'w')
f.write(FLAGS.dataset + '_' + FLAGS.attack + '_' + str(FLAGS.num_data_in_class*10) + '_Detector\n')

keras.backend.set_learning_phase(0)
if FLAGS.dataset == 'CIFAR10': kmodel = load_model('saved_models/cifar10_ResNet32v1_model.h5')
elif FLAGS.dataset == 'MNIST': kmodel = load_model('saved_models/MNIST_model.h5')

# ----------------------------------------------------------------------
def preprocess_input2(x):
        
    return x/255

# ----------------------------------------------------------------------
def load_img(dataset):
    location = os.getcwd() + '/Data/' + dataset
    file_list = os.listdir(location)
    
    count_data = np.zeros(10)
    
    adv_img = []
    adv_img_test = []
    clean_img = []
    clean_img_test = []
    total = 0
    
    for i in tqdm(file_list):
        path_dir = location + '/' + i
        img_list = os.listdir(path_dir)
        img_list.sort()

        for file_name in img_list:
            img = np.load(path_dir + '/' + file_name)
            img_plx1 = np.asarray(img, dtype='float32')
        
            if 'adv' in file_name:
                if count_data[int(i)] < 2* FLAGS.num_data_in_class:
                    adv_img.append(img_plx1)
                    count_data[int(i)] = count_data[int(i)]+1
                else:
                    adv_img_test.append(img_plx1)
                            
            if 'origin' in file_name:
                if count_data[int(i)] < 2* FLAGS.num_data_in_class:
                    clean_img.append(img_plx1)
                    count_data[int(i)] = count_data[int(i)]+1
                else:
                    clean_img_test.append(img_plx1)
                    
            total = total + 1
                
    adv_img = np.asarray(adv_img, dtype='float32')
    clean_img = np.asarray(clean_img, dtype='float32')
    adv_img_test = np.asarray(adv_img_test, dtype='float32')
    clean_img_test = np.asarray(clean_img_test, dtype='float32')
                                                  
    return adv_img, clean_img, adv_img_test, clean_img_test, total

# ----------------------------------------------------------------------
def get_last_hidden_output(adv_img, clean_img):
    hidden_x = []
    hidden_y = []
        
    for i in tqdm(range(len(adv_img))):
        adv_buffer = adv_img[i][np.newaxis, :, :, :]
        clean_buffer = clean_img[i][np.newaxis, :, :, :]
        
        adv_out_buffer = kmodel.predict(preprocess_input2(adv_buffer.copy()))
        clean_out_buffer = kmodel.predict(preprocess_input2(clean_buffer.copy()))
        
        hidden_x.append(adv_out_buffer[0])
        hidden_y.append([1])
        hidden_x.append(clean_out_buffer[0])
        hidden_y.append([0])
    
    hidden_x = np.asarray(hidden_x, dtype='float32')
    hidden_y = np.asarray(hidden_y, dtype='float32')
        
    return hidden_x, hidden_y
         
# ----------------------------------------------------------------------
def model_prediction(model, x, y, epoch):
    train_count = 0
    ans = model.predict(x)
    y_predict = np.zeros_like(y)
    
    for i in range(len(ans[0])):
        if ans[0][i][0] >= 0.5:
            y_predict[i] = 1
        else:
            y_predict[i] = 0
            
        if ans[0][i][0] != 0.5:
            train_count = train_count + 1
    
    tp=0
    tn=0
    fp=0
    fn=0
    
    for i in range(len(y_predict)):
        if y_predict[i] == y[i] and y[i] == 1:
            tp = tp + 1
        if y_predict[i] == y[i] and y[i] == 0:
            tn = tn + 1
        if y_predict[i] != y[i] and y[i] == 0:
            fp = fp + 1
        if y_predict[i] != y[i] and y[i] == 1:
            fn = fn + 1
          
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 = 2* (precision*recall) / (precision+recall)
    
    f.write('Iteration : ' + str(epoch) + '\n')
    f.write('Accuracy: ' + str((tp+tn)/(tp+tn+fp+fn)*100) +'%\n')
    f.write('Precision: ' + str(precision)+',  Recall: ' + str(recall)+'\nF1 Score: ' + str(F1) +'\n\n')
    print('Accuracy: ' + str((tp+tn)/(tp+tn+fp+fn)*100) +'%\n')
    # f.write('테스트 데이터 수: '+ str(len(x))+ ',  바뀐 개수: '+ str(train_count)  +'\n\n')
    
# ----------------------------------------------------------------------
if __name__ == '__main__': 
    print('\n===============Data Load================\n')
    adv, clean, _, _, _ = load_img(train_dataset)
    _, _, adv_test, clean_test, total = load_img(train_dataset)
    print('\n========Training Data Calculate=========\n')
    x_train, y_train = get_last_hidden_output(adv, clean)
    print('\n==========Test Data Calculate===========\n')
    x_test, y_test = get_last_hidden_output(adv_test, clean_test)

    f.write('Training data: '+str(len(x_train))+'\n')
    f.write('Test data: '+ str(len(x_test))+'\n')
    print('\n')
    #draw the latent function value
    for iterate in iteration:
        print('Iteration : ', iterate)
        
        start = timeit.default_timer()
        
        k1 = GPy.kern.Exponential(input_dim=10, variance=1.0, lengthscale=0.001)
        m = GPy.models.GPClassification(x_train, y_train, kernel=k1)
        
        print('\n=============Model Training==============\n')
        for i in range(iterate):
            m.optimize('lbfgs', max_iters=1000)
            
        end = timeit.default_timer()
        print(str(end - start) + ' seconds')
        
        model_prediction(m, x_test, y_test, iterate)
    
    f.close()
# ----------------------------------------------------------------------
