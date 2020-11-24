# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:35:37 2020

@author: lawle
"""

from __future__ import print_function

import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import tensorflow as tf

from keras import backend as K
from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint


class MNIST_CNN():
    
    def __init__(self, batch_size=128, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs
        
        # input image dimensions
        self.num_classes = 10
        self.img_rows, self.img_cols = 28, 28
        
        # 데이터에 따라 GPU 메모리 동적할당
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.Session(config=config)


    def load_mnist_data(self):
        # 학습 데이터 : 60000개
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('Num train data:', len(x_train))
        print('Num test data:', len(x_test))

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def generate_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'MNIST_model.h5'
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)
        
        self.model = model
        self.callbacks = [checkpoint]
        
    
    def train(self):
        self.model.fit(self.x_train, self.y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1,
                      validation_data=(self.x_test, self.y_test),
                      callbacks=self.callbacks)


    def test(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
    
    def getModel(self):
        self.load_mnist_data()
        self.generate_model()
        self.train()
        self.test()