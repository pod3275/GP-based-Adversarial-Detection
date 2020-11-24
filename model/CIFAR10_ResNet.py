# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:14:05 2020

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
import numpy as np
import tensorflow as tf

from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2


class CIFAR10_ResNet():
    
    def __init__(self, batch_size=128, epochs=120, data_augmentation=False,
                 subtract_pixel_mean=False, resnet_n=5):
        # Training parameters
        self.batch_size = batch_size  # orig paper trained all networks with batch_size=128
        self.epochs = epochs
        self.data_augmentation = data_augmentation
        self.num_classes = 10
        
        # Subtracting pixel mean improves accuracy
        self.subtract_pixel_mean = subtract_pixel_mean
        
        # Model parameter
        # ----------------------------------------------------------------------------
        #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
        # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
        #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
        # ----------------------------------------------------------------------------
        # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
        # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
        # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
        # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
        # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
        # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
        # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
        # ---------------------------------------------------------------------------
        self.n = resnet_n
        
        # Model version
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
        self.version = 1
        
        # Computed depth from supplied model parameter n
        if self.version == 1:
            self.depth = self.n * 6 + 2
        elif self.version == 2:
            self.depth = self.n * 9 + 2
        
        # Model name, depth and version
        self.model_type = 'ResNet%dv%d' % (self.depth, self.version)
    
        # 데이터에 따라 GPU 메모리 동적할당
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.Session(config=config)    
    
    def lr_schedule(self, epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 70, 100, 120, 150 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 150:
            lr *= 0.5e-3
        elif epoch > 120:
            lr *= 1e-3
        elif epoch > 100:
            lr *= 1e-2
        elif epoch > 70:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr
    
    
    def resnet_layer(self, inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))
    
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x
    
    
    def resnet_v1(self, input_shape, depth, num_classes=10):
        """ResNet Version 1 Model builder [a]
        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)
    
        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     strides=strides)
                y = self.resnet_layer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                         num_filters=num_filters,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=None,
                                         batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2
    
        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)
    
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    
    def resnet_v2(self, input_shape, depth, num_classes=10):
        """ResNet Version 2 Model builder [b]
        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)
    
        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = self.resnet_layer(inputs=inputs,
                             num_filters=num_filters_in,
                             conv_first=True)
    
        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample
    
                # bottleneck residual unit
                y = self.resnet_layer(inputs=x,
                                     num_filters=num_filters_in,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=activation,
                                     batch_normalization=batch_normalization,
                                     conv_first=False)
                y = self.resnet_layer(inputs=y,
                                     num_filters=num_filters_in,
                                     conv_first=False)
                y = self.resnet_layer(inputs=y,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                         num_filters=num_filters_out,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=None,
                                         batch_normalization=False)
                x = keras.layers.add([x, y])
    
            num_filters_in = num_filters_out
    
        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)
    
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    
    def load_cifar10_data(self):
        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Input image dimensions.
        self.input_shape = x_train.shape[1:]
        
        # Normalize data.
        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255
        
        # If subtract pixel mean is enabled
        if self.subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            self.x_train -= x_train_mean
            self.x_test -= x_train_mean
        
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print('y_train shape:', y_train.shape)
        print(x_test.shape[0], 'test samples')
        
        # Convert class vectors to binary class matrices.
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)
    
    
    def generate_model(self):
        if self.version == 2:
            self.model = self.resnet_v2(input_shape=self.input_shape, depth=self.depth)
        else:
            self.model = self.resnet_v1(input_shape=self.input_shape, depth=self.depth)
        
        self.model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=self.lr_schedule(0)),
                          metrics=['accuracy'])
        self.model.summary()
        
        # Prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'cifar10_%s_model.h5' % self.model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        
        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)
        
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        
        self.callbacks = [checkpoint, lr_reducer, lr_scheduler]
    
    
    def train(self):
        # Run training, with or without data augmentation.
        if not self.data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(self.x_train, self.y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          validation_data=(self.x_test, self.y_test),
                          shuffle=True,
                          callbacks=self.callbacks)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
        
            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(self.x_train)
        
            # Fit the model on the batches generated by datagen.flow().
            self.model.fit_generator(datagen.flow(self.x_train, self.y_train, 
                                                  batch_size=self.batch_size),
                                    validation_data=(self.x_test, self.y_test),
                                    epochs=self.epochs, verbose=1, workers=4,
                                    callbacks=self.callbacks)
    
    
    def test(self):
        # Score trained model.
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        
        
    def getModel(self):
        self.load_cifar10_data()
        self.generate_model()
        self.train()
        self.test()