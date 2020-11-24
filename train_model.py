# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:38:29 2018

@author: lawle
"""

import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    flags.DEFINE_string('dataset', 'CIFAR10', 'Training dataset name.')
    
    if FLAGS.dataset == "MNIST":
        from model.MNIST_CNN import MNIST_CNN
        model = MNIST_CNN(batch_size=32, epochs=1)
        model.getModel()
    
    elif FLAGS.dataset == 'CIFAR10':
        from model.CIFAR10_ResNet import CIFAR10_ResNet
        CIFAR10_ResNet(batch_size=32, epochs=1).getModel()
