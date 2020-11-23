# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:38:29 2018

@author: lawle
"""

import tensorflow as tf

if __name__ == '__main__':
    
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    flags.DEFINE_string('dataset', 'MNIST', 'Training dataset name.')
    
    if FLAGS.dataset == "MNIST":
        from MNIST_CNN import MNIST_CNN
        MNIST_CNN().getModel()
    
    elif FLAGS.dataset == 'CIFAR10':
        from CIFAR10_ResNet import CIFAR10_ResNet
        CIFAR10_ResNet().getModel()
