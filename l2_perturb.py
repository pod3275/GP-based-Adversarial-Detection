# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:10:17 2018

@author: lawle
"""

import tensorflow as tf
from utils import l2_perturb

if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    flags.DEFINE_string('dataset', 'CIFAR10', 'target dataset of the trained model')
    flags.DEFINE_string('attack', 'BIM_e9', 'attack to check the perturbation')
    
    l2_perturb(FLAGS.dataset, FLAGS.attack)





