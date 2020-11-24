# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:04:06 2018

@author: lawle
"""

import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from keras.datasets import mnist, cifar10
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, CarliniWagnerL2, SaliencyMapMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from keras.models import load_model
from keras import backend as K


class AdvAttack():
    
    def __init__(self, dataset, attack):
        self.dataset = dataset
        self.attack = attack
        
        self.save_loc = f'/adv_data/{self.dataset}/{self.attack}'
        
        self.out_file = open(f"results/{self.dataset}_{self.attack}.txt", "w")
        self.out_file.write(f"{self.dataset}_{self.attack}_Classifier\n")
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        keras.backend.set_session(self.sess)
        keras.backend.set_learning_phase(0)
        
        self.execute_attack()
        self.out_file.close()


    def _load_dataset_model(self):
        if self.dataset == 'MNIST':
            self.img_rows, self.img_cols = 28, 28
            self.num_classes = 10
            
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
            if K.image_data_format() == 'channels_first':
                self.x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
                self.x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
                self.input_shape = (1, self.img_rows, self.img_cols)
            else:
                self.x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
                self.x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
                self.input_shape = (self.img_rows, self.img_cols, 1)
                
            self.y_train = y_train.reshape(y_train.shape[0], 1)
            self.y_test = y_test.reshape(y_test.shape[0], 1)
            
            model_file_name = 'MNIST_model.h5'
        
        elif self.dataset == 'CIFAR10':
            print("\n load data...")
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
            print("data loaded.")
            
            model_file_name = 'cifar10_ResNet32v1_model.h5'
            
        self.img_rows, self.img_cols, self.n_channels = self.x_train.shape[1:4]
        self.n_classes = self.y_train.shape[1]

        print("\n load model...")
        self.model = load_model(f'saved_models/{model_file_name}')
        self.wrapped_model = KerasModelWrapper(self.model)
        print("model loaded.")
        

    def _set_placeholder(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.img_rows, self.img_cols, self.n_channels))
        self.y = tf.placeholder(tf.float32, shape=(None, self.n_classes))
        if self.attack == 'BIM':
            self.y = tf.placeholder(tf.float32, shape=(None, 10))
            
        origin_preds = np.argmax(self.model.predict(self.x_test/255), axis=1)
        self.origin_preds = origin_preds.reshape(origin_preds.shape[0], 1)


    def save_images(self, adv_x, save_dir):
        saved_class_idx = np.zeros(10).astype('int32')
        origin_correct_count=0
        adv_correct_count=0
                
        for i in tqdm(range(len(self.x_test))):
            test_input = self.x_test[i][np.newaxis,:,:,:]
            y_input = [self.origin_preds[i]]
    
            if self.attack == 'BIM':
                y_input = keras.utils.to_categorical(y_input, 10)
    
            adv_x_eval = adv_x.eval(session=self.sess, 
                                    feed_dict={self.x:test_input/255,
                                               self.y:y_input})
            adv_x_eval = adv_x_eval*255
            adv_pred = np.argmax(self.model.predict(adv_x_eval/255), axis=1)
            correct_label = self.y_test[i][0]
            
            if not(os.path.isdir(os.getcwd() + f"{save_dir}/{correct_label}/")):
                    os.makedirs(os.getcwd() + f"{save_dir}/{correct_label}/")
                                 
            if self.origin_preds[i][0] == correct_label and adv_pred[0] != correct_label:
                np.save(os.getcwd() + f"{save_dir}/{correct_label}/{saved_class_idx[correct_label]}_origin.npy", self.x_test[i])
                np.save(os.getcwd() + f"{save_dir}/{correct_label}/{saved_class_idx[correct_label]}_adv.npy", adv_x_eval[0])
                saved_class_idx[correct_label] += 1
            
            if self.origin_preds[i][0] == correct_label:
                origin_correct_count +=1
                
            if adv_pred[0] == correct_label:
                adv_correct_count +=1
                
            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(adv_x_eval[0] / 255)  # division by 255 to convert [0, 255] to [0, 1]
            # plt.axis('off')
            
        self.out_file.write('saved_class_idx: ' + str(saved_class_idx) + '\n')
        self.out_file.write('Origin accuracy: ' + str(origin_correct_count/len(self.x_test) * 100) + '%\n')
        self.out_file.write('Adv accuracy: ' + str(adv_correct_count/len(self.x_test) * 100) + '%\n\n')

    
    def _FGSM(self):
        fgsm_attack = FastGradientMethod(self.wrapped_model, sess=self.sess)
        eps = 0
        
        if self.dataset == 'MNIST':
            for _ in range(5):
                eps = eps + 0.1
                params = {'eps': eps,
                          'clip_min': 0.,
                          'clip_max': 1.}
                adv_x = fgsm_attack.generate(self.x, **params)
                
                print(f'Epsilon: {eps}')
                self.out_file.write(f'Epsilon: {eps}\n')
                self.save_images(adv_x, self.save_loc + f'_e{eps}')
                
        if self.dataset == 'CIFAR10':
            for _ in range(10):
                eps = eps + 1
                params = {'eps': eps/255,
                          'clip_min': 0.,
                          'clip_max': 1.}
                adv_x = fgsm_attack.generate(self.x, **params)
                
                print(f'Epsilon: {eps}')
                self.out_file.write(f'Epsilon: {eps}\n')
                self.save_images(adv_x, self.save_loc + f'_e{eps}')
                
    
    def _JSMA(self):
        jsma_attack = SaliencyMapMethod(self.wrapped_model, sess=self.sess)
        params = {'clip_min': 0., 'clip_max': 1.}
        adv_x = jsma_attack.generate(self.x, **params)
        self.save_images(adv_x, self.save_loc)
           
    
    def _BIM(self):
        bim_attack = BasicIterativeMethod(self.wrapped_model, sess=self.sess)
        eps = 0
        
        if self.dataset == 'MNIST':
            for _ in range(5):
                eps = eps + 0.1
                params = {'eps': eps,
                          'eps_iter': eps/10,
                          'nb_iter': 10,
                          'y': self.y,
                          'clip_min': 0.,
                          'clip_max': 1.}
                adv_x = bim_attack.generate(self.x, **params)
                adv_x = tf.stop_gradient(adv_x)
                
                print(f'Epsilon: {eps}')
                self.out_file.write(f'Epsilon: {eps}\n')
                self.save_images(adv_x, self.save_loc + f'_e{eps}')
                
        if self.dataset == 'CIFAR10':
            for _ in range(10):
                eps = eps + 1
                params = {'eps': eps/255,
                          'eps_iter': eps/255/10,
                          'nb_iter': 10,
                          'y': self.y,
                          'clip_min': 0.,
                          'clip_max': 1.}
                adv_x = bim_attack.generate(self.x, **params)
                adv_x = tf.stop_gradient(adv_x)
                
                print(f'Epsilon: {eps}')
                self.out_file.write(f'Epsilon: {eps}\n')
                self.save_images(adv_x, self.save_loc + f'_e{eps}')


    def _DeepFool(self):
        deepfool_attack = DeepFool(self.wrapped_model, sess=self.sess)
        params = {'nb_candidate': 10,
                  'max_iter': 100,
                  'clip_min': 0.,
                  'clip_max': 1.,
                  'verbose':False}
        adv_x = deepfool_attack.generate(self.x, **params)
        self.save_images(adv_x, self.save_loc)
    
    
    def _CW(self):
        cw_attack = CarliniWagnerL2(self.wrapped_model, sess=self.sess)
        params = {'batch_size':1,
                  'max_iterations':1000,
                  'binary_search_steps':9,
                  'initial_const':1e-3,
                  'learning_rate':5e-3,
                  'clip_min': 0.,
                  'clip_max': 1.}
        adv_x = cw_attack.generate(self.x, **params)
        self.save_images(adv_x, self.save_loc)


    def execute_attack(self):
        self._load_dataset_model()
        self._set_placeholder()
        if self.attack == "FGSM": self._FGSM()
        elif self.attack == "BIM": self._BIM()
        elif self.attack == "JSMA": self._JSMA()
        elif self.attack == "DeepFool": self._DeepFool()
        elif self.attack == "CW": self._CW()


if __name__ == "__main__":
    
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    flags.DEFINE_string('dataset', 'CIFAR10', 'Training dataset name')
    flags.DEFINE_string('attack', 'DeepFool', 'Adversarial attack name')
    
    print(f"{FLAGS.attack} attack on {FLAGS.dataset} classification model.")
    
    attack_obj = AdvAttack(FLAGS.dataset, FLAGS.attack)