import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
data_name = 'MNIST'
attack_name = 'CW'
num_data_in_class = 30

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_img():
    location = os.getcwd() + '/Data/' + data_name + '/' + attack_name
    file_list = os.listdir(location)
        
    count_data = np.zeros(10)
    
    adv_img = []
    adv_img_test = []
    clean_img = []
    clean_img_test = []
    clean_y = []
    total = 0
    
    for i in tqdm(file_list):
        path_dir = location + '/' + i
        img_list = os.listdir(path_dir)
        img_list.sort()

        for file_name in img_list:
            img = np.load(path_dir + '/' + file_name)
            img_plx1 = np.asarray(img, dtype='float32')
        
            if 'adv' in file_name:
                if count_data[int(i)] < 2* num_data_in_class:
                    adv_img.append(img_plx1)
                    count_data[int(i)] = count_data[int(i)]+1
                    clean_y.append(int(i))
                else:
                    adv_img_test.append(img_plx1)
                            
            if 'origin' in file_name:
                if count_data[int(i)] < 2* num_data_in_class:
                    clean_img.append(img_plx1)
                    count_data[int(i)] = count_data[int(i)]+1
                else:
                    clean_img_test.append(img_plx1)
                    
            total = total + 1
                
    adv_img = np.asarray(adv_img, dtype='float32')/255
    clean_img = np.asarray(clean_img, dtype='float32')/255
    adv_img_test = np.asarray(adv_img_test, dtype='float32')/255
    clean_img_test = np.asarray(clean_img_test, dtype='float32')/255
                                                  
    return adv_img, clean_img, adv_img_test, clean_img_test, clean_y, total

f = open(data_name + '_' + attack_name + '_' + str(num_data_in_class*10) + '.txt', 'w')

if data_name == 'MNIST':
    img_rows = 28
    img_cols = 28
    img_chan = 1
    nb_classes = 10

elif data_name == 'CIFAR10':
    img_rows = 32
    img_cols = 32
    img_chan = 3
    nb_classes = 10

input_shape=(img_rows, img_cols, img_chan)
sess = tf.InteractiveSession()
K.set_session(sess)

X_train_adv, X_train, X_test_adv, X_test, clean_y, count = load_img()

print('\nPreparing clean/adversarial mixed dataset')
X_all_train = np.vstack([X_train, X_train_adv])
y_all_train = np.vstack([np.zeros([X_train.shape[0], 1]),
                         np.ones([X_train_adv.shape[0], 1])])
ind = np.random.permutation(X_all_train.shape[0])
X_all_train = X_all_train[ind]
y_all_train = y_all_train[ind]


X_all_test = np.vstack([X_test, X_test_adv])
y_all_test = np.vstack([np.zeros([X_test.shape[0], 1]),
                        np.ones([X_test_adv.shape[0], 1])])
ind = np.random.permutation(X_all_test.shape[0])
X_all_test = X_all_test[ind]
y_all_test = y_all_test[ind]

for c in range(5):
    print('\nBuilding model')
    if data_name == 'MNIST':
        model = Sequential([
            Conv2D(32, (3, 3), input_shape=input_shape),
            Activation('relu'),
            Conv2D(32, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(1),
            Activation('sigmoid')])
            
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        model.fit(X_all_train, y_all_train, epochs=2, validation_split=0)
    
        f.write('Iteration : ' + str(c+1))
        print('Iteration : ' + str(c+1))
        score = model.evaluate(X_all_test, y_all_test)
        print('\nloss: {0:.4f} acc: {1:.4f}\n'.format(score[0], score[1]))
        f.write('\nloss: {0:.4f} acc: {1:.4f}\n'.format(score[0], score[1]))
    
    elif data_name == 'CIFAR10':
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            LeakyReLU(alpha=0.2),
            Conv2D(32, (3, 3)),
            LeakyReLU(alpha=0.2),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.2),
            Conv2D(64, (3, 3), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2D(64, (3, 3)),
            LeakyReLU(alpha=0.2),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256),
            Activation('relu'),
            Dropout(0.5),
            Dense(1),
            Activation('sigmoid')])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        model.fit(X_all_train, y_all_train, epochs=7, validation_split=0)
    
        f.write('Iteration : ' + str(c+1))
        print('Iteration : ' + str(c+1))
        score = model.evaluate(X_all_test, y_all_test)
        print('\nloss: {0:.4f} acc: {1:.4f}\n'.format(score[0], score[1]))
        f.write('\nloss: {0:.4f} acc: {1:.4f}\n'.format(score[0], score[1]))

sess.close()
f.close()
