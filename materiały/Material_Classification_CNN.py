# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:33:57 2017
@author: Eadan Valent
"""

import numpy as np
import scipy.io
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# folders of speckle images
folders = ['T:\\GA Matlab Codes\\Material Recongnition\\data\\He-Ne\\NoLensNoTube_LP\\Adj Cam Settings\\Wood_B\\03-Sep-2017\\',
           'T:\\GA Matlab Codes\\Material Recongnition\\data\\He-Ne\\NoLensNoTube_LP\\Adj Cam Settings\\Wood_D\\03-Sep-2017\\']

# determine subimage size, whether its coarse grained, and CNN parameters 
kernal_1 = 2
stride_1 = 1
course = 1.0
target_size = 250
batch_size = 20
epochs = 15
N_x = np.int(np.floor(1002*course/target_size))

X = []
y = []
# retrieve images and their labels (classes)
for Class in range(len(folders)):
    print('Reading class #'+str(Class))
    folder = folders[Class]
    filename = folder+'Summary.mat'
    N = int(scipy.io.loadmat(filename)['N'])+1
    for i in range(1,N):
        filename = folder+'Image'+str(i)+'.mat'
        Xi = scipy.io.loadmat(filename)['Image']
        Xi = Xi[:,326:]
        Xi = Xi/np.mean(Xi)
   #     Xi = imresize(Xi,course)
        for j_x in range(N_x):
            for j_y in range(N_x):
                Xi_part = Xi[j_x*target_size:(j_x+1)*target_size,j_y*target_size:(j_y+1)*target_size]
                X.append(Xi_part)
                y.append(Class)
        
X = np.array(X)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

img_rows, img_cols = np.shape(x_train)[1], np.shape(x_train)[2]

num_classes = len(np.unique(y))

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define neueral network architecture 
model = Sequential()
model.add(Conv2D(4, kernel_size=(kernal_1,kernal_1),
                 strides=(stride_1,stride_1),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(2, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Acc:', np.around(score[1], decimals=2)) 
print('Test loss:', np.around(score[0], decimals=2))
#print('Course:',course)
print('Target Size:',target_size)  
print('kernal_1:',kernal_1)
print('stride_1:',stride_1)