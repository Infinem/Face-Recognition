import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2 as cv

import keras 
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization


def smile_detect(image):
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    #Output Layer
    model.add(Dense(units = 1,kernel_initializer="uniform", activation = 'sigmoid'))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # Compiling Neural Network
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.load_weights("smile.h5")

    
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (64,64), interpolation=cv.INTER_CUBIC)
    image = image.reshape(1,64,64,3)

    if (model.predict_classes(image)):
        answer = 1
    else:
        answer = 0
    
    del model
    K.clear_session()
    return answer


