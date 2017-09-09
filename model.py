# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 23:03:59 2017

@author: los
"""
import csv
import numpy as np
import cv2
import sklearn
from keras.models import Sequential
#from keras import layers
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D#, MaxPooling2D
from keras.layers import Lambda,Cropping2D
from sklearn.model_selection import train_test_split

correction = 0.2
"""
Read Data
"""
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
"""
Split Data to train_samples and validation_samples
"""
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
"""
Define data generator
"""
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(3):
                        filename = batch_sample[i].split('/')[-1]
                        current_path = 'data/IMG/' + filename
                        image = cv2.imread(current_path)
                        images.append(image)
                    center_angle = float(batch_sample[3])
                    angles.append(center_angle)
                    angles.append(center_angle + correction)
                    angles.append(center_angle - correction)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
"""
Define model
"""
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.7))
model.add(Dense(50))
model.add(Dropout(0.7))
model.add(Dense(10))
model.add(Dense(1))
"""
Train model
"""
model.compile(loss='mse',optimizer='adam')
print(len(train_samples))
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=10)
model.save('model.h5')