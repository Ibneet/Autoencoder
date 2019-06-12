import tensorflow as tf
import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt


train_data = '/Users/ibneetkaur/Desktop/images/train'

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize((cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),(100, 100))
            train_images.append([np.array(img)])
        else:
            print('Image not loaded')
    return train_images

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.utils import print_summary


training_images = np.asarray(train_data_with_label())
training_images= np.array([i[0] for i in training_images]).reshape(-1,100,100,3)

input_img = Input(shape=(100, 100, 3)) 



x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x2 = MaxPooling2D((2, 2), padding='same')(x1)
x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
x4 = MaxPooling2D((2, 2), padding='same')(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
encoded = MaxPooling2D((2, 2), padding='same')(x5)

x6 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(64, (3, 3), activation='relu')(x9)
x11 = UpSampling2D((2, 2))(x10)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x11)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

x2_img_m = Model(input_img, x2)

x7_img_m = Model(input_img, x7)

x11_img_m = Model(input_img, x11)

autoencoder.compile(optimizer='RMSProp', loss='mse',metrics=['accuracy'])


print_summary(autoencoder)

training_images = training_images.astype('float32')
training_images /= 255

autoencoder.fit(training_images, training_images,
                epochs=50,
                batch_size=1,
                shuffle=True)

decoded_imgs = autoencoder.predict(training_images)
encoded_imgs = encoder.predict(training_images)
x2_imgs = x2_img_m.predict(training_images)
x7_imgs = x7_img_m.predict(training_images)
x11_imgs = x11_img_m.predict(training_images)




plt.figure(figsize=(40, 40))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(training_images[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.figure(figsize=(40,40))
for i in range(10):     
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(encoded_imgs[i].reshape(13, 13 * 32).T)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(40,40))
for i in range(10):     
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(decoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(40,40))
for i in range(10):     
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x7_imgs[i].reshape(26, 26*32).T)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.figure(figsize=(40,40))
for i in range(10):     
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x11_imgs[i].reshape(100, 100*64).T)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.figure(figsize=(40,40))
for i in range(10):     
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x2_imgs[i].reshape(50, 50*64).T)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    