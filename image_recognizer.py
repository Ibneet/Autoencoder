import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import shuffle

in_data = '/Users/ibneetkaur/Desktop/coding/image_recognizer/train'

def refining_img():
    labeled_images = []
    for i in tqdm(os.listdir(in_data)):
        path = os.path.join(in_data,i)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize((cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),(100, 100))
       #    img = cv2.resize(img,(100,100))
            labeled_images.append([np.array(img), img_label(i)])
        else:
            print('Image not loaded')
        shuffle(labeled_images)
    return labeled_images

def img_label(img):
    img_lbl = []
    label = img.split('.')[0]
    if label == 'Ibneet':
        img_lbl = np.array([1,0])
    elif label == 'Gurjot':
        img_lbl = np.array([0,1])   
    return img_lbl 

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.utils import print_summary

images_with_label = np.asarray(refining_img())

images = np.array([i[0] for i in images_with_label]).reshape(-1,100,100,3)
label = np.array([i[1] for i in images_with_label])

train_img, test_img, train_lbl, test_lbl = train_test_split(images, label, test_size=0.2,
                                                    random_state=42)


input_img = Input(shape=(100, 100, 3))



x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x2 = MaxPooling2D((2, 2), padding='same')(x1)
x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
x4 = MaxPooling2D((2, 2), padding='same')(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
encoded = MaxPooling2D((2, 2), padding='same')(x5)

y1 = Flatten()(x5)
y2 = Dense(1024, activation='relu')(y1)
y3 = Dropout(0.2)(y2)
y4 = Dense(2, activation='softmax')(y3)

#x6 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
#x7 = UpSampling2D((2, 2))(x6)
#x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
#x10 = Conv2D(64, (3, 3), activation='relu')(x9)
#x11 = UpSampling2D((2, 2))(x10)
#decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x11)

autoencoder = Model(input_img, y4)

encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


print_summary(autoencoder)

train_img = train_img.astype('float32')
train_img /= 255

test_img = test_img.astype('float32')
test_img /= 255

autoencoder.fit(train_img, train_lbl,
                epochs=50,
                batch_size=20,
                shuffle=True)

predicted_imgs = autoencoder.predict(test_img)


plt.figure(figsize=(40, 40))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(train_img[i])
    plt.title(train_lbl[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(40, 40))
for i in range(20):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(test_img[i])
    if np.argmax(predicted_imgs[i]) == 1:
        lbl = 'Gurjot'
    else:
        lbl = 'Ibneet'
    plt.title(lbl)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#############################################################################################
from playsound import playsound

def detect():
    faceCascade = cv2.CascadeClassifier('/Users/ibneetkaur/Downloads/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3,640) 
    cap.set(4,480) 
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(
                img_rgb,     
                scaleFactor=1.2,
                minNeighbors=5,  
                minSize=(100, 100)
                )
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
           
                image = cv2.resize(img[y:y+h,x:x+w],(100,100))
#                image = img_rgb[y:y+h,x:x+w]
                image = np.reshape(image,(-1,100,100,3))
                image = image.astype('float32')
                image /= 255
                 
                predicted_imgs = autoencoder.predict(image)
                
            
                if predicted_imgs[:,1] > 0.7:
                    lbl = 'Gurjot'
                    wel = playsound("welcome1.mp3")
                elif predicted_imgs[:,0] > 0.7 :
                    lbl = 'Ibneet'
                    wel = playsound("welcome2.mp3")
                else:
                    lbl = 'Unkown'
                
                id = lbl
                print(predicted_imgs)
               
                cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
                wel
                cv2.imshow('Input',img)
                k = cv2.waitKey(10) & 0xff
                if k == 27: # press 'ESC' to quit
                    break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    
enter = input('Press a to start webcam predictions.')

if enter == 'a':
    detect()





from gtts import gTTS

message1 = gTTS(text='Hi Gurjot! How are you.', lang='en', slow=False)
message2 = gTTS(text='Hi Ibneet! How are you.', lang='en', slow=False)

message1.save("welcome1.mp3")
message2.save("welcome2.mp3")





























