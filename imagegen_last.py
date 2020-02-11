Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
 
 
 
np.random.seed(3)
 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
 
 
data_datagen = ImageDataGenerator(rescale=1./255)
 
data_datagen = ImageDataGenerator( rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,			   
        horizontal_flip=True,
        fill_mode='nearest' )
 
 
filename_in_dir = [] 
 
for root, dirs, files in os.walk('./downloads/test'):
    for  fname in files:
        full_fname = os.path.join(root, fname)
        filename_in_dir.append(full_fname)
 
for file_image in filename_in_dir:
    print (file_image)
    img = load_img(file_image) 
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
 
    i = 0
 
    for batch in data_datagen.flow(x,save_to_dir='./augumentation/york_gen', save_prefix='dog', save_format='jpg'):
        i += 1
        if i > 6:
            break
