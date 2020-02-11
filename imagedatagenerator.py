import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
 
 
 
np.random.seed(3)
 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
 
 
data_datagen = ImageDataGenerator(rescale=1./255)
 
data_datagen = ImageDataGenerator( rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest' )
 
 
filename_in_dir = [] 
 
for root, dirs, files in os.walk('./downloads/pome'):
    for  fname in files:
        full_fname = os.path.join(root, fname)
        filename_in_dir.append(full_fname)
 
for file_image in filename_in_dir:
    print (file_image)
    img = load_img(file_image) 
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
 
    i = 0
 
    for batch in data_datagen.flow(x,save_to_dir='./augumentation/pome_gen', save_prefix='dog', save_format='jpg'):
        i += 1
        if i > 4:
            break
