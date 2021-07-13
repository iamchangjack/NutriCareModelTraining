# -*- coding: utf-8 -*-
"""
@original author: Robert Kamunde
@edited by SIT-ICT-SE ICT2111 AY2020/2021 Team 14 for NutriCare project
"""


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)
print(tf.test.gpu_device_name())

K.clear_session()

n_classes = 59   # based on amount of classes in dataset. change when dataset changes.

img_width, img_height, channels = 299, 299, 3
dataset_dir = 'foodSG'   # dataset images should be unzipped here. they should be in folders (e.g. chicken rice, laksa)

# how it should look like:
# PROJECT DIRECTORY
# > foodsg
# >> chicken rice (example)
# >>> 1.jpg
# >>> 2.jpg
# >>> ... (and so on)
# >> duck rice (example)
# >>> 1.jpg
# >>> 2.jpg
# >>> ... (and so on)

train_data_dir = 'foodSG/train' # not needed, but useful in future
validation_data_dir = 'foodSG/test' # not needed, but useful in future
nb_train_samples = 8967  # i.e. amount of images in foodsg/train folder
nb_validation_samples = 2212 # i.e. amount of images in foodsg/test folder. should be n_classes*10


batch_size = 16  # higher values = faster training, but more memory needed
# recommended batch size values are 16, 32, 64, 128, 256

EPOCHS = 20     # more epochs = better accuracy, but will plateau eventually

# below based on: https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

# creating image data generator for train images, to make up for small dataset

train_datagen = ImageDataGenerator(
                 rescale=1./255,
                 rotation_range=40,
                 width_shift_range=0.2,
                 height_shift_range=0.2,
                 shear_range=0.2,
                 zoom_range=0.2,
                 horizontal_flip=True,
                 fill_mode='nearest',
                 validation_split=0.2)  # 20% of the dataset is used as the test/validate set

# creating image data generator for validation images. they should not be augmented other than rescaling.
# still need ImageDatGenerator for the validation_split

test_datagen = ImageDataGenerator(
                rescale=1. / 255,
                validation_split=0.2)

#train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical')

#validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical')

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=123,
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    dataset_dir,     # same directory as before - but validation split will mean different images are used
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=123,
    shuffle=True)

# not sure if train_generator and validation_generator has any mix of images.
# mixing = not good, validation should not mix with train, leads to misleading model

# instantiate the model base
# weights - use the weights that was acquired when the model is initially trained on the ImageNet dataset
# include_top=False will essentially allow the model to be fine-tuned. i.e. the 'top layer' is excluded, allowing for our own 'top layer'
# input_shape - standardises all inputs to be of a certain dimension

nutricare_model= MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape = (img_height,img_width,channels)) # MV uses 299, 299, 3
x = nutricare_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

# REMEMBER TO CHANGE BELOW NUMBER VALUE TO WHICHEVER AMOUNT OF CLASSES IN YOUR DATASET. or just use n_classes
predictions = Dense(n_classes,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=nutricare_model.input, outputs=predictions)

base_learning_rate = 0.0001

model.compile(optimizer=SGD(lr=base_learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='MN_best_model_3class_sept.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('MN_history.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('MN_model_trained.h5')

# ====

train_generator = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)