# -*- coding: utf-8 -*-
"""
@original author: Robert Kamunde
@edited by SIT-ICT-SE ICT2111 AY2020/2021 Team 14 for NutriCare project
"""


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.efficientnet import EfficientNetB2   # seems to work?
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np

import tensorflow as tf
print(tf.__version__)
print(tf.test.gpu_device_name())

K.clear_session()

n_classes = 35   # Testing 35 classes of SG foods, so n_classes = 35
img_width, img_height = 299, 299    # may need to adjust based on efficientnet needs. keeping at default of 299/299
train_data_dir = 'foodSG/train'
validation_data_dir = 'foodSG/test'
nb_train_samples = 6595 # i.e. amount of images in foodsg/train folder
nb_validation_samples = 350 # i.e. amount of images in foodsg/test folder

# batch size - higher = faster learning, but more memory needed.
# was 20 but hit OOM. try 10. Setting too high (e.g. 10) WILL MAKE YOUR COMPUTER CRASH

batch_size = 10

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


efficient_net_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape = (299,299,3))
x = efficient_net_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

# REMEMBER TO CHANGE BELOW NUMBER VALUE TO WHICHEVER AMOUNT OF CLASSES IN YOUR DATASET
predictions = Dense(35,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=efficient_net_model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_3class_sept.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=20,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained.h5')
