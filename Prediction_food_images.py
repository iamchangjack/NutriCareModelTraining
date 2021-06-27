"""
@original author: Robert Kamunde
@edited by SIT-ICT-SE ICT2111 AY2020/2021 Team 14 for NutriCare project
"""

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

# grabs the labels.txt file to create list of foods
def create_foodlist(path):
    list_ = list()

    with open(path, 'r') as txt:
        foods = [read.strip() for read in txt.readlines()]
        for f in foods:
            list_.append(f)
            print("Appended food - " + f)
    return list_    

# loading the model that was trained and fine-tuned
my_model = load_model('model_trained.h5', compile = False)
food_list = create_foodlist("foodsg/meta/labels.txt")

# deprecated function, this method is not called but may be useful in the future
def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)    #Returns the indices of the maximum values along an axis, In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    food_list.sort()
    pred_value = food_list[index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        print("Image appears to be - " + pred_value)
        plt.show()

image1 = 'lor_mee.jpg'      # replace with whichever file name of the food you want to detect.
image2 = 'kway_chap.jpg'

img = keras.preprocessing.image.load_img(
    image2, target_size=(299, 299)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = my_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(food_list[np.argmax(score)], 100 * np.max(score))
)
