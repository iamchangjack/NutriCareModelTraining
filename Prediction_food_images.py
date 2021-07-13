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
MN_model = load_model('MN_model_trained.h5', compile = False)
MN_colab_model = load_model('MN_colab_model.h5', compile = False)

food_list = create_foodlist("meta.txt")

def predict_class(model, images, size, show = True):
  for img in images:

    img_name = img

    img = image.load_img(img, target_size=(size, size))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)    # Returns the indices of the maximum values along an axis, In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    food_list.sort()
    pred_value = food_list[index]

    score = tf.nn.softmax(pred)

    print("Image name:" + img_name)
    print("Highest: " + food_list[index])

    # printing % chance of food

    print(np.max(score))    # chance is still very low (4%) even if top-5 is accurate. may be good enough?
    print(np.min(score))    # least likely food chance. should be a significant % smaller than the max.

    # this chance seems small but note that the matrix array has % chance of EVERY SINGLE other food in the class.

    # does this mean it gets worse when we add more classes? will find out when doubling the dataset with other data

    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        print("Image appears to be - " + pred_value)
        plt.show()

image1 = 'lor_mee.jpg'      # replace with whichever file name of the food you want to detect.
image2 = 'kway_chap.jpg'
image3 = 'fried_rice.jpg'
image4 = 'murtabak.jpg'
image5 = 'char_siew_rice.jpg'

images = []
images.append(image1)
images.append(image2)
images.append(image3)
images.append(image4)
images.append(image5)

print("Predicting using MobileNetV2")

predict_class(MN_model, images, 299, True)

print("Predicting using EfficientNetB2")

predict_class(MN_colab_model, images, 299, True)

# trying different syntax to get results

img = tf.keras.preprocessing.image.load_img(image1, target_size=(299,299))
input_arr = keras.preprocessing.image.img_to_array(img)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = MN_model.predict(input_arr)

score = tf.nn.softmax(predictions[0])

print("printing food list")
print(food_list)
print(score)


