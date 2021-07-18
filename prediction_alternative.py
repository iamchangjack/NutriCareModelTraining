from PIL.ImageOps import crop
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import cv2 as cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # prevent GPU usage

import seaborn as sns
sns.set_style('darkgrid')

# stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def print_in_color(txt_msg,fore_tupple,back_tupple,):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m'
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True) # returns default print color to back to black
    return

def classify(sdir, csv_path,  model_path, crop_image = False):
    # read in the csv file
    class_df=pd.read_csv(csv_path)
    img_height=int(class_df['height'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_width, img_height)
    scale=class_df['scale by'].iloc[0]
    try:
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1])
        print (s1,s2)
    path_list=[]
    paths=os.listdir(sdir)
    for f in paths:
        path_list.append(os.path.join(sdir,f))
    print (' Model is being loaded- this will take about 10 seconds')
    model=load_model(model_path)

    image_count=len(path_list)

    #print("Image count: " + str(image_count))   # only give 1 - guess this means the given image

    index_list=[]
    prob_list=[]

    # stuff for top-5
    food_list_5=[]
    prob_list_5=[]
    # end of stuff for top-5

    cropped_image_list=[]
    good_image_count=0
    for i in range (image_count):
        img=plt.imread(path_list[i])
        if crop_image == True:
            status, img=crop(img)
        else:
            status=True
        if status== True:
            good_image_count +=1
            img=cv2.resize(img, img_size)
            cropped_image_list.append(img)
            img=img*s2 - s1
            img=np.expand_dims(img, axis=0)

            # prediction rates are given here
            p= np.squeeze (model.predict(img))

            #print("Printing max...")
            #print(p)
            index=np.argmax(p)

            probability_array, food_index_array = tf.nn.top_k(p,5)
            # values = probabilities. indices = index of food items

            #print("Printing values")    # prints the array of top 5 probabilities
            #print(probability_array.numpy())
            for i in probability_array.numpy():
                #print(i)
                prob_list_5.append(i)

            #print("Printing indices")   # prints the array of top 5 predicted food (by their index number)
            #print(food_index_array.numpy())
            #for i in food_index_array.numpy():
            #    print(i)
            prob=p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count==1: # provided image is good and probability matrix has been acquired
        class_name= class_df['class'].iloc[index_list[0]]

        # list of food index numbers are nice
        # but list of food names are better
        for i in food_index_array.numpy():
            food_name = class_df['class'].iloc[i]   # tally to the .csv file for the food name
            #print(food_name)
            food_list_5.append(food_name)

        probability= prob_list[0]
        return class_name, probability, food_list_5, prob_list_5
    elif good_image_count == 0:
        return None, None, None, None
    most=0
    for i in range (len(index_list)-1):
        key= index_list[i]
        keycount=0
        for j in range (i+1, len(index_list)):
            nkey= index_list[j]
            if nkey == key:
                keycount +=1
        if keycount> most:
            most=keycount
            isave=i
    best_index=index_list[isave]
    psum=0
    bestsum=0
    for i in range (len(index_list)):
        psum += prob_list[i]
        if index_list[i]==best_index:
            bestsum += prob_list[i]
    img= cropped_image_list[isave]/255
    class_name=class_df['class'].iloc[best_index]
    plt.title(class_name, color='blue', fontsize=16)
    plt.axis('off')
    plt.imshow(img)
    return class_name, bestsum/image_count

working_dir  = os.getcwd()
store_path=os.path.join(working_dir, 'storage')
if os.path.isdir(store_path):
    shutil.rmtree(store_path)
os.mkdir(store_path)
img_path='test images/char_siew_rice.jpg'
img=cv2.imread(img_path)
split=os.path.split(img_path)
file_name=split[1]
class_name=os.path.split(split[0])[1]
full_name=class_name + '-' +file_name
dst_path=os.path.join(store_path, full_name)
cv2.imwrite(dst_path, img)
# check if the directory was created and image stored
print (os.listdir(working_dir))

csv_path='class_dict_foodsg.csv' # path to class_dict.csv
model_path='EN_model_colab_prefinetune.hdf5' # path to the trained model
class_name, probability, name_list, probability_list =classify(store_path, csv_path,  model_path, crop_image = False) # run the classifier
#msg=f'image {full_name} is of {class_name} with a probability of {probability * 100: 6.2f} %'

for index, item in enumerate(probability_list):
    count = index + 1
    item_formatted = "{:.2%}".format(item)
    print(str(count) + ". " + name_list[index] + " - " + item_formatted)

#print_in_color(msg, (0,255,255), (65,85,55))