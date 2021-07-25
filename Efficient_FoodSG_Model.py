# source: https://www.kaggle.com/gpiosenka/how-to-balance-a-dataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint  # for model checkpoint creation
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_style('darkgrid')
from sklearn.metrics import confusion_matrix, classification_report

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR) # stop annoying tensorflow warning messages


def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)  # get a sample batch from the generator
    plt.figure(figsize=(20, 20))
    length = len(labels)
    if length < 25:  # show maximum of 25 images
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()


def print_in_color(txt_msg, fore_tuple, back_tuple, ):
    # prints text in colour. good to emphasize certain print logs
    # text_msg is the text, fore_tuple is foreground color tupple (r,g,b), back_tpple is background tuple (r,g,b)
    rf, gf, bf = fore_tuple
    rb, gb, bb = back_tuple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) + ';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(
        bb) + 'm'
    print(msg.format(mat), flush=True)
    print('\33[0m', flush=True)  # returns default print color to back to black
    return


class learning_rate_adjustor(keras.callbacks.Callback):
    def __init__(self, model, base_model, patience, stop_patience, threshold, factor, dwell, batches, initial_epoch,
                 epochs, ask_epoch):
        # callback method, to be added in callbacks when fitting model
        # overrides training verbose and allows for choice of more epochs/fine tuning/ending the training

        super(learning_rate_adjustor, self).__init__()
        self.model = model
        self.base_model = base_model
        self.patience = patience  # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience = stop_patience  # specifies how many times to adjust lr without improvement to stop training
        self.threshold = threshold  # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor = factor  # factor by which to reduce the learning rate
        self.dwell = dwell
        self.batches = batches  # number of training batch to runn per epoch
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch  # save this value to restore if restarting training
        # callback variables
        self.count = 0  # how many times lr has been reduced without improvement
        self.stop_count = 0
        self.best_epoch = 1  # epoch with the lowest loss
        self.initial_lr = float(
            tf.keras.backend.get_value(model.optimizer.lr))  # get the initiallearning rate and save it
        self.highest_tracc = 0.0  # set highest training accuracy to 0 initially
        self.lowest_vloss = np.inf  # set lowest validation loss to infinity initially
        self.best_weights = self.model.get_weights()  # set best weights to model's initial weights
        self.initial_weights = self.model.get_weights()  # save initial weights if they have to get restored

    def on_train_begin(self, logs=None):
        if self.base_model != None:
            status = base_model.trainable
            if status:
                msg = ' initializing callback starting train with base_model trainable'
            else:
                msg = 'initializing callback starting training with base_model not trainable'
        else:
            msg = 'initialing callback and starting training'
        print_in_color(msg, (244, 252, 3), (55, 65, 80))
        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format('Epoch', 'Loss', 'Accuracy',
                                                                                         'V_loss', 'V_acc', 'LR',
                                                                                         'Next LR', 'Monitor',
                                                                                         'Duration')
        print_in_color(msg, (244, 252, 3), (55, 65, 80))
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))

        self.model.set_weights(self.best_weights)  # set the weights of the model to the best weights
        msg = f'Training is completed - model is set with weights from epoch {self.best_epoch} '
        print_in_color(msg, (0, 255, 0), (55, 65, 80))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print_in_color(msg, (0, 255, 0), (55, 65, 80))

    def on_train_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy') * 100  # get training accuracy
        loss = logs.get('loss')
        msg = '{0:20s}processing batch {1:4s} of {2:5s} accuracy= {3:8.3f}  loss: {4:8.5f}'.format(' ', str(batch),
                                                                                                   str(self.batches),
                                                                                                   acc, loss)
        print(msg, '\r', end='')  # prints over on the same line to show running batch count

    def on_epoch_begin(self, epoch, logs=None):
        self.now = time.time()

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        later = time.time()
        duration = later - self.now
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))  # get the current learning rate
        current_lr = lr
        v_loss = logs.get('val_loss')  # get the validation loss for this epoch
        acc = logs.get('accuracy')  # get training accuracy
        v_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        if acc < self.threshold:  # if training accuracy is below threshold adjust lr based on training accuracy
            monitor = 'accuracy'
            if acc > self.highest_tracc:  # training accuracy improved in the epoch
                self.highest_tracc = acc  # set new highest training accuracy
                self.best_weights = self.model.get_weights()  # traing accuracy improved so save the weights
                self.count = 0  # set count to 0 since training accuracy improved
                self.stop_count = 0  # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                color = (0, 255, 0)
                self.best_epoch = epoch + 1  # set the value of best epoch for this epoch
            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count >= self.patience - 1:  # lr should be adjusted
                    color = (245, 170, 66)
                    lr = lr * self.factor  # adjust the learning by factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)  # set the learning rate in the optimizer
                    self.count = 0  # reset the count to 0
                    self.stop_count = self.stop_count + 1  # count the number of consecutive lr adjustments
                    self.count = 0  # reset counter
                    if self.dwell:
                        self.model.set_weights(
                            self.best_weights)  # return to better point in N space
                    else:
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                else:
                    self.count = self.count + 1  # increment patience counter
        else:  # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor = 'val_loss'
            if v_loss < self.lowest_vloss:  # check if the validation loss improved
                self.lowest_vloss = v_loss  # replace lowest validation loss with new validation loss
                self.best_weights = self.model.get_weights()  # validation loss improved so save the weights
                self.count = 0  # reset count since validation loss improved
                self.stop_count = 0
                color = (0, 255, 0)
                self.best_epoch = epoch + 1  # set the value of the best epoch to this epoch
            else:  # validation loss did not improve
                if self.count >= self.patience - 1:  # need to adjust lr
                    color = (245, 170, 66)
                    lr = lr * self.factor  # adjust the learning rate
                    self.stop_count = self.stop_count + 1  # increment stop counter because lr was adjusted
                    self.count = 0  # reset counter
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)  # set the learning rate in the optimizer
                    if self.dwell:
                        self.model.set_weights(self.best_weights)  # return to better point in N space
                else:
                    self.count = self.count + 1  # increment the patience counter
                if acc > self.highest_tracc:
                    self.highest_tracc = acc
        msg = f'{str(epoch + 1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{duration:^8.2f}'
        print_in_color(msg, color, (55, 65, 80))
        if self.stop_count > self.stop_patience - 1:  # check if learning rate has been adjusted stop_count times with no improvement
            msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print_in_color(msg, (0, 255, 255), (55, 65, 80))
            self.model.stop_training = True  # stop trainings
        else:
            if self.ask_epoch != None:
                if epoch + 1 >= self.ask_epoch:
                    msg = 'enter H to halt ,F to fine tune model, or an integer for number of epochs to run then ask again'
                    print_in_color(msg, (0, 255, 255), (55, 65, 80))
                    ans = input('')
                    if ans == 'H' or ans == 'h':
                        msg = f'training has been halted at epoch {epoch + 1} due to user input'
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        self.model.stop_training = True  # stop training
                    elif ans == 'F' or ans == 'f':
                        msg = 'setting base_model as trainable for fine tuning of model'
                        self.base_model.trainable = True
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format('Epoch',
                                                                                                         'Loss',
                                                                                                         'Accuracy',
                                                                                                         'V_loss',
                                                                                                         'V_acc', 'LR',
                                                                                                         'Next LR',
                                                                                                         'Monitor',
                                                                                                         'Duration')
                        print_in_color(msg, (244, 252, 3), (55, 65, 80))
                        self.count = 0
                        self.stop_count = 0
                        self.ask_epoch = epoch + 1 + self.ask_epoch_initial

                    else:
                        ans = int(ans)
                        self.ask_epoch += ans
                        msg = f' training will continue until epoch ' + str(self.ask_epoch)
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format('Epoch',
                                                                                                         'Loss',
                                                                                                         'Accuracy',
                                                                                                         'V_loss',
                                                                                                         'V_acc', 'LR',
                                                                                                         'Next LR',
                                                                                                         'Monitor',
                                                                                                         'Duration')
                        print_in_color(msg, (244, 252, 3), (55, 65, 80))


def tr_plot(tr_data, start_epoch):
    # plot training accuracy and loss graph

    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout

    plt.show()


def print_info(test_gen, preds, print_code, save_dir, subject):
    class_dict = test_gen.class_indices
    labels = test_gen.labels
    file_names = test_gen.filenames
    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    new_dict = {}
    error_indices = []
    y_pred = []
    for key, value in class_dict.items():
        new_dict[value] = key  # dictionary {integer of class number: string of class name}
    # store new_dict as a text fine in the save_dir
    classes = list(new_dict.values())  # list of string of class names
    errors = 0
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]  # labels are integer values
        if pred_index != true_index:  # a misclassification has occurred
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors = errors + 1
        y_pred.append(pred_index)
    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
            msg = '{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class', 'True Class', 'Probability')
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
            for i in range(r):
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname = split2[1] + '/' + split1[1]
                msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i], true_class[i], ' ',
                                                                       prob_list[i])
                print_in_color(msg, (255, 255, 255), (55, 65, 60))
                # print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg = 'With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count)  # list containg how many times a class c had an error
                plot_class.append(value)  # stores the class
        fig = plt.figure()
        fig.set_figheight(len(plot_class) / 3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title(' Errors by Class on Test Set')
    y_true = np.array(labels)
    y_pred = np.array(y_pred)
    if len(classes) <= 30:
        # create a confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        length = len(classes)
        if length < 8:
            fig_width = 8
            fig_height = 8
        else:
            fig_width = int(length * .5)
            fig_height = int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(length) + .5, classes, rotation=90)
        plt.yticks(np.arange(length) + .5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)


def save_model(save_path, model, model_name, subject, accuracy, img_size, scalar, generator):
    # saves a trained model in the provided save_path under .h5 format

    # first save the model
    save_id = str(model_name + '-' + subject + '-' + str(acc)[:str(acc).rfind('.') + 3] + '.h5')
    model_save_loc = os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print_in_color('model was saved as ' + model_save_loc, (0, 255, 0), (55, 65, 80))
    # now create the class_df and convert to csv file
    class_dict = generator.class_indices
    height = []
    width = []
    scale = []
    for i in range(len(class_dict)):
        height.append(img_size[0])
        width.append(img_size[1])
        scale.append(scalar)
    Index_series = pd.Series(list(class_dict.values()), name='class_index')
    Class_series = pd.Series(list(class_dict.keys()), name='class')
    Height_series = pd.Series(height, name='height')
    Width_series = pd.Series(width, name='width')
    Scale_series = pd.Series(scale, name='scale by')
    class_df = pd.concat([Index_series, Class_series, Height_series, Width_series, Scale_series], axis=1)
    csv_name = 'class_dict.csv'
    csv_save_loc = os.path.join(save_path, csv_name)
    class_df.to_csv(csv_save_loc, index=False)
    print_in_color('class csv file was saved as ' + csv_save_loc, (0, 255, 0), (55, 65, 80))
    return model_save_loc, csv_save_loc

def trim(df, max_size, min_size, column):


    df = df.copy()
    sample_list = []
    groups = train_df.groupby(column)
    for label in train_df[column].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count > max_size:
            samples = group.sample(max_size, replace=False, weights=None, random_state=123, axis=0).reset_index(
                drop=True)
            sample_list.append(samples)
        elif sample_count >= min_size:
            sample_list.append(group)
    df = pd.concat(sample_list, axis=0).reset_index(drop=True)
    balance = list(df[column].value_counts())
    print(balance)
    return df

# SET THE FOLLOWING VARIABLES AS DESIRED.
# START OF CUSTOM VARIABLES

sdir = 'foodsg' # name of your dataset folder
max_size = 150  # maximum amount of images to use per class. depends on how imbalanced your dataset is.
min_size = 0    # not really important
training_split = .9        # 90% of dataset used for training
validation_split = .05        # 5% of dataset used for validation

# height and weight - desired dimensions to input images into model.
height = 224
width = 306
channels = 3

img_shape = (height, width, channels)
img_size = (height, width)

# batch size - recommended 16/32/64/128. Higher = faster but more memory needed. Google Colab GPU runs well with 32.
batch_size = 16

model_name = 'EfficientNetB2'   # just for naming in documentation. has no matter on the actual model instantiated.
base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet", input_shape=img_shape,pooling='max')
# change above model to whichever desired network.
# remember to change img_shape (above) if new network requires certain shapes

# END OF CUSTOM VARIABLES

classlist = os.listdir(sdir)
filepaths = []
labels = []
for food_class in classlist:
    classpath = os.path.join(sdir, food_class)
    flist = os.listdir(classpath)
    for f in flist:
        fpath = os.path.join(classpath, f)
        filepaths.append(fpath)
        labels.append(food_class)
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)

# remaining 5% used for testing the model when it is fully trained

database_split = validation_split / (1 - training_split) # factor for database split
strat = df['labels']
train_df, dummy_df = train_test_split(df, train_size=training_split, shuffle=True, random_state=123, stratify=strat)
strat = dummy_df['labels']
valid_df, test_df = train_test_split(dummy_df, train_size=database_split, shuffle=True, random_state=123, stratify=strat)
print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
balance = list(train_df['labels'].value_counts())
print(balance)


train_df = trim(train_df, max_size, min_size, 'labels')

test_df = trim(test_df, max_size, min_size, 'labels')
valid_df = trim(valid_df, max_size, min_size, 'labels')

working_dir = r'./'
aug_dir = os.path.join(working_dir, 'aug')
if os.path.isdir(aug_dir):
    shutil.rmtree(aug_dir)
os.mkdir(aug_dir)
for label in train_df['labels'].unique():
    dir_path = os.path.join(aug_dir, label)
    os.mkdir(dir_path)

target = max_size  # set the target count for each class in df
gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                         height_shift_range=.2, zoom_range=.2)
groups = train_df.groupby('labels')  # group by class
for label in train_df['labels'].unique():  # for every class
    group = groups.get_group(label)  # a dataframe holding only rows with the specified label
    sample_count = len(group)  # determine how many samples there are in this class
    if sample_count < target:  # if the class has less than target number of images
        aug_img_count = 0
        delta = target - sample_count  # number of augmented images to create
        target_dir = os.path.join(aug_dir, label)  # define where to write the images
        aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=(336, 336), class_mode=None,
                                          batch_size=1, shuffle=False, save_to_dir=target_dir, save_prefix='aug-',
                                          save_format='jpg')
        while aug_img_count < delta:
            images = next(aug_gen)
            aug_img_count += len(images)

aug_list = os.listdir(aug_dir)
for d in aug_list:
    dpath = os.path.join(aug_dir, d)
    size = len(os.listdir(dpath))
    print(f'{d:25s}   {size}')

aug_fpaths = []
aug_labels = []
classlist = os.listdir(aug_dir)
for klass in classlist:
    classpath = os.path.join(aug_dir, klass)
    flist = os.listdir(classpath)
    for f in flist:
        fpath = os.path.join(classpath, f)
        aug_fpaths.append(fpath)
        aug_labels.append(klass)
Fseries = pd.Series(aug_fpaths, name='filepaths')
Lseries = pd.Series(aug_labels, name='labels')
aug_df = pd.concat([Fseries, Lseries], axis=1)
ndf = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
print(list(ndf['labels'].value_counts()))

length = len(test_df)
test_batch_size = \
sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)
print('test batch size: ', test_batch_size, '  test steps: ', test_steps)

def scalar(img):
    return img  # EfficientNet expects pixels in range 0 to 255 so no scaling is required

# initial ImageDataGenerators
trgen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)

tvgen = ImageDataGenerator(preprocessing_function=scalar)

# tie augmentation to their respective dataset split
train_gen = trgen.flow_from_dataframe(ndf, x_col='filepaths', y_col='labels', target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)

valid_gen = tvgen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)

# no shuffling for test_gen
test_gen = tvgen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                     class_mode='categorical',
                                     color_mode='rgb', shuffle=False, batch_size=test_batch_size)

classes = list(train_gen.class_indices.keys())  # construct list of class labels
class_count = len(classes)
train_steps = np.ceil(len(train_gen.labels) / batch_size)   # i.e. amount of batches

# build the layers
x = base_model.output
x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dropout(rate=.45, seed=123)(x)
output = Dense(class_count, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# uncomment below line if you need to load an already-trained model from a pb directory
# model = tf.keras.models.load_model('colab_trained_model.pb')
# if you only have h5, run convert_h5_to_pb.py to get a pb directory, then point to it.

model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 40         # 'max' limit of epochs. can't hit this level without running training for days.
patience = 1        # number of epochs to wait to adjust learning rate if accuracy does not improve
stop_patience = 3   # number of epochs to wait before stopping training if monitored value does not improve
threshold = .9      # if train accuracy is < threshold, adjust monitor accuracy, else monitor validation loss
factor = .5         # factor to reduce learning rate
dwell = True        # experimental, reset back to previous weight if accuracy does not improve
ask_epoch = 1       # number of epochs to run before prompting for decision

batches = train_steps   # (size of dataset) / (given batch size)

checkpointer = ModelCheckpoint(filepath='EN_model_checkpoint.hdf5', verbose=1, save_best_only=True)
# instantiated but not added to callbacks.
# saving takes some time depending on size of dataset

callbacks = [
    learning_rate_adjustor(model=model, base_model=base_model, patience=patience, stop_patience=stop_patience, threshold=threshold,
                           factor=factor, dwell=dwell, batches=batches, initial_epoch=0, epochs=epochs, ask_epoch=ask_epoch)]

history = model.fit(x=train_gen, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=valid_gen,
                    validation_steps=None, shuffle=False, initial_epoch=0)

tr_plot(history, 0)
save_dir = './'
subject = 'nutricare'
acc = model.evaluate(test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps, return_dict=False)[1] * 100
msg = f'accuracy on the test set is {acc:5.2f} %'
print_in_color(msg, (0, 255, 0), (55, 65, 80))
save_id = str(model_name + '-' + subject + '-' + str(acc)[:str(acc).rfind('.') + 3] + '.h5')
save_loc = os.path.join(save_dir, save_id)
model.save(save_loc)
generator = train_gen
scale = 1

print("Saving Model:")

model_save_loc, csv_save_loc = save_model(save_dir, model, model_name, subject, acc, img_size, scale, generator)
# save the trained model and class_dict file

print_code = 0
preds = model.predict(test_gen)

print("Printing Confusion Matrix :")

print_info(test_gen, preds, print_code, save_dir, subject)

print("Model training ended, files saved.")