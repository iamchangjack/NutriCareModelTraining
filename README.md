Side repository for the model and dataset of the [NutriCare project](https://github.com/wangzksit/NutriCare).

Code based off robert-kamunde's [Food 101 Project](https://github.com/robert-kamunde/Food101Project)

# TO TRAIN THE MODEL

- Run Food101Model.py to tune and train the model ( EfficientNetB2 is used here). This is a lengthy process and may overwrite the existing (and partially) trained model.

# TO USE THE MODEL TO PREDICT IMAGES

A partially trained model (trained on 10 batches over 20 epochs) has already been created. 

- Run Prediction_food_images.py to predict new food images. ( the images to be predicted here were in the working directory, 6 images (char_siew_rice, dumplings, fried_rice, kway_chap, lor_mee and murtabak) are provided to demo. They are images that Jack took by himself so they shouldn't be a duplicate of an image in the test/train folders)

