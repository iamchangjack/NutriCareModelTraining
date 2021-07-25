Side repository for the model training process of the [NutriCare project](https://github.com/wangzksit/NutriCare).

Code based off robert-kamunde's [Food 101 Project](https://github.com/robert-kamunde/Food101Project) and gpipsenka's [How to Balance a Dataset](https://www.kaggle.com/gpiosenka/how-to-balance-a-dataset). Many thanks to them.

# TO TRAIN THE MODEL

- Run Efficient_FoodSG_Model.py  to tune and train the model (as the name implies EfficientNetB2 is used here).
- Prior to running, ensure that the dataset (should be FoodSG, but any other dataset will work) is completely unzipped in the same running directory. 

# TO SET UP THE DATASET
- Download the FoodSG dataset on Kaggle. Any other dataset also works fine.
- Unzip the dataset contents into .../NutriCareModelTraining
- The 'foodsg' folder should be accessible from .../NutriCareModelTraining.
- Upon opening the 'foodsg' folder, class folders (e.g. 'chicken_rice') containing images should be immediately accessible.

# TO USE THE MODEL TO PREDICT IMAGES

A partially trained model, EN_Model.h5 (trained on 10 batches over 20 epochs) has already been created for inferring.

- Run prediction_alternative.py to predict new food images.
- The images to be predicted here are in the working directory. For your own images, you should edit the image path directly within prediction.alternative.py

# IMAGE SCRAPPING
Image scrapping is done via a slightly edited Python script hosted on https://github.com/iamchangjack/google-images-downloadPATCHED. All credits for the image scrapping go to those involved in the [original project](https://github.com/hardikvasa/google-images-download)
