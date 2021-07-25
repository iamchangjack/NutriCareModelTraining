import tensorflow as tf
model = tf.keras.models.load_model('NutriCareModel_18_07.h5')
model.save('NutriCareModel_18_07')

model = tf.keras.models.load_model('NutriCareModel_18_07')
model.summary()

print('done')