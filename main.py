import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import keras
from tensorflow.keras import layers


data_train_path='/content/drive/MyDrive/Fruits_Vegetables/train'
data_test_path='/content/drive/MyDrive/Fruits_Vegetables/test'
data_validation_path='/content/drive/MyDrive/Fruits_Vegetables/validation'


img_t=180
img_a=180

data_train= tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    batch_size=32,
    image_size=(img_t,img_a),
    validation_split=False
)

data_cat=data_train.class_names

data_test= tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    shuffle=False,
    batch_size=32,
    image_size=(img_t,img_a),
    validation_split=False
)

data_validation= tf.keras.utils.image_dataset_from_directory(
    data_validation_path,
    shuffle=False,
    batch_size=32,
    image_size=(img_t,img_a),
    validation_split=False
)



model=keras.Sequential([layers.Rescaling(1./255)])
model.add(keras.layers.Conv2D(16,(3,3)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(32,(3,3)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(64,(3,3)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=128,activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(units=len(data_cat)))

model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
hist=model.fit(data_train,epochs=8,validation_data=data_validation)

model.evaluate(data_test)

import cv2
path="/content/drive/MyDrive/Fruits_Vegetables/test/tomato/Image_5.jpg"
test_img=cv2.imread(path)
test_img=cv2.resize(test_img,(img_a,img_t))
test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
test_img_final=np.expand_dims(test_img,0)
prediction=model.predict(test_img_final)

score=tf.nn.softmax(prediction)
plt.imshow(test_img)
print(data_cat[np.argmax(score)],"  -----  ",np.max(score))
