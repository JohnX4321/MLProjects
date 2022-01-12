#analyticsvidya
import numpy as np
import matplotlib.pyplot as plt
import glob,cv2,os,random
import matplotlib.image as mpimg
import tensorflow.keras as keras
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.layers import *

SIZE=256
SEED_TRAINING=121
SEED_TESTING=197
SEED_VALIDATION=164
CHANNELS=3
N_CLASSES=3
EPOCHS=50
BATCH_SIZE=16
INPUT_SHAPE=(SIZE,SIZE,CHANNELS)

train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_gen=train_datagen.flow_from_directory(directory='train',target_size=(256,256),batch_size=BATCH_SIZE,class_mode='categorical',color_mode='rgb')
valid_gen=validation_datagen.flow_from_directory(directory='valid',target_size=(256,256),batch_size=BATCH_SIZE,class_mode='categorical',color_mode='rgb')
test_gen=test_datagen.flow_from_directory(directory="test",target_size=(256,256),batch_size=BATCH_SIZE,class_mode="categorical",color_mode="rgb")

model=keras.models.Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=INPUT_SHAPE),
    MaxPooling2D((2,2)),
    Dropout(0.5),
    Conv2D(64,(3,3),activation="relu",padding="same"),
    MaxPooling2D((2,2)),
    Dropout(0.5),
    Conv2D(64,(3,3),activation="relu",padding="same"),
    MaxPooling2D((2,2)),
Conv2D(64,(3,3),activation="relu",padding="same"),
    MaxPooling2D((2,2)),
Conv2D(64,(3,3),activation="relu",padding="same"),
    MaxPooling2D((2,2)),
Conv2D(64,(3,3),activation="relu",padding="same"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32,activation="relu"),
    Dense(N_CLASSES,activation="softmax")
])

model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(),metrics=["accuracy"])
model.save(filepath="model.h5")
history=model.fit_generator(train_gen,steps_per_epoch=train_gen.n,epochs=EPOCHS,validation_data=valid_gen,validation_steps=valid_gen.n)
