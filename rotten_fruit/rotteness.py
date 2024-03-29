import datetime
import os
import time
from pathlib import Path

from keras import backend as K
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#%% set configuration
img_width, img_height = 32, 32
category = os.environ.get("FRUIT_CATEGORY", "oranges")

train_data_dir = "dataset/" + category
batch_size = 15

if K.image_data_format() == "channels_first":
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#%% create paths
results_path = Path("results") / category
results_path.mkdir(parents=True, exist_ok=True)

#%% create training and set set
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
)

test_set = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode="binary",
    subset="validation",
)


training_set = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode="binary",
    subset="training",
)


#%% create model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(32, activation="relu"))

model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(
    loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)
model.summary()
with open(results_path / "model.yaml", "w") as f:
    f.write(model.to_yaml())

#%% train it!
t1 = time.time()
print(datetime.datetime.now())
model.fit_generator(
    training_set,
    steps_per_epoch=2000,
    epochs=10,
    validation_data=test_set,
    validation_steps=200,
    verbose=1,
)
print("Training took %s seconds" % (time.time() - t1))

#%% save weights to file
model.save_weights(results_path / f"weights.h5")

#%% evaluate model
print(dict(
    zip(
        model.metrics_names,
        model.evaluate_generator(test_set, verbose=True, steps=1500),
    )
))