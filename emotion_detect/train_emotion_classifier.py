from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from load_and_process import load_fer2013

from load_and_process import preprocess_input

from models.cnn import mini_XCEPTION

from sklearn.model_selection import train_test_split


batch_size=32
num_epochs=10000
input_shape=(48,48,1)
validation_split=.2
verbose=1
num_classes=7
patience=50
base_path='models/'


data_generator=ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True
)


model=mini_XCEPTION(input_shape,num_classes)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()