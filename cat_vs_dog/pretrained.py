from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

base=InceptionV3(weights='imagenet',include_top=False)
x=base.output
x=GlobalAveragePooling2D()(x)

#fully connlayer
x=Dense(1024,activation='relu')(x)
#classify
pred=Dense(1,activation='sigmoid')(x)
model=Model(inputs=base.input,outputs=pred)

#train top layer
for layer in base.layers:
    layer.trainable=False

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_set=train_datagen.flow_from_directory(r'CNN',target_size=(299,299),batch_size=32,class_mode='binary')

test_set=test_datagen.flow_from_directory(r'Convolutional_Neural_Networks/dataset/test_set',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'binary')
model.fit_generator(train_set,steps_per_epoch=25,validation_data=test_set,validation_steps=10)
model.save('.h5')

predout=model.predict([])