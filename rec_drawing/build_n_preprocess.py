import numpy as np
from keras import Sequential
from keras.layers import Dense,Flatten,Dropout,MaxPooling2D,Conv2D

#Adding labels
cat = np.c_[cat, np.zeros(len(cat))]
giraffe = np.c_[giraffe, np.ones(len(giraffe))]
sheep = np.c_[sheep, 2*np.ones(len(sheep))]
bat = np.c_[bat, 3*np.ones(len(bat))]
octopus = np.c_[octopus, 4*np.ones(len(octopus))]
camel = np.c_[camel, 5*np.ones(len(camel))]

# Merging arrays and splitting the features and labels
X = np.concatenate((cat[:10000,:-1], giraffe[:10000,:-1], sheep[:10000,:-1], bat[:10000,:-1],\
                    octopus[:10000,:-1], camel[:10000, :-1]), axis=0).astype('float32') # all columns but the last
y = np.concatenate((cat[:10000,-1], giraffe[:10000,-1], sheep[:10000,-1], bat[:10000,-1], \
                    octopus[:10000,-1],  camel[:10000,-1]), axis=0).astype('float32') # the last column

# We then split data between train and test (80 - 20 usual ratio). Also normalizing the value between 0 and 1
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2,random_state=0)

# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

def cnn_model():
    model=Sequential()
    model.add(Conv2D(30,(3,3),input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
import pickle
with open('model_cnn.pkl','wb') as file:
    pickle.dump(cnn_model(),file)