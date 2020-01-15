from mnist import MNIST
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.callbacks import ModelCheckpoint

emnist_data=MNIST(path='data\\',return_type='numpy')
emnist_data.select_emnist('letters')
X,y=emnist_data.load_training()


X=X.reshape(124800,28,28)
y=y.reshape(125800,1)

y=y-1
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=111)

X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

y_train=np_utils.to_categorical(y_train,num_classes=26)
y_test=np_utils.to_categorical(y_test,num_classes=26)

model=Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


score=model.evaluate(X_test,y_test,verbose=0)
accuracy=score[1]*100
print('Before Training - Test accuracy: %.4f%%' % accuracy)



# Train the model

checkpointer = ModelCheckpoint(filepath='emnist.model.best.hdf5',

                               verbose=1, save_best_only=True)

hist = model.fit(X_train, y_train, batch_size=128, epochs=10,

          validation_split=0.2, callbacks=[checkpointer],

          verbose=1, shuffle=True)



# Load the Model with the Best Classification Accuracy on the Validation Set

model.load_weights('emnist.model.best.hdf5')



# Save the best model

model.save('eminst_mlp_model.h5')



# Evaluate test accuracy

score = model.evaluate(X_test, y_test, verbose=0)

accuracy = 100*score[1]

print('Test accuracy: %.4f%%' % accuracy)


