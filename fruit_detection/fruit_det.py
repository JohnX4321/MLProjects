import sys,numpy as np,math,cv2
from keras import applications,regularizers
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

np.random.seed(123)

img_width,img_height=224,224
top_model_wt_path='bottleneck_fc_model.h5'
train_data_dir='data/train'
batch_size=32
valid_data_dir='data/valid/'

def save_bottleneck_features():
    model=applications.VGG16(include_top=False,weights='imagenet')
    datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    gen=datagen.flow_from_directory(train_data_dir,target_size=(img_width,img_height),batch_size=batch_size,shuffle=False)
    no_train_samp=len(gen.filenames)
    predict_size_train=int(math.ceil(no_train_samp/batch_size))
    bottleneck_ft_train=model.predict_generator(gen,predict_size_train)
    np.save('bottle_ft.npy',bottleneck_ft_train)

def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1./255)

    generator_top = datagen_top.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

    num_classes = len(generator_top.class_indices)

    # save the class indices to use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(validation_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode=None,
                                                    shuffle=False)

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')
    validation_data = np.load('bottleneck_features_validation.npy')

    # build the model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    eval_loss, eval_accuracy = model.evaluate(validation_data, validation_labels,
                                                batch_size=batch_size, verbose=1)

    print("\n")
    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def predict(image_path):
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # image_path = 'test/cherry/1056.jpg'

    orig = cv2.imread(image_path)

    print('[INFO] loading and preprocessing image...')
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    model = applications.VGG16(include_top=False, weights='imagenet')

    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    print("Image ID: {}, Label: {}".format(inID, label))

    cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    cv2.imshow('Classification', orig)
    cv2.imwrite('predicted.jpg', orig)
    cv2.waitKey(0)


if __name__ == '__main__':
    # save_bottleneck_features()
    # train_top_model()
    image_path = sys.argv[1]
    predict(image_path)