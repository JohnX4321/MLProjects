import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


# A callback to stop our tranning

# when reaching enough accuracy

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        # If you are using Tensorflow 1.x, replace 'accuracy' for 'acc' in the next line

        if logs.get('acc') > 0.99:
            print("\nReached 99.0% accuracy so cancelling training!")

            self.model.stop_training = True


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=25, callbacks=[myCallback()])

model.evaluate(x_test, y_test)

model.save('models/my_mnist_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model_file('models/my_mnist_model.h5');
tflite_model = converter.convert()
open("models/converted_mnist_model.tflite", "wb").write(tflite_model)
