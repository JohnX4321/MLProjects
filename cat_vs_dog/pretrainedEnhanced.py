# Load in the old pretrained model.
from keras.models import load_model

model = load_model('catDogPretrained.h5')

# Test how many layers there are in our model
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# Freeze some of the layers
for layer in model.layers[:298]:
    layer.trainable = False

# Allow these layers to train
for layer in model.layers[298:]:
    layer.trainable = True

# Recompile the model, use adam with a low LR.
from keras.optimizers import adam

model.compile(optimizer=adam(lr=0.001), loss='binary_crossentropy')

# Fit the model
model.fit_generator(training_set, steps_per_epoch=25, epochs=5, validation_data=test_set,
                    validation_steps=25)
# Save the model
model.save('catDogPretrainedEnhanced.h5')

#major change