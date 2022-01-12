from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,Activation,Flatten,Dense,Reshape,Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np


class ConvAutoEncoder:
    @staticmethod
    def build(width,height,depth,filters=(32,64),latentDim=16):
        inputShape=(height,width,depth)
        chanDim=-1
        inputs=Input(shape=inputShape)
        x=inputs
        for f in filters:
            x=Conv2D(f,(3,3),strides=2,padding="same")(x)
            x=LeakyReLU(alpha=0.2)(x)
            x=BatchNormalization(axis=chanDim)(x)
        volSize=K.int_shape(x)
        x=Flatten()(x)
        latent=Dense(latentDim)(x)
        encoder=Model(inputs,latent,name="encoder")
        latentInp=Input(shape=(latentDim,))
        x=Dense(np.prod(volSize[1:]))(latentInp)
        x=Reshape((volSize[1],volSize[2],volSize[3]))(x)
        for f in filters[::-1]:
            x=Conv2DTranspose(f,(3,3),strides=2,padding="same")(x)
            x=LeakyReLU(alpha=0.2)(x)
            x=BatchNormalization(axis=chanDim)(x)
        x=Conv2DTranspose(depth,(3,3),padding="same")(x)
        outputs=Activation("sigmoid")(x)
        decoder=Model(latentInp,outputs,name="decoder")
        autoencoder=Model(inputs,decoder(encoder(inputs)),name="autoencoder")
        return (encoder,decoder,autoencoder)

