import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub

hide_streamlit_style="""
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
"""

st.markdown(hide_streamlit_style,unsafe_allow_html=True)
st.title('PLDP')

def main():
    file_uploaded=st.file_uploader('Choose an image...',type='jpg')
    if file_uploaded is not None:
        img=Image.open(file_uploaded)
        st.write("Uploaded Image...")
        figure=plt.figure()
        plt.imshow(img)
        plt.axis("off")
        st.pyplot(figure)
        res,c=predict(img)
        st.write("Prediction: {}".format(res))
        st.write("Confidence: {}%".format(c))

def predict(img):
    with st.spinner("Loading..."):
        classifier=keras.models.load_model("model.h5",compile=False)
    shape=((256,256,3))
    model=keras.Sequential([hub.KerasLayer(classifier,input_shape=shape)])
    test=img.resize((256,256))
    test=keras.preprocessing.image.img_to_array(test)
    test/=255.0
    test=np.expand_dims(test)
    class_name=['Potato_Early_blight','Potato_Late_blight','Potato_Healthy']
    pred=model.predict(test)
    c=round(100*(np.max(pred[0])),2)
    return pred,c