import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

@st.cache(allow_output_mutation=True)
def load_model(path='../models/inception_v3'):
    """Retrieves the trained model"""
    model = keras_load_model(path)
    return model

@st.cache()
def load_list_of_images_available(all_image_files, image_files_dtype):
    """Retrieves list of available images given the current selections"""
    list_of_files = all_image_files.get(image_files_dtype)
    return list_of_files

if __name__ == '__main__':

    st.title('Welcome To Crop Health Detection App!')
    instructions = """
        Either upload your own image or select from the sidebar to get a preconfigured image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file:
        img = Image.open(file)
    else:
        image_name = st.sidebar.selectbox("Sample Images", os.listdir("img"))
        img = Image.open(os.path.join("img", image_name))
    
    model_dir = "../models/"
    #model_name = st.sidebar.selectbox("Models", os.listdir(model_dir))
    model_name = "inception_v3"
        
    input_img = img.resize((128,128))

    st.title("Here is the image you've selected")
    resized_image = img.resize((250,250))
    st.image(resized_image)

    # Fallback for tf.__version__ < 2.3
    # efficientnet was available from v2.3 
    if model_name == "efficientnet_b4" or model_name == "resnet50":
        model_name = "inception_v3"
    model_path = os.path.join(model_dir, model_name)

    model = load_model(model_path)
    input_img=tf.data.Dataset.from_tensors([np.array(input_img)])

    prediction = model.predict(input_img)
    # st.write(prediction)

    if prediction[0][0] > 0.5:
        st.title(f"Unhealthy Crop Confidence: {prediction[0][0]*100:.2f}")
    else:
        st.title(f"Healthy Crop Confidence: {prediction[0][1]*100:.2f}")

