# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from matplotlib.cbook import file_requires_unicode
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
#loading the model
#model_path = "C:/Users/Houda/Documents/OpenClassrooms/P6/Model/"
model = load_model('C:/Users/Houda/Documents/OpenClassrooms/P6/Model/model-VGG16-03-0.99.h5')

#model = load_model('C:/Users/Houda/Documents/OpenClassrooms/P6/Model/model.h5')

CLASS_NAME = ['Chihuahua', 'Japanese spaniel', 'Maltese dog']

st.title('Dog Breed Prediction')
st.markdown('Submit a dog image')
dog_image = st.file_uploader('Choose an image ...', type='jpg')
submit = st.button('Predict')

if submit:
  if dog_image is not None:
    file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes,1)
    st.image(opencv_image, channels='BGR')
    opencv_image = cv2.resize(opencv_image, (224,224))
    opencv_image.shape = (1,224,224,3)
    y_pred = model.predict(opencv_image)
    st.title(str("The dog breed is " + CLASS_NAME[np.argmax(y_pred)]))
