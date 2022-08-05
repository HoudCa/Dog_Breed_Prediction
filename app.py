# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:51:28 2022

@author: Houda
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2 as cv
import gradio as gr
from gradio.networking import INITIAL_PORT_VALUE, LOCALHOST_NAME

print(tf.__version__)

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

#f1_score(y_true, y_pred, average='weighted')

# Load model
#model_path = 'C:/Users/Houda/Documents/OpenClassrooms/P6/Dog_Heroku_New/'
model = load_model('model-VGG16.h5')
#, custom_objects={"f1_m": f1_m})
#model = tf.keras.applications.vgg16.VGG16()
breed_labels = ['Chihuahua', 'Japanese spaniel', 'Maltese dog']


# Define the full prediction function
def breed_prediction(inp):
    # Convert to RGB
    img = cv.cvtColor(inp,cv.COLOR_BGR2RGB)
    # Resize image
    dim = (224, 224)
    img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
    # Equalization
    img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    # Apply non-local means filter on test img
    dst_img = cv.fastNlMeansDenoisingColored(
        src=img_equ,
        dst=None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21)

    # Convert modified img to array
    img_array = tf.keras.preprocessing.image.img_to_array(dst_img)
    
    # Apply preprocess 
    img_array = img_array.reshape((-1, 224, 224, 3))
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    
    # Predictions
    prediction = model.predict(img_array).flatten()
    
    #return prediction
    return {breed_labels[i]: float(prediction[i]) for i in range(len(breed_labels))}

# Construct the interface
image = gr.inputs.Image(shape=(224,224))
label = gr.outputs.Label(num_top_classes=3)

iface = gr.Interface(
    fn=breed_prediction,
    inputs=image,
    outputs=label,
    capture_session=True,
    live=True,
    verbose=True,
    title="Dogs breed prediction from picture\nwith VGG16 model",
    allow_flagging=False,
    allow_screenshot=True,
    server_port=INITIAL_PORT_VALUE,
    server_name=LOCALHOST_NAME
)

if __name__ == "__main__":
    print("server_name:", LOCALHOST_NAME)
    print("server_port:", INITIAL_PORT_VALUE)

    # iface.launch(inbrowser=True)
    iface.launch()