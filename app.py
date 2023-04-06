import streamlit as st
from keras.models import load_model
import numpy as np
import io
from PIL import Image


#"""Load model once at running time for all the predictions"""
print('[INFO] : Model loading ................')
global model
model = load_model('cat_dog_classifier.h5')
print('[INFO] : Model loaded')

st.title('What is this image? :cat: :dog:')
global data

uploaded_file = st.file_uploader("Upload a file to classify")
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    img = Image.open(io.BytesIO(bytes_data))
    data = img.resize((128, 128), Image.ANTIALIAS)
    st.image(img, caption='Image to be classified')
    
def predict():
    global data
    data = np.expand_dims(data, axis=0)

    # Scaling
    data = data.astype('float') / 255 

    # Prediction
    result = model.predict(data)

    pred_prob = result.item()

    if pred_prob > .5:
        label = 'Dog'
        accuracy = round(pred_prob * 100, 2)
    else:
        label = 'Cat'
        accuracy = round((1 - pred_prob) * 100, 2)
       
    st.success('This is a ' + label + ' predicted with confidence ' + str(accuracy))

trigger = st.button('Predict', on_click=predict)
               
