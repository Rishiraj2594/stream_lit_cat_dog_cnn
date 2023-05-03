import streamlit as st
from keras.models import load_model
import numpy as np
import io
from PIL import Image
# pip install opencv-python
# import cv2


#"""Load model once at running time for all the predictions"""
print('[INFO] : Model loading ................')
global model
model_unet = load_model('unet.h5')
print('[INFO] : Model loaded')

st.title('Lane Segmentation')
global data
seg_type = st.radio("Model Type", ('Unet', 'Vgg-FCN'))
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    img = Image.open(io.BytesIO(bytes_data))
    data = img.resize((128, 128), Image.ANTIALIAS)
    st.image(img, caption='orginal image')
    
# Function to calculate mask over image
def weighted_img(img, initial_img, α=1., β=0.5, γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

# Function to process an individual image and it's mask
def process_image_mask(image, mask):
    # Round to closest
    mask = np.round(mask)
    
    # Convert to mask image
    zero_image = np.zeros_like(mask)
    mask = np.dstack((mask, zero_image, zero_image))
    mask = np.asarray(mask, np.float32)
    
    # Convert to image image
    image = np.asarray(image, np.float32)
    
    # Get the final image
    final_image = weighted_img(mask, image)

    return final_image    

def predict():
    global data
    data = np.expand_dims(data, axis=0)

    # Scaling
    data = data.astype('float') / 255 
    
    if seg_type == "Unet":
        pred_mask = model_unet.predict(data)
        result = process_image_mask(data[0], pred_mask[0]))
        st.image(result, caption='mask of image')
        st.image(data, caption=' image without mask')
#     else:
#         pred_mask = model_vgg_fcn.predict(data)
#         st.image(pred_mask, caption='image with image')
#         st.image(data, caption=' image without mask')
    st.success('This is a  masked image')

trigger = st.button('Predict', on_click=predict)
               
