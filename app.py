import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import seaborn as sns

# Function to give color to segmented images
def give_color_to_seg_img(seg, n_classes=13):
    
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

# Load the model
@st.cache(allow_output_mutation=True)
def load_my_model():
    return load_model('VGG19_BAIK.h5')

model = load_my_model()

# Define the input shape for your model
input_shape = (256, 256)  # Replace 'height' and 'width' with the required dimensions

# Function to perform prediction and visualization
def predict_and_visualize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)  # Resize image to match model input shape
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    pred = model.predict(img)

    _p = give_color_to_seg_img(np.argmax(pred[0], axis=-1))
    predimg = cv2.addWeighted(img[0] / 255, 0.5, _p, 0.5, 0)

    return predimg


# Streamlit App
st.title('Image Segmentation Prediction')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        predicted_image = predict_and_visualize(image)

        st.image(predicted_image, caption='Segmentation Prediction', use_column_width=True)