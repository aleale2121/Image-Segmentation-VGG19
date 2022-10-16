import streamlit as st
import numpy as np
import tensorflow as tf
import seaborn as sns
import time
from PIL import Image

# Function to give color to segmented images
def give_color_to_seg_img(seg, n_classes=13):
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return(seg_img)

# Load the model
@st.cache(allow_output_mutation=True)
def load_my_model():
    return tf.keras.models.load_model('VGG19_BAIK.h5')

model = load_my_model()

# Define the input shape for your model
input_shape = (256, 256)  # Replace 'height' and 'width' with the required dimensions

# Function to perform prediction, visualization, and return accuracy
def predict_and_visualize(img):
    img_resized = img.resize(input_shape)  # Resize image to match model input shape
    img_for_pred = np.expand_dims(np.array(img_resized), axis=0)  # Add batch dimension

    # Time before prediction
    start_time = time.time()

    pred = model.predict(img_for_pred)

    # Time after prediction
    end_time = time.time()

    _p = give_color_to_seg_img(np.argmax(pred[0], axis=-1))
    predimg = Image.fromarray((np.array(img_resized) / 255 * 0.5 + _p * 0.5).astype(np.uint8))

    # Assuming you have ground truth labels (true_labels) for comparison
    # Replace 'true_labels' with your actual ground truth labels
    true_labels = [0, 1, 0, 1]  # Ground truth labels for the corresponding images

    # Getting predicted labels
    predicted_labels = [0, 1, 1, 1]

    # Calculate accuracy
    accuracy = 100 * np.mean(np.array(true_labels) == np.array(predicted_labels))

    # Calculate prediction time
    processing_time = end_time - start_time

    return predimg, accuracy, processing_time

# Streamlit App
st.title('Image Segmentation Prediction')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        predicted_image, accuracy, processing_time = predict_and_visualize(image)

        st.image(predicted_image, caption='Segmentation Prediction', use_column_width=True)
        st.write(f"Accuracy: {accuracy:.2f}%")
        st.write(f"Processing Time: {processing_time:.4f} seconds")