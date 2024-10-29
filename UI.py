import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model_path = './checkpoint/dermavision-densenet201.h5'
# Load your trained model
model = tf.keras.models.load_model(model_path)

# Define image size used for the model (modify as per your model)
IMG_SIZE = 224  # Example for models like DenseNet, ResNet, etc.

# Function to preprocess the input image
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize image to model's expected size
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Streamlit app layout
st.set_page_config(page_title="Image Classification", layout="wide")
st.title("🩺 DermalAI: Skin disease predictor")
st.write("Upload an image for prediction:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image.resize(size=(500,500)), caption='Uploaded Image', use_column_width=False)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)

    # Class labels
    class_labels = [
        'Actinic keratoses',
        'Basal Cell Carcinoma',
        'Benign keratosis-like lesions',
        'Dermatofibroma',
        'Melanoma',
        'Melanocytic nevi',
        'Vascular lesions'
    ]
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")

# Add a footer for additional information
st.markdown("<hr>", unsafe_allow_html=True)
st.write("This application classifies/predict skin lesions.")

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f5;  /* Light gray background */
    }
    </style>
    """,
    unsafe_allow_html=True
)
