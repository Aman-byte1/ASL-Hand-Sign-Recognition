import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

# Load your trained model (adjust the path as needed)
MODEL_PATH = r"C:\Users\dell\Desktop\fetera\asl_model.h5"
model = load_model(MODEL_PATH)

# Define the labels corresponding to the classes
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "T", "U", "V", "W", "X", "Y", "Z"]

def preprocess_frame(frame):
    """
    Resize the frame to the dimensions expected by your model,
    normalize pixel values, and expand dimensions.
    """
    # Resize to 100x100 (change if your training data uses a different size)
    processed = cv2.resize(frame, (100, 100))
    # Normalize pixel values to [0, 1]
    processed = processed.astype('float32') / 255.0
    # Expand dimensions to create batch of 1
    processed = np.expand_dims(processed, axis=0)
    return processed

def predict_sign(frame):
    """
    Run the ASL hand sign model on the frame and return the predicted label.
    """
    processed = preprocess_frame(frame)
    predictions = model.predict(processed)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

st.title("ASL Hand Sign Recognition")

# Select mode: Image Upload
st.markdown("**Upload an image file**")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file as a numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Get prediction
    predicted = predict_sign(frame)

    # Display the prediction on the image
    cv2.putText(frame, f'Predicted: {predicted}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR to RGB before displaying with Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f"Prediction: {predicted}", use_column_width=True)