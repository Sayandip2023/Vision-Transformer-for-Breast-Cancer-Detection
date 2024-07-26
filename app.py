# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained model
model = load_model("C:/Users/sayan/Downloads/vgg16.h5")

# Define the paths to the two images
image_paths = {
    "Image 1": "C:/Users/sayan/Downloads/Screenshot_2024-07-26_104257.png",
    "Image 2": "C:/Users/sayan/Downloads/Screenshot_2024-07-26_104109.png"
}

# Define the function to preprocess the image
def preprocess_image(image):
    # Convert the image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))  # Resize the image to 224x224 pixels
    image_array = np.array(image)  # Convert image to array
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define the main function to run the Streamlit app
def main():
    st.title("Image Classification with VGG16")

    # Create a 2-column layout for image selection
    col1, col2 = st.columns(2)

    # Display the images in columns with reduced width
    with col1:
        st.image(image_paths["Image 1"], caption="Image 1", width=150)
    with col2:
        st.image(image_paths["Image 2"], caption="Image 2", width=150)

    # Allow user to select an image
    image_selection = st.radio("Select an image to classify:", list(image_paths.keys()))
    
    if image_selection in image_paths:
        # Load and display the selected image with reduced width
        image = Image.open(image_paths[image_selection])
        st.image(image, caption=f'Selected Image: {image_selection}', width=300)
        st.write("")

        # Preprocess the image
        image_array = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming the output is one-hot encoded

        # Display the prediction
        st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
