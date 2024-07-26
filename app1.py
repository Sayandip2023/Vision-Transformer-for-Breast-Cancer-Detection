import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import ViTForImageClassification
import time

# Define the device to be used for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Vision Transformer model
model_path = 'C:/Users/sayan/Downloads/Vision_Transformer.pth'
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the paths to the two images
image_paths = {
    "Image 1": "C:/Users/sayan/Downloads/Screenshot 2024-07-26 104257.png",
    "Image 2": "C:/Users/sayan/Downloads/Screenshot 2024-07-26 104109.png"
}

# Preprocess image function
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))  # Resize the image to 224x224 pixels
    image_array = np.array(image)  # Convert image to array
    image_array = image_array / 255.0  # Normalize pixel values
    image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)  # Change to (C, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

# Predict function
def predict(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        start_time = time.time()  # Start the timer
        outputs = model(image_tensor).logits
        _, predicted = torch.max(outputs, 1)
        end_time = time.time()  # End the timer
        inference_time = end_time - start_time  # Calculate the inference time
    return predicted.item(), inference_time

# Streamlit app
def main():
    st.title("Image Classification with Vision Transformer")

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
        st.image(image, caption=f'Selected Image: {image_selection}', width=150)
        st.write("")

        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Make prediction and measure inference time
        prediction, inference_time = predict(image_tensor, model, device)

        # Display the prediction
        if prediction == 0:
            result_message = "The model predicts this image is benign."
        else:
            result_message = "The model predicts this image is malignant."

        st.write(f"Predicted class: {prediction}")
        st.write(result_message)
        st.write(f"Inference time: {inference_time:.4f} seconds")

if __name__ == "__main__":
    main()
