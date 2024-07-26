# vit_script.py

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Define preprocessing function
def image_processor(image_path, target_size):
    """Preprocess images for ViT model"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = 255 - image
    image = image.astype(np.float32) / 255.0
    return image

# Load and preprocess data
def load_and_preprocess_data():
    # Read metadata
    df_meta = pd.read_csv('/path/to/meta.csv')
    df_dicom = pd.read_csv('/path/to/dicom_info.csv')
    
    # Filter image paths
    imdir = '/path/to/jpeg'
    full_mammo = df_dicom[df_dicom.SeriesDescription == 'full mammogram images'].image_path
    full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', imdir, regex=True)

    # Load dataset
    mass_train = pd.read_csv('/path/to/mass_case_description_train_set.csv')
    mass_test = pd.read_csv('/path/to/mass_case_description_test_set.csv')

    # Fix image paths
    def fix_image_path(data):
        for index, img in enumerate(data.values):
            img_name = img[11].split("/")[2]
            data.iloc[index, 11] = full_mammo_dict.get(img_name, img[11])
    
    fix_image_path(mass_train)
    fix_image_path(mass_test)
    
    # Merge datasets
    full_mass = pd.concat([mass_train, mass_test], axis=0)

    # Process images
    target_size = (224, 224, 3)
    full_mass['processed_images'] = full_mass['image_file_path'].apply(lambda x: image_processor(x, target_size))
    
    # Map labels
    class_mapper = {'MALIGNANT': 1, 'BENIGN': 0}
    full_mass['labels'] = full_mass['pathology'].replace(class_mapper)
    
    # Convert images to array
    X_resized = np.array(full_mass['processed_images'].tolist())
    
    # Convert labels to integer format
    le = LabelEncoder()
    y_labels = le.fit_transform(full_mass['labels'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resized, y_labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Convert data to PyTorch tensors
def convert_to_tensors(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    return TensorDataset(X_train_tensor, y_train_tensor), TensorDataset(X_test_tensor, y_test_tensor)

# Build and compile ViT model
def build_vit_model():
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)
    return model

# Train the model
def train_model(model, train_loader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Main execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    train_dataset, test_dataset = convert_to_tensors(X_train, X_test, y_train, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = build_vit_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_model(model, train_loader, optimizer, criterion, num_epochs=5)
    evaluate_model(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), '/path/to/vit.pth')
