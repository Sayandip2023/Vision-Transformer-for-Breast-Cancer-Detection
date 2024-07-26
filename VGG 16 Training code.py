# vgg16_script.py

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Define preprocessing function
def image_processor(image_path, target_size):
    """Preprocess images for CNN model"""
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resized, full_mass['labels'].values, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Build and compile VGG16 model
def build_vgg16_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3)
    history = model.fit(X_train, y_train, epochs=30, validation_split=0.1, batch_size=64, callbacks=[reduce_lr])
    return history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

# Main execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = build_vgg16_model(input_shape=(224, 224, 3))
    history = train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    model.save('/path/to/vgg16.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.show()
