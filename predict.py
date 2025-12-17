import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Configuration
IMG_SIZE = 150
MODEL_PATH = os.path.join('models', 'pneumonia_best_model.keras')
TEST_DIR = os.path.join('data', 'test')

# Load the trained model once
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

def predict_image(file_path):
    # Load and preprocess image
    img = keras.utils.load_img(file_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.utils.img_to_array(img)
    img_array = img_array / 255.0   # normalizing it
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension

    # Predict
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0]

    if score > 0.50:
        return "PNEUMONIA", score * 100, img
    else:
        return "NORMAL", score * 100, img
    
# Gather Random Images
