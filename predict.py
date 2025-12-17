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
#model.summary() # check if model is good, comment after

def predict_image(file_path):
    # Load and preprocess image
    img = keras.utils.load_img(file_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.utils.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension

    # Predict
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0]
    threshold = 0.30    # we prioritize recall here since we are dealing with medical analysis

    if score > threshold:
        return "PNEUMONIA", score * 100, img
    else:
        return "NORMAL", (1 - score) * 100, img
    
# Gather 5 random images from each folder
normal_dir = os.path.join(TEST_DIR, 'NORMAL')
pneumonia_dir = os.path.join(TEST_DIR, 'PNEUMONIA')

normal_files = [os.path.join(normal_dir, f) for f in random.sample(os.listdir(normal_dir), 5)]
pneumonia_files = [os.path.join(pneumonia_dir, f) for f in random.sample(os.listdir(pneumonia_dir), 5)]
all_files = normal_files + pneumonia_files
random.shuffle(all_files)   # mix the files

# Plot results
plt.figure(figsize=(15, 8))
for i, file_path in enumerate(all_files):
    label, confidence, img = predict_image(file_path)

    # Get the actual value from the folder name
    actual = "PNEUMONIA" if "PNEUMONIA" in file_path else "NORMAL"

    # Color code: green if correct, red if wrong
    color = 'green' if label == actual else 'red'

    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {label} ({confidence:.1f}%)\nActual: {actual}", color=color)
    plt.axis('off')

plt.tight_layout()

# Allow saving multiple figures
save_dir = 'results'
base_name = 'prediction_results'
extension = '.png'
counter = 1

filename = f"{base_name}{extension}"    # start with the default name
save_path = os.path.join(save_dir, filename)

while os.path.exists(save_path):
    filename = f"{base_name}_{counter}{extension}"  # add _1, _2, ..., to end of file if exists
    save_path = os.path.join(save_dir, filename)
    counter += 1    # increase file by 1 everytime we find a name that already exists

plt.savefig(save_path)
print(f"Saved prediction grid to '{save_path}'")
plt.show()