# CNN model that predicts Pneumonia or Normal given X-ray dataset
import tensorflow as tf
import keras
import os

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32

TRAIN_DIR = os.path.join('data', 'train')
TEST_DIR = os.path.join('data', 'test')
VAL_DIR = os.path.join('data', 'val')

# Data loading
print("Loading Data...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary'
    color_mode='grayscale'
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary'
    color_mode='grayscale'
)

validate_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary'
    color_mode='grayscale'
)