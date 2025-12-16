# CNN model that predicts Pneumonia or Normal given X-ray dataset
import tensorflow as tf
import keras
from keras import layers, models, callbacks
import os

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32

TRAIN_DIR = os.path.join('data', 'train')
TEST_DIR = os.path.join('data', 'test')
VAL_DIR = os.path.join('data', 'val')

# Load the dataset
print("Loading Data...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='grayscale'
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='grayscale'
)

validate_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='grayscale'
)

# CNN Model
model = models.Sequential() # initializing the CNN

model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))    # 1 channel for grayscale
model.add(layers.Rescaling(1./255)) # normalizing pixel values

# Block 1
model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))  # convolution 1
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2),strides=2, padding='same'))   # pooling 1

# Block 2 (with Dropout)
model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))  # convolution 2
model.add(layers.Dropout(0.1))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2),strides=2, padding='same'))   # pooling 2

# Block 3
model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))  # convolution 3
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2),strides=2, padding='same'))   # pooling 3

# Block 4 (with Dropout)
model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))  # convolution 4
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2),strides=2, padding='same'))   # pooling 4

# Block 5
model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))  # convolution 5
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2),strides=2, padding='same'))   # pooling 5

