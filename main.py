# CNN model that predicts Pneumonia or Normal given X-ray dataset
import keras
from keras import layers, models, callbacks
import os

# Configuration
IMG_SIZE = 150  # better detail needed for x-ray images
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

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='grayscale'
)

# Data augmentation layer to address overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# CNN Model
model = models.Sequential() # initializing the CNN

model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))    # 1 channel for grayscale
model.add(layers.Rescaling(1./255)) # normalizing pixel values

# Add Augmentation (only active during training)
model.add(data_augmentation)

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

# Flatten and Dense
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))    # binary output (normal or pneumonia)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Defining all Callbacks
# Save the best model
checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join('models', 'pneumonia_best_model.keras'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Lower learning rate if stuck
lr_reduction = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2, # reduce after 2 epochs
    verbose=1,
    factor=0.3,
    min_lr=0.000001
)

# Stop if not improving
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5, # stop after 5 epochs of no improvement
    restore_best_weights=True
)

# Training Model
print("\nStarting Training...")
history = model.fit(
    train_ds,
    epochs=15,
    validation_data=test_ds,    # using test data instead because val_ds only contains 16 photos
    callbacks=[checkpoint, lr_reduction, early_stopping]    # use all callbacks
)

# Evaluate
print("\nEvaluating Model...")
score = model.evaluate(test_ds)
print(f"Test Accuracy: {score[1]*100:.2f}%")

model.save(os.path.join('models', 'pneumonia_model.keras'))