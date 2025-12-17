import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# Configuration
IMG_SIZE = 150
BATCH_SIZE = 32
THRESHOLD = 0.30    # high recall (if model is 30% or more sure, flag as positive)

# Paths
MODEL_PATH = os.path.join('models', 'pneumonia_best_model.keras')
TEST_DIR = os.path.join('data', 'test')

# Make sure results folder exists
if not os.path.exists('results'):
    os.makedirs('results')

# Load Data
print("Loading Test Data...")
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='grayscale',
    shuffle=False   # critical for evaluation so labels match predictions
)

print("Loading Model...")
model = keras.models.load_model(MODEL_PATH)

# Get predictions
print("Running predictions...")
y_pred_probs = model.predict(test_ds)   # model returns probabilities
y_pred = (y_pred_probs > THRESHOLD).astype(int).flatten()   # apply threshold
y_true = np.concatenate([y for x, y in test_ds], axis=0)    # concatenate all the batches

# Generate Metrics (Precision, Recall, and F1-Score)
print("\n" + "-"*40)
print(f"Evaluation Report (Threshold: {THRESHOLD})")
print("-"*40)
# Target names: 0 = NORMAL, 1 = PNEUMONIA
report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'])
print(report)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# PLot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Threshold: {THRESHOLD})')

# Save it to the results folder
save_path = os.path.join('results', 'confusion_matrix.png')
plt.savefig(save_path)
print(f"\nSaved confusion matrix to '{save_path}'")
plt.show()