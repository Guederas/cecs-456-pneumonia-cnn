# CECS 456 Pneumonia Detection CNN
1. **Download** the dataset from Kaggle: [Chest X_Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

2. **Create** a folder named `data` in this directory.

3. **Unzip** the images into `data/`.

4. **Install Dependencies:**
    Make sure you have Python 3.11 installed. It is recommended to use a virtual environment.
    ```bash
    pip install tensorflow numpy matplotlib seaborn scikit-learn
    ```
    *(Note: If you are on an Apple Silicon Mac, you may need `tensorflow-macos` and `tensorflow-metal` depending on your version).*

5. **Train the Model:**
    Run the main script to train the CNN. This will save the best model to the `models/` folder and generate a training accuracy plot.
    ```bash
    python main.py
    ```

6. **Evaluate Performance:**
    Run the evaluation script to generate the **Confusion Matrix** and calculate Precision/Recall scores. The confusion matrix image will be saved in the `results/` folder.
    ```bash
    python evaluate_model.py
    ```

7. **Test Predictions:**
    Run the prediction script to visualize the model's performance on a batch of random test images. The resulting grid will be saved to the `results/` folder.
    ```bash
    python predict.py
    ```

## Project Structure
* `main.py`: Contains the CNN architecture, data augmentation, and training loop.
* `evaluate_model.py`: Generates the Confusion Matrix and classification report (Precision/Recall).
* `predict.py`: Loads random test images and displays them with the model's predictions and confidence scores.
* `models/`: Stores the trained models.
* `results/`: Stores output graphs and prediction visualizations.
* `data/`: Contains the Train, Test, and Validation datasets.