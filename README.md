# Music Genre Classification Using CNN and TensorFlow

This project develops a **Convolutional Neural Network (CNN)** model to classify music genres based on **Mel spectrograms**. The model is built and trained using **TensorFlow** and **Keras**. It achieves high accuracy on both training and validation datasets. The project also includes features to save the trained model, evaluate its performance, and visualize key metrics such as accuracy and loss over multiple epochs.

## Features
- **CNN-based Architecture**: The model is designed to classify music genres using Mel spectrogram images.
- **Training and Validation**: The model is trained and evaluated on separate datasets to assess its ability to generalize to unseen data.
- **Performance Visualization**: Plots of accuracy and loss over time to help visualize model performance and detect any overfitting.
- **Model Saving and Reloading**: The trained model and its training history are saved for future use and analysis.
- **Detailed Evaluation**: Includes metrics such as accuracy and loss for both training and validation sets.

## Dataset
The input data consists of **Mel spectrograms** extracted from audio files, which are preprocessed into a suitable format for CNNs.

### Preprocessing Steps:
1. **Loading Audio**: Audio files are loaded and converted into Mel spectrogram images.
2. **Chunking Audio**: Each audio file is split into overlapping chunks for continuity.
3. **Mel Spectrogram**: Spectrograms are computed for each chunk.
4. **Resizing**: Spectrograms are resized to a fixed shape.
5. **Appending Data**: The processed spectrograms are appended for model training.

## Model Architecture
The CNN architecture consists of several convolutional and pooling layers to extract features from the input spectrograms, followed by fully connected layers to classify the input into one of the target music genres.

**Key Layers**:
- **Conv2D**: Extracts spatial features from the Mel spectrogram images.
- **MaxPooling2D**: Reduces dimensionality to capture prominent features.
- **Dropout**: Prevents overfitting by randomly dropping units during training.
- **Dense**: Fully connected layers for classification.

### Example Model Summary:
```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 210, 210, 32)      320
...
dense_1 (Dense)              (None, 10)                1000
=================================================================
Total params: 55,520,000
Trainable params: 55,500,000
```

## Training
The model is trained using **categorical cross-entropy** as the loss function and the **Adam optimizer**. The dataset is divided into training and validation sets, and performance metrics (accuracy and loss) are tracked over **30 epochs**.

### Training Code:
```python
training_history = model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_data=(X_test, Y_test))
```

### Model Evaluation:
```python
train_loss, train_accuracy = model.evaluate(X_train, Y_train)
val_loss, val_accuracy = model.evaluate(X_test, Y_test)
```

**Results**:
- **Training Accuracy**: 99.66%
- **Validation Accuracy**: 93.03%

## Model Saving and Reloading
The trained model and training history are saved for future use, allowing reloading without retraining.

- **Save the Model**:
  ```python
  model.save("Trained_model.h5")
  ```
- **Reload the Model**:
  ```python
  model = tf.keras.models.load_model("Trained_model.h5")
  ```

## Visualization
Two key metrics, **accuracy** and **loss**, are visualized over the training epochs for both the training and validation datasets. These visualizations help in understanding how well the model is learning and detecting any signs of overfitting.

### Accuracy and Loss Plots:
```python
import matplotlib.pyplot as plt

# Accuracy Plot
plt.plot(training_history.history['accuracy'], label='Training Accuracy')
plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Visualization of Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.plot(training_history.history['loss'], label='Training Loss')
plt.plot(training_history.history['val_loss'], label='Validation Loss')
plt.title('Visualization of Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
```

### Accuracy Graph:
![Accuracy Graph](path/to/accuracy_graph.png)

### Loss Graph:
![Loss Graph](path/to/loss_graph.png)

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

You can install the required packages using:
```bash
pip install tensorflow keras numpy matplotlib
```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
   ```

2. **Train the Model**:
   Run the training script to start training the model on your dataset.
   ```bash
   python train_model.py
   ```

3. **Evaluate the Model**:
   Evaluate the model's performance on test data using:
   ```bash
   python evaluate_model.py
   ```

4. **Visualize Results**:
   View accuracy and loss plots to analyze the model's performance.
