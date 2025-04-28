import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from preprocess import create_generators
from utils import plot_training_curves
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Set the data directory
data_dir = os.path.join("data", "train")

# Create training and validation data generators
train_gen, val_gen = create_generators(data_dir, img_size=(150, 150), batch_size=32)

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Define the CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Add EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with class weights
history = model.fit(
    train_gen,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,  # Add validation steps
    epochs=10,
    callbacks=[reduce_lr, early_stopping],
    class_weight=class_weights_dict  # Add class weights
)

# Save the trained CNN model weights
model.save_weights("cnn_model.weights.h5")

# Plot training curves
plot_training_curves(history, title="CNN Training", save_path="Bigdataassessment2/cnn_training_curves.png")
