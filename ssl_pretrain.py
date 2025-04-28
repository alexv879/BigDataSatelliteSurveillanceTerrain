import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
from utils import plot_training_curves

# Constants for SSL task
IMG_SIZE = (64, 64)  # Target image size for SSL task
BATCH_SIZE = 32  # Batch size for training
NUM_ROTATIONS = 4  # Number of rotation classes: 0°, 90°, 180°, or 270°
EPOCHS = 5  # Number of training epochs

def rotate_image(img, angle):
    """
    Rotate image by the given angle.
    Args:
        img: Input image tensor.
        angle: Rotation angle (0, 90, 180, or 270 degrees).
    Returns:
        Rotated image tensor.
    """
    if angle == 0:
        return img
    else:
        return tf.image.rot90(img, k=angle // 90)

def add_rotation_labels(generator):
    """
    Modify the generator output to include rotation labels.
    """
    while True:
        images, _ = next(generator)  # Get a batch of images
        batch_size = images.shape[0]
        # Randomly assign a rotation angle to each image
        rotation_labels = np.random.randint(0, NUM_ROTATIONS, size=batch_size)
        # Apply the rotation to each image
        rotated_images = np.array([rotate_image(img, angle * 90).numpy() for img, angle in zip(images, rotation_labels)])
        # Convert rotation labels to one-hot encoding
        rotation_labels_cat = tf.keras.utils.to_categorical(rotation_labels, NUM_ROTATIONS)
        yield (rotated_images, rotation_labels_cat)

# Create ImageDataGenerator for SSL pre-training
data_dir = os.path.join("data", "train")  # Path to training data
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Ensure validation split is consistent
)
ssl_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None  # Ignore original labels for SSL pre-training
)

if ssl_generator.samples == 0:
    raise ValueError(f"No images found in the directory: {data_dir}. Ensure the dataset is correctly set up.")

# Wrap the generator with rotation labels
ssl_train_gen = add_rotation_labels(ssl_generator)

# Create a validation generator for SSL pre-training
ssl_val_gen = add_rotation_labels(datagen.flow_from_directory(
    directory=data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    subset='validation'  # Use validation split
))

# Build a CNN model for rotation prediction
ssl_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_ROTATIONS, activation='softmax')  # Output layer for 4 rotation classes
])

# Compile the SSL model
ssl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add logging for training progress
print("Starting SSL pre-training...")
history = ssl_model.fit(
    ssl_train_gen,
    steps_per_epoch=ssl_generator.samples // BATCH_SIZE,
    validation_data=ssl_val_gen,
    validation_steps=ssl_val_gen.samples // BATCH_SIZE,  # Use validation generator's samples
    epochs=EPOCHS,
    verbose=1  # Enable detailed logging
)
print("SSL pre-training completed.")

# Save the training history for later analysis
with open("ssl_training_history.json", "w") as f:
    json.dump(history.history, f)

# Save the pretrained weights for later use
ssl_model.save_weights("ssl_pretrained.weights.h5")

# Save training curves
plot_training_curves(history, title="SSL Training Curves", save_path="Bigdataassessment2/ssl_training_curves.png")
print("SSL Training Curves saved as ssl_training_curves.png")