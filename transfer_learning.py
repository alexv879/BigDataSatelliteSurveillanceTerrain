import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import create_generators
from utils import evaluate_model, plot_training_curves
import os
import json

# Set data directory and image shape
data_dir = os.path.join("data", "train")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
image_shape = (64, 64, 3)

# Create train and validation generators
train_gen, val_gen = create_generators(data_dir, img_size=(64, 64), batch_size=32)

if not train_gen.class_indices:
    raise ValueError("No class indices found in the training generator. Ensure the dataset is correctly set up.")

# Load ResNet50 with frozen base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=image_shape)
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initial training with frozen base
# Add EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with the base model frozen
history_frozen = model.fit(
    train_gen,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,  # Add validation steps
    epochs=5,
    callbacks=[early_stopping]
)

# Save model weights after frozen base training
model.save_weights("transfer_learning_frozen.weights.h5")

# Save training history
with open("transfer_learning_frozen_history.json", "w") as f:
    json.dump(history_frozen.history, f)

# Plot training curves for frozen base
plot_training_curves(history_frozen, title="Transfer Learning - Frozen Base", save_path="Bigdataassessment2/transfer_learning_frozen_base.png")
print("Transfer Learning Frozen Base Training Curves saved as transfer_learning_frozen_base.png")

# Fine-tuning
# Unfreeze the last 10 layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Freeze all layers except the last 10
    layer.trainable = False

model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,  # Add validation steps
    epochs=3
)

# Save model weights after fine-tuning
model.save_weights("transfer_learning_finetuned.weights.h5")

# Plot training curves for fine-tuning
if 'history_finetune' in locals():
    plot_training_curves(history_finetune, title="Transfer Learning - Fine-Tuning", save_path="Bigdataassessment2/transfer_learning_fine_tuning.png")
    print("Transfer Learning Fine-Tuning Training Curves saved as transfer_learning_fine_tuning.png")
else:
    print("Fine-tuning failed. Skipping training curve plot.")

# Evaluate the model
evaluate_model(model, val_gen, class_indices=train_gen.class_indices)
