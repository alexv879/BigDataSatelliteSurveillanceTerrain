"""
Modern EfficientNetV2 Implementation with Advanced Training Techniques
Cutting-edge CNN architecture optimized for satellite terrain classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
from typing import Tuple, Optional, Dict
import numpy as np


class StochasticDepth(layers.Layer):
    """Implements stochastic depth for better regularization"""

    def __init__(self, drop_prob: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):
        if training and self.drop_prob > 0.0:
            keep_prob = 1 - self.drop_prob
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random.uniform([batch_size, 1, 1, 1])
            binary_tensor = tf.floor(random_tensor)
            output = inputs / keep_prob * binary_tensor
            return output
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


def create_efficientnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    model_variant: str = "B0",
    pretrained: bool = True,
    fine_tune_layers: int = 50,
    dropout_rate: float = 0.3,
    use_attention: bool = True
) -> keras.Model:
    """
    Create modern EfficientNetV2 model with advanced features

    Args:
        input_shape: Input image dimensions
        num_classes: Number of terrain classes
        model_variant: EfficientNet variant (B0, B1, B2, B3)
        pretrained: Use ImageNet pretrained weights
        fine_tune_layers: Number of top layers to fine-tune
        dropout_rate: Dropout rate in classification head
        use_attention: Add attention mechanism

    Returns:
        Compiled Keras model
    """
    # Select base model
    base_models = {
        "B0": EfficientNetV2B0,
        "B1": EfficientNetV2B1,
        "B2": EfficientNetV2B2,
        "B3": EfficientNetV2B3,
    }

    base_model_class = base_models.get(model_variant, EfficientNetV2B0)

    # Load base model
    base_model = base_model_class(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=input_shape,
        pooling=None
    )

    # Freeze base model initially
    base_model.trainable = False

    # Unfreeze top layers for fine-tuning
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    # Build model
    inputs = layers.Input(shape=input_shape)

    # Preprocessing
    x = layers.Rescaling(1./255)(inputs)

    # Base model
    x = base_model(x, training=False)

    # Attention mechanism (Squeeze-and-Excitation)
    if use_attention:
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(x.shape[-1] // 16, activation='relu')(se)
        se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
        se = layers.Reshape((1, 1, x.shape[-1]))(se)
        x = layers.Multiply()([x, se])

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head with dropout
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    return model


def create_efficientnet_with_mixed_precision(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    learning_rate: float = 1e-3,
    **kwargs
) -> keras.Model:
    """
    Create EfficientNetV2 with mixed precision and advanced optimization

    Args:
        input_shape: Input image dimensions
        num_classes: Number of terrain classes
        learning_rate: Initial learning rate
        **kwargs: Additional arguments for create_efficientnet_model

    Returns:
        Compiled model with mixed precision
    """
    # Enable mixed precision for better performance
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model = create_efficientnet_model(input_shape, num_classes, **kwargs)

    # Advanced optimizer with weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.0001,
        clipnorm=1.0  # Gradient clipping
    )

    # Wrap optimizer for mixed precision
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Compile with label smoothing
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1,
            from_logits=False
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ],
    )

    return model


def get_training_callbacks(
    model_name: str = "efficientnet",
    monitor: str = "val_accuracy",
    patience: int = 10
) -> list:
    """
    Get advanced training callbacks for better training

    Args:
        model_name: Name for saving checkpoints
        monitor: Metric to monitor
        patience: Patience for early stopping

    Returns:
        List of callbacks
    """
    callbacks = [
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            f"models/checkpoints/{model_name}_best.h5",
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        ),

        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='max'
        ),

        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max'
        ),

        # Cosine annealing
        keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * tf.math.cos(epoch / 100 * np.pi / 2)
        ),

        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=f"logs/{model_name}",
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),

        # CSV Logger
        keras.callbacks.CSVLogger(
            f"logs/{model_name}_training.csv",
            append=True
        ),
    ]

    return callbacks


def create_progressive_training_schedule(
    model: keras.Model,
    initial_epochs: int = 10,
    fine_tune_epochs: int = 20
) -> Dict[str, int]:
    """
    Create progressive training schedule for transfer learning

    Args:
        model: Keras model
        initial_epochs: Epochs for frozen base training
        fine_tune_epochs: Epochs for fine-tuning

    Returns:
        Training schedule dictionary
    """
    return {
        "frozen_training": initial_epochs,
        "fine_tuning": fine_tune_epochs,
        "total_epochs": initial_epochs + fine_tune_epochs
    }


if __name__ == "__main__":
    # Example usage
    model = create_efficientnet_with_mixed_precision(
        input_shape=(224, 224, 3),
        num_classes=10,
        model_variant="B2",
        pretrained=True,
        fine_tune_layers=50
    )

    print(model.summary())
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
