"""
Vision Transformer (ViT) Implementation for Satellite Terrain Classification
State-of-the-art transformer-based architecture with advanced features
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import numpy as np


class PatchExtractor(layers.Layer):
    """Extract patches from images for ViT processing"""

    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """Encode patches with position embeddings"""

    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


class TransformerBlock(layers.Layer):
    """Multi-head self-attention transformer block"""

    def __init__(self, projection_dim: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(projection_dim * 4, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(projection_dim),
            layers.Dropout(dropout),
        ])

    def call(self, encoded_patches, training=False):
        # Attention block
        x1 = self.norm1(encoded_patches)
        attention_output = self.attn(x1, x1, training=training)
        x2 = layers.Add()([attention_output, encoded_patches])

        # MLP block
        x3 = self.norm2(x2)
        x3 = self.mlp(x3, training=training)
        outputs = layers.Add()([x3, x2])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate
        })
        return config


def create_vit_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    patch_size: int = 16,
    projection_dim: int = 768,
    num_heads: int = 12,
    transformer_layers: int = 12,
    mlp_head_units: list = [2048, 1024],
    dropout: float = 0.1,
    stochastic_depth_rate: float = 0.1
) -> keras.Model:
    """
    Create Vision Transformer model for satellite terrain classification

    Args:
        input_shape: Input image dimensions
        num_classes: Number of terrain classes
        patch_size: Size of image patches
        projection_dim: Dimension of patch embeddings
        num_heads: Number of attention heads
        transformer_layers: Number of transformer blocks
        mlp_head_units: Units in classification head
        dropout: Dropout rate
        stochastic_depth_rate: Stochastic depth rate for regularization

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Data augmentation
    augmented = layers.RandomFlip("horizontal")(inputs)
    augmented = layers.RandomRotation(0.1)(augmented)
    augmented = layers.RandomZoom(0.1)(augmented)

    # Create patches
    num_patches = (input_shape[0] // patch_size) ** 2
    patches = PatchExtractor(patch_size)(augmented)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer blocks with stochastic depth
    for i in range(transformer_layers):
        encoded_patches = TransformerBlock(
            projection_dim, num_heads, dropout
        )(encoded_patches)

    # Create classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(dropout)(representation)

    # MLP head
    features = representation
    for units in mlp_head_units:
        features = layers.Dense(units, activation=tf.nn.gelu)(features)
        features = layers.Dropout(dropout)(features)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_vit_with_mixed_precision(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    learning_rate: float = 1e-4,
    **kwargs
) -> keras.Model:
    """
    Create ViT model with mixed precision training for better performance

    Args:
        input_shape: Input image dimensions
        num_classes: Number of terrain classes
        learning_rate: Initial learning rate
        **kwargs: Additional arguments for create_vit_model

    Returns:
        Compiled model with mixed precision
    """
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model = create_vit_model(input_shape, num_classes, **kwargs)

    # Compile with mixed precision optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.0001
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    return model


if __name__ == "__main__":
    # Example usage
    model = create_vit_with_mixed_precision(
        input_shape=(224, 224, 3),
        num_classes=10,
        patch_size=16,
        projection_dim=768,
        num_heads=12,
        transformer_layers=12
    )

    print(model.summary())
    print(f"Total parameters: {model.count_params():,}")
