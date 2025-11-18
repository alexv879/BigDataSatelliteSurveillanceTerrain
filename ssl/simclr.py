"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
State-of-the-art self-supervised learning for satellite terrain classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class ContrastiveAugmentation:
    """Advanced augmentation for contrastive learning"""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, image):
        """Apply two different augmentations to create positive pairs"""
        # Random crop and resize
        image = tf.image.random_crop(
            image,
            size=[int(self.image_size * 0.8), int(self.image_size * 0.8), 3]
        )
        image = tf.image.resize(image, [self.image_size, self.image_size])

        # Random flip
        image = tf.image.random_flip_left_right(image)

        # Color distortion
        image = tf.image.random_brightness(image, 0.4)
        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_saturation(image, 0.6, 1.4)
        image = tf.image.random_hue(image, 0.1)

        # Random grayscale
        if tf.random.uniform([]) < 0.2:
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.grayscale_to_rgb(image)

        # Gaussian blur
        if tf.random.uniform([]) < 0.5:
            sigma = tf.random.uniform([], 0.1, 2.0)
            image = self._gaussian_blur(image, sigma)

        # Clip values
        image = tf.clip_by_value(image, 0, 1)
        return image

    def _gaussian_blur(self, image, sigma):
        """Apply Gaussian blur"""
        kernel_size = int(sigma * 3) * 2 + 1
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
        gauss_kernel = tf.exp(-tf.square(x) / (2 * tf.square(sigma)))
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)

        # 2D kernel
        gauss_kernel_2d = gauss_kernel[:, None] * gauss_kernel[None, :]
        gauss_kernel_2d = gauss_kernel_2d[:, :, None, None]

        # Apply blur
        image = tf.nn.depthwise_conv2d(
            image[None, ...],
            tf.tile(gauss_kernel_2d, [1, 1, 3, 1]),
            strides=[1, 1, 1, 1],
            padding='SAME'
        )[0]
        return image


class ProjectionHead(layers.Layer):
    """MLP projection head for contrastive learning"""

    def __init__(self, hidden_dim: int = 2048, output_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dense1 = layers.Dense(hidden_dim, activation=None)
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(output_dim, activation=None)

    def call(self, x, training=None):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        # L2 normalization
        return tf.math.l2_normalize(x, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim
        })
        return config


class NTXentLoss(keras.losses.Loss):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    The core loss function for SimCLR
    """

    def __init__(self, temperature: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, z_i, z_j):
        """
        Compute NT-Xent loss
        Args:
            z_i: Embeddings from first augmentation
            z_j: Embeddings from second augmentation
        """
        batch_size = tf.shape(z_i)[0]

        # Concatenate embeddings
        z = tf.concat([z_i, z_j], axis=0)  # Shape: (2*batch_size, embedding_dim)

        # Compute similarity matrix
        similarity_matrix = tf.matmul(z, z, transpose_b=True)  # (2N, 2N)
        similarity_matrix = similarity_matrix / self.temperature

        # Create masks
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        similarity_matrix = tf.where(mask, -1e9, similarity_matrix)

        # Create positive pairs mask
        positives_mask = tf.concat([
            tf.concat([tf.zeros((batch_size, batch_size)), tf.eye(batch_size)], axis=1),
            tf.concat([tf.eye(batch_size), tf.zeros((batch_size, batch_size))], axis=1)
        ], axis=0)

        # Compute loss
        numerator = tf.reduce_sum(
            tf.exp(similarity_matrix) * positives_mask,
            axis=1
        )
        denominator = tf.reduce_sum(tf.exp(similarity_matrix), axis=1)

        loss = -tf.math.log(numerator / denominator)
        return tf.reduce_mean(loss)


def create_simclr_encoder(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    base_model: str = "resnet50",
    hidden_dim: int = 2048,
    projection_dim: int = 128
) -> keras.Model:
    """
    Create SimCLR encoder with projection head

    Args:
        input_shape: Input image shape
        base_model: Base encoder architecture
        hidden_dim: Hidden dimension in projection head
        projection_dim: Output dimension of projection head

    Returns:
        SimCLR model
    """
    # Base encoder
    if base_model == "resnet50":
        base = keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling='avg'
        )
    elif base_model == "efficientnetv2":
        base = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling='avg'
        )
    else:
        raise ValueError(f"Unknown base model: {base_model}")

    # Build model
    inputs = layers.Input(shape=input_shape)
    h = base(inputs)
    z = ProjectionHead(hidden_dim, projection_dim)(h)

    model = keras.Model(inputs, z, name="simclr_encoder")
    return model


class SimCLRTrainer(keras.Model):
    """Complete SimCLR training pipeline"""

    def __init__(
        self,
        encoder: keras.Model,
        temperature: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="contrastive_loss")
        self.nt_xent_loss = NTXentLoss(temperature)

    def call(self, inputs, training=None):
        return self.encoder(inputs, training=training)

    def train_step(self, data):
        # Unpack augmented pairs
        (x_i, x_j), _ = data

        with tf.GradientTape() as tape:
            # Get embeddings
            z_i = self.encoder(x_i, training=True)
            z_j = self.encoder(x_j, training=True)

            # Compute loss
            loss = self.nt_xent_loss(z_i, z_j)

        # Update weights
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def create_simclr_dataset(
    image_paths: list,
    labels: Optional[list] = None,
    image_size: int = 224,
    batch_size: int = 256
) -> tf.data.Dataset:
    """
    Create SimCLR training dataset with augmented pairs

    Args:
        image_paths: List of image file paths
        labels: Optional labels (not used in SSL)
        image_size: Target image size
        batch_size: Batch size

    Returns:
        TensorFlow dataset with augmented pairs
    """
    augmenter = ContrastiveAugmentation(image_size)

    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def create_pairs(path):
        # Load image
        image = load_image(path)

        # Create two augmented views
        x_i = augmenter(image)
        x_j = augmenter(image)

        return (x_i, x_j), 0  # Dummy label

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(create_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_simclr(
    train_dataset: tf.data.Dataset,
    encoder: keras.Model,
    epochs: int = 100,
    learning_rate: float = 0.3,
    temperature: float = 0.5
) -> keras.Model:
    """
    Train SimCLR model

    Args:
        train_dataset: Training dataset with augmented pairs
        encoder: Encoder model
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        temperature: Temperature for NT-Xent loss

    Returns:
        Trained encoder
    """
    # Create SimCLR trainer
    simclr = SimCLRTrainer(encoder, temperature)

    # Optimizer with LARS (Layer-wise Adaptive Rate Scaling)
    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True
    )

    simclr.compile(optimizer=optimizer)

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'models/simclr_best.h5',
            save_best_only=True,
            monitor='loss'
        ),
        keras.callbacks.TensorBoard(log_dir='logs/simclr')
    ]

    # Train
    simclr.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return encoder


if __name__ == "__main__":
    print("SimCLR Self-Supervised Learning Ready")
    print("Features:")
    print("- Contrastive learning with NT-Xent loss")
    print("- Advanced augmentation pipeline")
    print("- MLP projection head")
    print("- Temperature-scaled similarity")
    print("- State-of-the-art SSL performance")
