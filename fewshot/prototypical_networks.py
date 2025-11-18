"""
Few-Shot Learning for Satellite Terrain Classification
Learn new terrain classes from just a few examples
Implements Prototypical Networks and Matching Networks
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple, List
from loguru import logger


class PrototypicalNetwork(keras.Model):
    """
    Prototypical Networks for Few-Shot Learning
    Paper: "Prototypical Networks for Few-shot Learning" (Snell et al., 2017)
    """

    def __init__(self, encoder: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def compute_prototypes(
        self,
        support_embeddings: tf.Tensor,
        support_labels: tf.Tensor,
        n_way: int
    ) -> tf.Tensor:
        """
        Compute class prototypes (mean of support embeddings per class)

        Args:
            support_embeddings: Support set embeddings (N, D)
            support_labels: Support labels (N,)
            n_way: Number of classes

        Returns:
            Prototypes (n_way, D)
        """
        prototypes = []

        for class_idx in range(n_way):
            # Get embeddings for this class
            class_mask = tf.equal(support_labels, class_idx)
            class_embeddings = tf.boolean_mask(support_embeddings, class_mask)

            # Compute prototype (mean)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        return tf.stack(prototypes)

    def compute_distances(
        self,
        query_embeddings: tf.Tensor,
        prototypes: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute Euclidean distances from queries to prototypes

        Args:
            query_embeddings: Query embeddings (M, D)
            prototypes: Class prototypes (n_way, D)

        Returns:
            Distances (M, n_way)
        """
        # Expand dimensions for broadcasting
        queries = tf.expand_dims(query_embeddings, axis=1)  # (M, 1, D)
        protos = tf.expand_dims(prototypes, axis=0)  # (1, n_way, D)

        # Euclidean distance
        distances = tf.reduce_sum(tf.square(queries - protos), axis=-1)

        return distances

    def call(
        self,
        support_images: tf.Tensor,
        support_labels: tf.Tensor,
        query_images: tf.Tensor,
        n_way: int,
        training=None
    ) -> tf.Tensor:
        """
        Forward pass for episodic training

        Args:
            support_images: Support set images
            support_labels: Support set labels
            query_images: Query set images
            n_way: Number of classes
            training: Training mode

        Returns:
            Query predictions (logits)
        """
        # Encode support and query sets
        support_embeddings = self.encoder(support_images, training=training)
        query_embeddings = self.encoder(query_images, training=training)

        # Compute prototypes
        prototypes = self.compute_prototypes(
            support_embeddings,
            support_labels,
            n_way
        )

        # Compute distances
        distances = self.compute_distances(query_embeddings, prototypes)

        # Convert distances to logits (negative distance)
        logits = -distances

        return logits


class MatchingNetwork(keras.Model):
    """
    Matching Networks for Few-Shot Learning
    Paper: "Matching Networks for One Shot Learning" (Vinyals et al., 2016)
    """

    def __init__(
        self,
        encoder: keras.Model,
        attention_model: Optional[keras.Model] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.attention_model = attention_model or self._build_attention()

    def _build_attention(self) -> keras.Model:
        """Build attention mechanism"""
        return keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='softmax')
        ])

    def compute_attention_weights(
        self,
        query_embedding: tf.Tensor,
        support_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute attention weights between query and support

        Args:
            query_embedding: Query embedding (D,)
            support_embeddings: Support embeddings (N, D)

        Returns:
            Attention weights (N,)
        """
        # Cosine similarity
        query_norm = tf.nn.l2_normalize(query_embedding, axis=-1)
        support_norm = tf.nn.l2_normalize(support_embeddings, axis=-1)

        similarities = tf.matmul(
            tf.expand_dims(query_norm, 0),
            support_norm,
            transpose_b=True
        )[0]

        # Softmax
        attention = tf.nn.softmax(similarities)

        return attention

    def call(
        self,
        support_images: tf.Tensor,
        support_labels: tf.Tensor,
        query_images: tf.Tensor,
        n_way: int,
        training=None
    ) -> tf.Tensor:
        """Forward pass with attention-based matching"""
        # Encode
        support_embeddings = self.encoder(support_images, training=training)
        query_embeddings = self.encoder(query_images, training=training)

        # For each query, compute attention over support
        predictions = []

        for query_emb in query_embeddings:
            # Attention weights
            attention = self.compute_attention_weights(query_emb, support_embeddings)

            # Weighted sum of labels
            one_hot_labels = tf.one_hot(support_labels, depth=n_way)
            prediction = tf.reduce_sum(
                tf.expand_dims(attention, -1) * one_hot_labels,
                axis=0
            )
            predictions.append(prediction)

        return tf.stack(predictions)


class EpisodeSampler:
    """Sample few-shot learning episodes"""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15
    ):
        """
        Initialize episode sampler

        Args:
            images: All training images
            labels: All labels
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
        """
        self.images = images
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        # Organize by class
        self.class_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def sample_episode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample one episode

        Returns:
            support_images, support_labels, query_images, query_labels
        """
        # Sample n_way classes
        available_classes = list(self.class_to_indices.keys())
        episode_classes = np.random.choice(
            available_classes,
            size=self.n_way,
            replace=False
        )

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_idx, true_class in enumerate(episode_classes):
            # Sample k_shot + n_query examples
            class_indices = self.class_to_indices[true_class]
            sampled_indices = np.random.choice(
                class_indices,
                size=self.k_shot + self.n_query,
                replace=False
            )

            # Split into support and query
            support_idx = sampled_indices[:self.k_shot]
            query_idx = sampled_indices[self.k_shot:]

            # Add to episode
            support_images.extend(self.images[support_idx])
            support_labels.extend([class_idx] * self.k_shot)

            query_images.extend(self.images[query_idx])
            query_labels.extend([class_idx] * self.n_query)

        return (
            np.array(support_images),
            np.array(support_labels),
            np.array(query_images),
            np.array(query_labels)
        )


def create_few_shot_encoder(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    embedding_dim: int = 128
) -> keras.Model:
    """
    Create encoder for few-shot learning

    Args:
        input_shape: Input image shape
        embedding_dim: Embedding dimension

    Returns:
        Encoder model
    """
    inputs = keras.Input(shape=input_shape)

    # Convolutional backbone
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Embedding
    embeddings = layers.Dense(embedding_dim, activation=None)(x)

    # L2 normalization
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    return keras.Model(inputs, embeddings, name='few_shot_encoder')


def train_prototypical_network(
    sampler: EpisodeSampler,
    num_episodes: int = 10000,
    n_way: int = 5,
    k_shot: int = 5,
    learning_rate: float = 1e-3
):
    """
    Train Prototypical Network

    Args:
        sampler: Episode sampler
        num_episodes: Number of training episodes
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        learning_rate: Learning rate

    Returns:
        Trained model
    """
    logger.info("Training Prototypical Network...")

    # Create encoder
    encoder = create_few_shot_encoder()

    # Create model
    model = PrototypicalNetwork(encoder)

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate)

    # Training loop
    for episode in range(num_episodes):
        # Sample episode
        support_imgs, support_lbls, query_imgs, query_lbls = sampler.sample_episode()

        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(
                tf.constant(support_imgs, dtype=tf.float32),
                tf.constant(support_lbls, dtype=tf.int32),
                tf.constant(query_imgs, dtype=tf.float32),
                n_way=n_way,
                training=True
            )

            # Loss
            loss = keras.losses.sparse_categorical_crossentropy(
                query_lbls,
                logits,
                from_logits=True
            )
            loss = tf.reduce_mean(loss)

        # Backprop
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Log
        if episode % 100 == 0:
            accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(logits, axis=-1, output_type=tf.int32),
                        query_lbls
                    ),
                    tf.float32
                )
            )
            logger.info(
                f"Episode {episode}/{num_episodes} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

    logger.info("Training completed!")
    return model


if __name__ == "__main__":
    print("Few-Shot Learning Ready")
    print("Features:")
    print("- Prototypical Networks: Learn from class prototypes")
    print("- Matching Networks: Attention-based matching")
    print("- Episode Sampler: N-way K-shot sampling")
    print("\nLearn new terrain classes from just 5 examples!")
