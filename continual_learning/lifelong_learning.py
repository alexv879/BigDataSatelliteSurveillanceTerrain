"""
Continual/Lifelong Learning for Satellite Terrain Classification
Update models over time without catastrophic forgetting
Critical for adapting to new terrain types and seasonal changes
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
import copy


class EWC(keras.Model):
    """
    Elastic Weight Consolidation
    Prevents catastrophic forgetting by regularizing important weights
    """

    def __init__(
        self,
        base_model: keras.Model,
        lambda_ewc: float = 0.4
    ):
        """
        Initialize EWC

        Args:
            base_model: Base model to wrap
            lambda_ewc: EWC regularization strength
        """
        super().__init__()
        self.base_model = base_model
        self.lambda_ewc = lambda_ewc

        # Fisher information matrix (importance of weights)
        self.fisher = {}
        self.optimal_params = {}

        logger.info(f"EWC initialized with lambda={lambda_ewc}")

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def compute_fisher(
        self,
        dataset: tf.data.Dataset,
        num_samples: int = 1000
    ):
        """
        Compute Fisher information matrix

        Args:
            dataset: Dataset from previous task
            num_samples: Number of samples for estimation
        """
        logger.info("Computing Fisher information matrix...")

        # Initialize Fisher with zeros
        for var in self.base_model.trainable_variables:
            self.fisher[var.name] = tf.Variable(
                tf.zeros_like(var),
                trainable=False
            )

        # Compute gradients
        samples_processed = 0

        for images, labels in dataset:
            if samples_processed >= num_samples:
                break

            with tf.GradientTape() as tape:
                predictions = self.base_model(images, training=False)
                loss = keras.losses.categorical_crossentropy(labels, predictions)

            # Compute gradients
            grads = tape.gradient(loss, self.base_model.trainable_variables)

            # Accumulate squared gradients (Fisher approximation)
            for var, grad in zip(self.base_model.trainable_variables, grads):
                if grad is not None:
                    self.fisher[var.name].assign_add(grad ** 2)

            samples_processed += len(images)

        # Average Fisher
        for var_name in self.fisher:
            self.fisher[var_name].assign(self.fisher[var_name] / samples_processed)

        # Store optimal parameters
        for var in self.base_model.trainable_variables:
            self.optimal_params[var.name] = tf.Variable(
                var.numpy(),
                trainable=False
            )

        logger.info("Fisher matrix computed")

    def ewc_loss(self) -> tf.Tensor:
        """
        Compute EWC regularization loss

        Returns:
            EWC loss
        """
        loss = 0.0

        for var in self.base_model.trainable_variables:
            if var.name in self.fisher:
                # L = λ/2 * Σ F_i * (θ_i - θ*_i)^2
                diff = var - self.optimal_params[var.name]
                loss += tf.reduce_sum(self.fisher[var.name] * (diff ** 2))

        return (self.lambda_ewc / 2) * loss

    def train_step(self, data):
        """Custom training step with EWC loss"""
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Standard loss
            loss = self.compiled_loss(y, y_pred)

            # Add EWC loss
            if self.fisher:  # If Fisher computed (not first task)
                loss += self.ewc_loss()

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


class PackNet:
    """
    PackNet: Network pruning and packing for continual learning
    Prune network for previous tasks, use remaining capacity for new tasks
    """

    def __init__(
        self,
        model: keras.Model,
        prune_percentage: float = 0.5
    ):
        """
        Initialize PackNet

        Args:
            model: Model to apply PackNet
            prune_percentage: Percentage of weights to prune per task
        """
        self.model = model
        self.prune_percentage = prune_percentage
        self.task_masks = []  # Masks for each task

        logger.info(f"PackNet initialized with {prune_percentage*100}% pruning")

    def train_task(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        epochs: int = 20
    ):
        """
        Train on new task and prune

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Training epochs
        """
        # Train on new task
        logger.info(f"Training task {len(self.task_masks) + 1}...")

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs
        )

        # Prune least important weights
        self._prune_and_pack()

        return history

    def _prune_and_pack(self):
        """Prune least important weights and create mask"""
        logger.info("Pruning and packing...")

        task_mask = {}

        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel.numpy()

                # Compute importance (magnitude)
                importance = np.abs(weights)

                # Already pruned weights (from previous tasks)
                already_pruned = np.zeros_like(weights, dtype=bool)
                for prev_mask in self.task_masks:
                    if layer.name in prev_mask:
                        already_pruned |= prev_mask[layer.name]

                # Flatten and get threshold
                available_weights = importance[~already_pruned]

                if len(available_weights) > 0:
                    threshold = np.percentile(
                        available_weights,
                        self.prune_percentage * 100
                    )

                    # Create mask for this task
                    mask = (importance < threshold) & ~already_pruned

                    task_mask[layer.name] = mask

                    # Apply pruning
                    weights[mask] = 0
                    layer.kernel.assign(weights)

        self.task_masks.append(task_mask)

        logger.info(f"Task {len(self.task_masks)} packed")


class LwF(keras.Model):
    """
    Learning without Forgetting (LwF)
    Use knowledge distillation to preserve previous task performance
    """

    def __init__(
        self,
        base_model: keras.Model,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        Initialize LwF

        Args:
            base_model: Base model
            temperature: Distillation temperature
            alpha: Weight for distillation loss
        """
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.alpha = alpha

        # Store previous model
        self.previous_model = None

        logger.info(f"LwF initialized with T={temperature}, alpha={alpha}")

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def set_previous_model(self):
        """Store current model as previous model"""
        # Clone model
        self.previous_model = keras.models.clone_model(self.base_model)
        self.previous_model.set_weights(self.base_model.get_weights())

        # Freeze previous model
        self.previous_model.trainable = False

        logger.info("Previous model stored")

    def distillation_loss(
        self,
        student_logits: tf.Tensor,
        teacher_logits: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute knowledge distillation loss

        Args:
            student_logits: Current model outputs
            teacher_logits: Previous model outputs

        Returns:
            Distillation loss
        """
        # Soften predictions
        student_soft = tf.nn.softmax(student_logits / self.temperature)
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)

        # KL divergence
        loss = tf.reduce_mean(
            tf.reduce_sum(
                teacher_soft * tf.math.log(teacher_soft / (student_soft + 1e-10)),
                axis=1
            )
        )

        # Scale by temperature^2
        return loss * (self.temperature ** 2)

    def train_step(self, data):
        """Custom training step with distillation"""
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Standard loss
            loss = self.compiled_loss(y, y_pred)

            # Add distillation loss if previous model exists
            if self.previous_model is not None:
                prev_logits = self.previous_model(x, training=False)
                dist_loss = self.distillation_loss(y_pred, prev_logits)
                loss = (1 - self.alpha) * loss + self.alpha * dist_loss

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


class ProgressiveNeuralNetwork:
    """
    Progressive Neural Networks
    Add new columns for new tasks while preserving old columns
    """

    def __init__(self, base_architecture: keras.Model):
        """
        Initialize Progressive NN

        Args:
            base_architecture: Architecture to use for each column
        """
        self.columns = [base_architecture]
        self.lateral_connections = []

        logger.info("Progressive Neural Network initialized")

    def add_task(self, new_architecture: Optional[keras.Model] = None):
        """
        Add new column for new task

        Args:
            new_architecture: Architecture for new column (uses base if None)
        """
        if new_architecture is None:
            # Clone base architecture
            new_col = keras.models.clone_model(self.columns[0])
        else:
            new_col = new_architecture

        self.columns.append(new_col)

        logger.info(f"Added task column {len(self.columns)}")

    def forward(
        self,
        inputs: tf.Tensor,
        task_id: int
    ) -> tf.Tensor:
        """
        Forward pass through progressive network

        Args:
            inputs: Input tensor
            task_id: Which task to execute

        Returns:
            Predictions
        """
        # Process through all columns up to task_id
        column_outputs = []

        for i in range(task_id + 1):
            if i == 0:
                # First column
                output = self.columns[i](inputs)
            else:
                # Lateral connections from previous columns
                lateral_input = sum(column_outputs)
                output = self.columns[i](inputs + lateral_input)

            column_outputs.append(output)

        return column_outputs[-1]


class MemoryReplay:
    """
    Experience Replay / Memory Rehearsal
    Store samples from previous tasks and replay during training
    """

    def __init__(
        self,
        memory_size: int = 2000,
        sampling_strategy: str = 'random'
    ):
        """
        Initialize memory replay

        Args:
            memory_size: Maximum memory buffer size
            sampling_strategy: 'random' or 'balanced'
        """
        self.memory_size = memory_size
        self.sampling_strategy = sampling_strategy

        self.memory_images = []
        self.memory_labels = []

        logger.info(f"Memory Replay initialized with size {memory_size}")

    def add_samples(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_samples: Optional[int] = None
    ):
        """
        Add samples to memory

        Args:
            images: Images to add
            labels: Labels
            n_samples: Number to add (None for all)
        """
        if n_samples is None:
            n_samples = len(images)

        # Random selection
        indices = np.random.choice(len(images), size=n_samples, replace=False)

        # Add to memory
        self.memory_images.extend(images[indices])
        self.memory_labels.extend(labels[indices])

        # Trim if exceeds size
        if len(self.memory_images) > self.memory_size:
            # Keep most recent
            self.memory_images = self.memory_images[-self.memory_size:]
            self.memory_labels = self.memory_labels[-self.memory_size:]

        logger.info(f"Memory size: {len(self.memory_images)}")

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from memory

        Args:
            batch_size: Batch size

        Returns:
            (images, labels)
        """
        if len(self.memory_images) == 0:
            return None, None

        if self.sampling_strategy == 'random':
            indices = np.random.choice(
                len(self.memory_images),
                size=min(batch_size, len(self.memory_images)),
                replace=False
            )
        else:  # balanced
            # Sample balanced across classes
            unique_labels = np.unique(self.memory_labels)
            per_class = batch_size // len(unique_labels)

            indices = []
            for label in unique_labels:
                label_indices = np.where(np.array(self.memory_labels) == label)[0]
                selected = np.random.choice(
                    label_indices,
                    size=min(per_class, len(label_indices)),
                    replace=False
                )
                indices.extend(selected)

            indices = np.array(indices)

        return (
            np.array([self.memory_images[i] for i in indices]),
            np.array([self.memory_labels[i] for i in indices])
        )

    def create_replay_dataset(
        self,
        new_dataset: tf.data.Dataset,
        replay_ratio: float = 0.5
    ) -> tf.data.Dataset:
        """
        Create dataset mixing new samples and replay

        Args:
            new_dataset: Dataset for new task
            replay_ratio: Ratio of replay samples

        Returns:
            Mixed dataset
        """
        if len(self.memory_images) == 0:
            return new_dataset

        # Create replay dataset
        replay_dataset = tf.data.Dataset.from_tensor_slices((
            np.array(self.memory_images),
            np.array(self.memory_labels)
        ))

        # Sample and mix
        # This is simplified - in practice would properly interleave
        mixed_dataset = new_dataset.concatenate(replay_dataset)
        mixed_dataset = mixed_dataset.shuffle(10000)

        return mixed_dataset


if __name__ == "__main__":
    print("Continual Learning Ready!")
    print("\nMethods:")
    print("- EWC (Elastic Weight Consolidation)")
    print("- PackNet (Network pruning and packing)")
    print("- LwF (Learning without Forgetting)")
    print("- Progressive Neural Networks")
    print("- Memory Replay / Experience Replay")
    print("\nBenefits:")
    print("- Prevent catastrophic forgetting")
    print("- Adapt to new terrain types")
    print("- Handle seasonal changes")
    print("- Continuous model improvement")
    print("\n🧠 Lifelong learning for satellite terrain classification!")
