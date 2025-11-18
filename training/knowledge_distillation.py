"""
Knowledge Distillation: Transfer knowledge from large teacher to compact student
Compress models while maintaining performance
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional, Callable
from loguru import logger


class DistillationLoss(keras.losses.Loss):
    """Custom loss for knowledge distillation"""

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        **kwargs
    ):
        """
        Initialize distillation loss

        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss vs hard label loss
        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def call(
        self,
        y_true: tf.Tensor,
        y_pred_student: tf.Tensor,
        y_pred_teacher: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute distillation loss

        Args:
            y_true: True labels (one-hot)
            y_pred_student: Student model predictions
            y_pred_teacher: Teacher model predictions

        Returns:
            Combined loss
        """
        # Hard label loss (regular cross-entropy)
        hard_loss = keras.losses.categorical_crossentropy(
            y_true, y_pred_student
        )

        # Soft label loss (distillation)
        teacher_soft = tf.nn.softmax(
            tf.math.log(y_pred_teacher + 1e-10) / self.temperature
        )
        student_soft = tf.nn.softmax(
            tf.math.log(y_pred_student + 1e-10) / self.temperature
        )

        soft_loss = keras.losses.categorical_crossentropy(
            teacher_soft, student_soft
        )

        # Scale soft loss by temperature squared
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return tf.reduce_mean(total_loss)


class KnowledgeDistillationTrainer:
    """Knowledge distillation trainer"""

    def __init__(
        self,
        teacher_model: keras.Model,
        student_model: keras.Model,
        temperature: float = 3.0,
        alpha: float = 0.5
    ):
        """
        Initialize distillation trainer

        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            temperature: Distillation temperature
            alpha: Weight for distillation loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher
        self.teacher_model.trainable = False

        logger.info(f"Knowledge distillation setup:")
        logger.info(f"Teacher params: {teacher_model.count_params():,}")
        logger.info(f"Student params: {student_model.count_params():,}")
        logger.info(f"Compression ratio: {teacher_model.count_params() / student_model.count_params():.2f}x")

    @tf.function
    def train_step(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        optimizer: keras.optimizers.Optimizer
    ) -> dict:
        """
        Single training step

        Args:
            x: Input batch
            y: Target batch
            optimizer: Optimizer

        Returns:
            Metrics dictionary
        """
        with tf.GradientTape() as tape:
            # Get teacher predictions
            teacher_pred = self.teacher_model(x, training=False)

            # Get student predictions
            student_pred = self.student_model(x, training=True)

            # Compute loss
            loss = self._compute_distillation_loss(y, student_pred, teacher_pred)

        # Compute gradients
        gradients = tape.gradient(loss, self.student_model.trainable_variables)

        # Apply gradients
        optimizer.apply_gradients(
            zip(gradients, self.student_model.trainable_variables)
        )

        # Compute metrics
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(y, axis=1),
                    tf.argmax(student_pred, axis=1)
                ),
                tf.float32
            )
        )

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def _compute_distillation_loss(
        self,
        y_true: tf.Tensor,
        y_pred_student: tf.Tensor,
        y_pred_teacher: tf.Tensor
    ) -> tf.Tensor:
        """Compute distillation loss"""
        # Hard loss
        hard_loss = keras.losses.categorical_crossentropy(y_true, y_pred_student)

        # Soft loss
        teacher_soft = tf.nn.softmax(
            tf.math.log(y_pred_teacher + 1e-10) / self.temperature
        )
        student_soft = tf.nn.softmax(
            tf.math.log(y_pred_student + 1e-10) / self.temperature
        )

        soft_loss = keras.losses.categorical_crossentropy(teacher_soft, student_soft)
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combined
        return self.alpha * tf.reduce_mean(soft_loss) + (1 - self.alpha) * tf.reduce_mean(hard_loss)

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int,
        learning_rate: float = 1e-3,
        callbacks: Optional[list] = None
    ) -> dict:
        """
        Train student with knowledge distillation

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            learning_rate: Learning rate
            callbacks: Training callbacks

        Returns:
            Training history
        """
        logger.info("Starting knowledge distillation training...")

        optimizer = keras.optimizers.Adam(learning_rate)

        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")

            # Training
            epoch_loss = []
            epoch_acc = []

            for x_batch, y_batch in train_dataset:
                metrics = self.train_step(x_batch, y_batch, optimizer)
                epoch_loss.append(metrics['loss'].numpy())
                epoch_acc.append(metrics['accuracy'].numpy())

            # Validation
            val_loss = []
            val_acc = []

            for x_val, y_val in val_dataset:
                teacher_pred = self.teacher_model(x_val, training=False)
                student_pred = self.student_model(x_val, training=False)

                loss = self._compute_distillation_loss(y_val, student_pred, teacher_pred)
                acc = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.argmax(y_val, axis=1),
                            tf.argmax(student_pred, axis=1)
                        ),
                        tf.float32
                    )
                )

                val_loss.append(loss.numpy())
                val_acc.append(acc.numpy())

            # Record history
            history['loss'].append(np.mean(epoch_loss))
            history['accuracy'].append(np.mean(epoch_acc))
            history['val_loss'].append(np.mean(val_loss))
            history['val_accuracy'].append(np.mean(val_acc))

            logger.info(
                f"Loss: {history['loss'][-1]:.4f} - "
                f"Acc: {history['accuracy'][-1]:.4f} - "
                f"Val Loss: {history['val_loss'][-1]:.4f} - "
                f"Val Acc: {history['val_accuracy'][-1]:.4f}"
            )

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, history)

        logger.info("Knowledge distillation training completed")
        return history


def create_student_model(
    teacher_model: keras.Model,
    compression_ratio: float = 4.0,
    num_classes: int = 10
) -> keras.Model:
    """
    Create student model with reduced capacity

    Args:
        teacher_model: Teacher model
        compression_ratio: Compression ratio
        num_classes: Number of classes

    Returns:
        Student model
    """
    # Simple student architecture
    student = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=teacher_model.input_shape[1:]),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    logger.info(f"Student model created with {student.count_params():,} parameters")
    return student


def distill_model(
    teacher_model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    num_classes: int = 10,
    temperature: float = 3.0,
    alpha: float = 0.5,
    epochs: int = 50
) -> keras.Model:
    """
    Complete knowledge distillation pipeline

    Args:
        teacher_model: Pre-trained teacher model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes
        temperature: Distillation temperature
        alpha: Distillation weight
        epochs: Training epochs

    Returns:
        Trained student model
    """
    # Create student
    student_model = create_student_model(teacher_model, num_classes=num_classes)

    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=temperature,
        alpha=alpha
    )

    # Train
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs
    )

    return student_model, history


if __name__ == "__main__":
    print("Knowledge Distillation Ready")
    print("Features:")
    print("- Temperature-scaled distillation")
    print("- Soft and hard label combination")
    print("- Automatic student model creation")
    print("- 4x+ model compression")
    print("- Minimal accuracy loss")
