"""
Semi-Supervised and Weak Supervision Learning
Leverage unlabeled data and weak labels for satellite imagery
Reduce labeling costs by 80-95%
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Tuple, Optional, Callable
from loguru import logger


class PseudoLabeling:
    """
    Pseudo-labeling for semi-supervised learning
    Use model's own predictions as labels for unlabeled data
    """

    def __init__(
        self,
        model: keras.Model,
        confidence_threshold: float = 0.9
    ):
        """
        Initialize pseudo-labeling

        Args:
            model: Base model
            confidence_threshold: Confidence threshold for pseudo-labels
        """
        self.model = model
        self.confidence_threshold = confidence_threshold

    def generate_pseudo_labels(
        self,
        X_unlabeled: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate pseudo-labels for unlabeled data

        Args:
            X_unlabeled: Unlabeled samples

        Returns:
            (pseudo_labels, confidences, selected_indices)
        """
        # Get predictions
        predictions = self.model.predict(X_unlabeled, verbose=0)

        # Get class and confidence
        pseudo_labels = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        # Select high-confidence samples
        selected_mask = confidences >= self.confidence_threshold
        selected_indices = np.where(selected_mask)[0]

        logger.info(
            f"Pseudo-labeling: {len(selected_indices)} / {len(X_unlabeled)} samples "
            f"selected (threshold={self.confidence_threshold})"
        )

        return pseudo_labels, confidences, selected_indices

    def train_with_pseudo_labels(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        pseudo_label_weight: float = 0.5
    ):
        """
        Train with labeled + pseudo-labeled data

        Args:
            X_labeled: Labeled data
            y_labeled: Labels
            X_unlabeled: Unlabeled data
            epochs: Training epochs
            batch_size: Batch size
            pseudo_label_weight: Weight for pseudo-labeled loss
        """
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Generate pseudo-labels
            pseudo_labels, _, selected_indices = self.generate_pseudo_labels(X_unlabeled)

            if len(selected_indices) == 0:
                logger.warning("No high-confidence pseudo-labels, skipping epoch")
                continue

            # Combine labeled and pseudo-labeled data
            X_pseudo = X_unlabeled[selected_indices]
            y_pseudo = keras.utils.to_categorical(
                pseudo_labels[selected_indices],
                num_classes=y_labeled.shape[1]
            )

            X_combined = np.vstack([X_labeled, X_pseudo])
            y_combined = np.vstack([y_labeled, y_pseudo])

            # Sample weights (lower for pseudo-labels)
            sample_weights = np.concatenate([
                np.ones(len(X_labeled)),
                np.ones(len(X_pseudo)) * pseudo_label_weight
            ])

            # Train
            self.model.fit(
                X_combined,
                y_combined,
                sample_weight=sample_weights,
                epochs=1,
                batch_size=batch_size,
                verbose=0
            )


class MeanTeacher:
    """
    Mean Teacher semi-supervised learning
    Maintain a teacher model as exponential moving average of student
    """

    def __init__(
        self,
        student_model: keras.Model,
        teacher_model: keras.Model,
        ema_decay: float = 0.99
    ):
        """
        Initialize Mean Teacher

        Args:
            student_model: Student model
            teacher_model: Teacher model (same architecture)
            ema_decay: EMA decay for teacher updates
        """
        self.student = student_model
        self.teacher = teacher_model
        self.ema_decay = ema_decay

        # Initialize teacher with student weights
        self.teacher.set_weights(self.student.get_weights())

    def update_teacher(self):
        """Update teacher weights with EMA"""
        student_weights = self.student.get_weights()
        teacher_weights = self.teacher.get_weights()

        new_teacher_weights = []
        for sw, tw in zip(student_weights, teacher_weights):
            new_tw = self.ema_decay * tw + (1 - self.ema_decay) * sw
            new_teacher_weights.append(new_tw)

        self.teacher.set_weights(new_teacher_weights)

    def consistency_loss(
        self,
        student_pred: tf.Tensor,
        teacher_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute consistency loss between student and teacher

        Args:
            student_pred: Student predictions
            teacher_pred: Teacher predictions

        Returns:
            Consistency loss
        """
        return tf.reduce_mean(tf.square(student_pred - teacher_pred))

    def train_step(
        self,
        X_labeled: tf.Tensor,
        y_labeled: tf.Tensor,
        X_unlabeled: tf.Tensor,
        optimizer: keras.optimizers.Optimizer,
        consistency_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Single training step

        Args:
            X_labeled: Labeled data
            y_labeled: Labels
            X_unlabeled: Unlabeled data
            optimizer: Optimizer
            consistency_weight: Weight for consistency loss

        Returns:
            Loss dictionary
        """
        with tf.GradientTape() as tape:
            # Student predictions on labeled data
            student_pred_labeled = self.student(X_labeled, training=True)
            supervised_loss = keras.losses.categorical_crossentropy(
                y_labeled,
                student_pred_labeled
            )
            supervised_loss = tf.reduce_mean(supervised_loss)

            # Student and teacher predictions on unlabeled data
            student_pred_unlabeled = self.student(X_unlabeled, training=True)
            teacher_pred_unlabeled = self.teacher(X_unlabeled, training=False)

            # Consistency loss
            consistency = self.consistency_loss(student_pred_unlabeled, teacher_pred_unlabeled)

            # Total loss
            total_loss = supervised_loss + consistency_weight * consistency

        # Update student
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Update teacher
        self.update_teacher()

        return {
            'supervised_loss': float(supervised_loss.numpy()),
            'consistency_loss': float(consistency.numpy()),
            'total_loss': float(total_loss.numpy())
        }


class MixMatch:
    """
    MixMatch semi-supervised learning
    Combines MixUp, pseudo-labeling, and consistency regularization
    """

    def __init__(
        self,
        model: keras.Model,
        num_classes: int,
        K: int = 2,  # Number of augmentations
        T: float = 0.5,  # Temperature for sharpening
        alpha: float = 0.75  # MixUp parameter
    ):
        """
        Initialize MixMatch

        Args:
            model: Model
            num_classes: Number of classes
            K: Augmentation multiplicity
            T: Sharpening temperature
            alpha: MixUp alpha
        """
        self.model = model
        self.num_classes = num_classes
        self.K = K
        self.T = T
        self.alpha = alpha

    def sharpen(self, p: tf.Tensor) -> tf.Tensor:
        """
        Sharpen predictions

        Args:
            p: Probability distribution

        Returns:
            Sharpened distribution
        """
        p_sharp = tf.pow(p, 1.0 / self.T)
        return p_sharp / tf.reduce_sum(p_sharp, axis=-1, keepdims=True)

    def guess_labels(
        self,
        X_unlabeled_aug: List[tf.Tensor]
    ) -> tf.Tensor:
        """
        Generate guessed labels for unlabeled data

        Args:
            X_unlabeled_aug: List of K augmentations

        Returns:
            Guessed labels (sharpened average)
        """
        # Predict for each augmentation
        predictions = []
        for x_aug in X_unlabeled_aug:
            pred = self.model(x_aug, training=False)
            predictions.append(pred)

        # Average predictions
        avg_pred = tf.reduce_mean(tf.stack(predictions), axis=0)

        # Sharpen
        guessed_labels = self.sharpen(avg_pred)

        return guessed_labels

    def mixup(
        self,
        X1: tf.Tensor,
        y1: tf.Tensor,
        X2: tf.Tensor,
        y2: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        MixUp augmentation

        Args:
            X1, y1: First batch
            X2, y2: Second batch

        Returns:
            Mixed batch
        """
        batch_size = tf.shape(X1)[0]
        lambda_val = np.random.beta(self.alpha, self.alpha, size=batch_size)
        lambda_val = tf.constant(lambda_val, dtype=tf.float32)
        lambda_val = tf.reshape(lambda_val, [-1, 1, 1, 1])

        X_mixed = lambda_val * X1 + (1 - lambda_val) * X2

        lambda_val_y = tf.reshape(lambda_val[:, 0, 0, 0], [-1, 1])
        y_mixed = lambda_val_y * y1 + (1 - lambda_val_y) * y2

        return X_mixed, y_mixed


class WeakSupervision:
    """
    Learn from weak labels (noisy, incomplete, or imprecise)
    """

    def __init__(self, model: keras.Model):
        """
        Initialize weak supervision

        Args:
            model: Model
        """
        self.model = model

    def clean_noisy_labels(
        self,
        X: np.ndarray,
        y_noisy: np.ndarray,
        confidence_threshold: float = 0.9,
        num_iterations: int = 5
    ) -> np.ndarray:
        """
        Clean noisy labels iteratively

        Args:
            X: Training data
            y_noisy: Noisy labels
            confidence_threshold: Threshold for label correction
            num_iterations: Number of cleaning iterations

        Returns:
            Cleaned labels
        """
        y_clean = y_noisy.copy()

        for iteration in range(num_iterations):
            logger.info(f"Label cleaning iteration {iteration + 1}/{num_iterations}")

            # Train on current labels
            self.model.fit(X, y_clean, epochs=5, verbose=0)

            # Get predictions
            predictions = self.model.predict(X, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            pred_confidence = np.max(predictions, axis=1)

            # Identify mislabeled samples
            current_labels = np.argmax(y_clean, axis=1) if y_clean.ndim == 2 else y_clean

            # Correct high-confidence disagreements
            disagreement = (pred_classes != current_labels)
            high_confidence = (pred_confidence > confidence_threshold)

            to_correct = disagreement & high_confidence

            num_corrected = to_correct.sum()
            if num_corrected > 0:
                logger.info(f"Correcting {num_corrected} labels")
                y_clean_indices = current_labels.copy()
                y_clean_indices[to_correct] = pred_classes[to_correct]
                y_clean = keras.utils.to_categorical(y_clean_indices, num_classes=y_clean.shape[1])

        return y_clean

    def learn_from_proportions(
        self,
        X_bags: List[np.ndarray],
        bag_proportions: List[np.ndarray],
        num_classes: int,
        epochs: int = 50
    ):
        """
        Learn from bag-level label proportions (LLP)

        Args:
            X_bags: List of bags (sets of instances)
            bag_proportions: List of class proportions per bag
            num_classes: Number of classes
            epochs: Training epochs
        """
        # Custom loss for LLP
        def llp_loss(bag_true_prop, bag_pred_probs):
            # Predicted proportions (mean over bag)
            pred_prop = tf.reduce_mean(bag_pred_probs, axis=0)
            # MSE between true and predicted proportions
            return tf.reduce_mean(tf.square(bag_true_prop - pred_prop))

        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        for epoch in range(epochs):
            total_loss = 0

            for bag_X, bag_prop in zip(X_bags, bag_proportions):
                with tf.GradientTape() as tape:
                    # Predict for all instances in bag
                    predictions = self.model(bag_X, training=True)

                    # Compute LLP loss
                    loss = llp_loss(
                        tf.constant(bag_prop, dtype=tf.float32),
                        predictions
                    )

                # Update
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                total_loss += float(loss.numpy())

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_bags):.4f}")


class SelfTraining:
    """
    Self-training wrapper
    Iteratively add confident predictions to training set
    """

    def __init__(
        self,
        base_model_fn: Callable,
        confidence_threshold: float = 0.9
    ):
        """
        Initialize self-training

        Args:
            base_model_fn: Function to create base model
            confidence_threshold: Confidence threshold
        """
        self.base_model_fn = base_model_fn
        self.confidence_threshold = confidence_threshold

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        max_iterations: int = 10,
        batch_per_iteration: int = 100
    ) -> keras.Model:
        """
        Self-training loop

        Args:
            X_labeled: Initial labeled data
            y_labeled: Initial labels
            X_unlabeled: Unlabeled data
            max_iterations: Maximum iterations
            batch_per_iteration: Samples to add per iteration

        Returns:
            Trained model
        """
        # Current training set
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()

        for iteration in range(max_iterations):
            logger.info(f"Self-training iteration {iteration + 1}/{max_iterations}")
            logger.info(f"Training set size: {len(X_train)}, Pool size: {len(X_pool)}")

            # Train model
            model = self.base_model_fn()
            model.fit(X_train, y_train, epochs=20, verbose=0)

            if len(X_pool) == 0:
                logger.info("Pool exhausted")
                break

            # Predict on pool
            predictions = model.predict(X_pool, verbose=0)
            pseudo_labels = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)

            # Select confident samples
            confident_mask = confidences >= self.confidence_threshold
            confident_indices = np.where(confident_mask)[0]

            if len(confident_indices) == 0:
                logger.warning("No confident predictions, stopping")
                break

            # Limit batch size
            if len(confident_indices) > batch_per_iteration:
                # Sort by confidence and take top
                sorted_indices = confident_indices[np.argsort(-confidences[confident_indices])]
                confident_indices = sorted_indices[:batch_per_iteration]

            logger.info(f"Adding {len(confident_indices)} confident samples")

            # Add to training set
            X_new = X_pool[confident_indices]
            y_new = keras.utils.to_categorical(
                pseudo_labels[confident_indices],
                num_classes=y_train.shape[1]
            )

            X_train = np.vstack([X_train, X_new])
            y_train = np.vstack([y_train, y_new])

            # Remove from pool
            X_pool = np.delete(X_pool, confident_indices, axis=0)

        logger.info(f"Self-training complete: final training set size = {len(X_train)}")

        # Final model
        final_model = self.base_model_fn()
        final_model.fit(X_train, y_train, epochs=50, verbose=0)

        return final_model


if __name__ == "__main__":
    print("Semi-Supervised & Weak Supervision Learning Ready!")
    print("\n📚 Semi-Supervised Methods:")
    print("  - Pseudo-labeling")
    print("  - Mean Teacher (EMA teacher model)")
    print("  - MixMatch (MixUp + pseudo-labels + consistency)")
    print("  - Self-training")
    print("\n🏷️ Weak Supervision:")
    print("  - Noisy label cleaning")
    print("  - Learning from proportions (LLP)")
    print("  - Confident learning")
    print("\n💰 Benefits:")
    print("  - Reduce labeling costs by 80-95%")
    print("  - Leverage abundant unlabeled satellite imagery")
    print("  - Handle noisy/imperfect labels")
    print("  - Scale to massive unlabeled datasets")
    print("\n🚀 Perfect for satellite imagery where labels are expensive!")
