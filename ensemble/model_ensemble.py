"""
Advanced Model Ensemble Methods
Combines multiple models for superior performance
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger


class WeightedEnsemble:
    """Weighted ensemble of multiple models"""

    def __init__(self, models: List[keras.Model], weights: Optional[List[float]] = None):
        """
        Initialize weighted ensemble

        Args:
            models: List of trained models
            weights: Optional weights for each model (normalized automatically)
        """
        self.models = models

        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        logger.info(f"Weighted ensemble created with {len(models)} models")
        logger.info(f"Weights: {self.weights}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using weighted ensemble

        Args:
            x: Input data

        Returns:
            Ensemble predictions
        """
        predictions = []

        for model in self.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += pred * weight

        return ensemble_pred

    def predict_with_uncertainty(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation

        Args:
            x: Input data

        Returns:
            mean_prediction, std_prediction
        """
        predictions = []

        for model in self.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted mean
        mean_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            mean_pred += pred * weight

        # Weighted std
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred


class StackingEnsemble:
    """Stacking ensemble with meta-learner"""

    def __init__(
        self,
        base_models: List[keras.Model],
        meta_learner: Optional[keras.Model] = None
    ):
        """
        Initialize stacking ensemble

        Args:
            base_models: List of base models
            meta_learner: Meta-learner model (trained on base model predictions)
        """
        self.base_models = base_models
        self.meta_learner = meta_learner
        logger.info(f"Stacking ensemble with {len(base_models)} base models")

    def create_meta_features(self, x: np.ndarray) -> np.ndarray:
        """
        Create meta-features from base model predictions

        Args:
            x: Input data

        Returns:
            Meta-features
        """
        meta_features = []

        for model in self.base_models:
            pred = model.predict(x, verbose=0)
            meta_features.append(pred)

        # Concatenate predictions
        return np.concatenate(meta_features, axis=-1)

    def train_meta_learner(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50
    ):
        """
        Train meta-learner on base model predictions

        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            epochs: Training epochs
        """
        logger.info("Generating meta-features for meta-learner training...")

        # Generate meta-features
        meta_train = self.create_meta_features(x_train)
        meta_val = self.create_meta_features(x_val)

        # Create meta-learner if not provided
        if self.meta_learner is None:
            num_classes = y_train.shape[1]
            input_dim = meta_train.shape[1]

            self.meta_learner = keras.Sequential([
                keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation='softmax')
            ])

            self.meta_learner.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        logger.info("Training meta-learner...")

        # Train meta-learner
        history = self.meta_learner.fit(
            meta_train, y_train,
            validation_data=(meta_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )

        logger.info("Meta-learner training completed")
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using stacking ensemble

        Args:
            x: Input data

        Returns:
            Ensemble predictions
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner first.")

        # Generate meta-features
        meta_features = self.create_meta_features(x)

        # Predict with meta-learner
        return self.meta_learner.predict(meta_features, verbose=0)

    def save(self, save_dir: str):
        """Save stacking ensemble"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save base models
        for i, model in enumerate(self.base_models):
            model.save(save_path / f"base_model_{i}.h5")

        # Save meta-learner
        if self.meta_learner:
            self.meta_learner.save(save_path / "meta_learner.h5")

        logger.info(f"Stacking ensemble saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> 'StackingEnsemble':
        """Load stacking ensemble"""
        save_path = Path(save_dir)

        # Load base models
        base_models = []
        i = 0
        while (save_path / f"base_model_{i}.h5").exists():
            model = keras.models.load_model(save_path / f"base_model_{i}.h5")
            base_models.append(model)
            i += 1

        # Load meta-learner
        meta_learner = None
        if (save_path / "meta_learner.h5").exists():
            meta_learner = keras.models.load_model(save_path / "meta_learner.h5")

        logger.info(f"Loaded stacking ensemble from {save_dir}")
        return cls(base_models, meta_learner)


class SnapshotEnsemble:
    """Snapshot ensemble using cyclic learning rate"""

    def __init__(self, model: keras.Model, num_snapshots: int = 5):
        """
        Initialize snapshot ensemble

        Args:
            model: Base model architecture
            num_snapshots: Number of snapshots to save
        """
        self.base_model = model
        self.num_snapshots = num_snapshots
        self.snapshots = []
        logger.info(f"Snapshot ensemble with {num_snapshots} snapshots")

    def get_cyclic_lr_callback(
        self,
        base_lr: float = 1e-4,
        max_lr: float = 1e-2,
        step_size: int = 2000
    ) -> keras.callbacks.LearningRateScheduler:
        """Get cyclic learning rate callback"""

        def cyclic_lr(epoch, lr):
            cycle = np.floor(1 + epoch / (2 * step_size))
            x = abs(epoch / step_size - 2 * cycle + 1)
            new_lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
            return new_lr

        return keras.callbacks.LearningRateScheduler(cyclic_lr)

    def train_with_snapshots(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        total_epochs: int = 100
    ):
        """
        Train with cyclic learning rate and save snapshots

        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            total_epochs: Total training epochs
        """
        snapshot_interval = total_epochs // self.num_snapshots

        for i in range(self.num_snapshots):
            logger.info(f"Training snapshot {i+1}/{self.num_snapshots}...")

            # Train for snapshot interval
            self.base_model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=snapshot_interval,
                callbacks=[self.get_cyclic_lr_callback()],
                verbose=1
            )

            # Clone and save snapshot
            snapshot = keras.models.clone_model(self.base_model)
            snapshot.set_weights(self.base_model.get_weights())
            self.snapshots.append(snapshot)

            logger.info(f"Snapshot {i+1} saved")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using snapshot ensemble

        Args:
            x: Input data

        Returns:
            Ensemble predictions
        """
        predictions = []

        for snapshot in self.snapshots:
            pred = snapshot.predict(x, verbose=0)
            predictions.append(pred)

        # Average predictions
        return np.mean(predictions, axis=0)


class BayesianModelAveraging:
    """Bayesian Model Averaging for ensemble"""

    def __init__(self, models: List[keras.Model]):
        """
        Initialize BMA

        Args:
            models: List of trained models
        """
        self.models = models
        self.weights = None
        logger.info(f"BMA with {len(models)} models")

    def fit_weights(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray
    ):
        """
        Fit BMA weights based on validation performance

        Args:
            x_val: Validation data
            y_val: Validation labels
        """
        likelihoods = []

        for model in self.models:
            pred = model.predict(x_val, verbose=0)

            # Compute log-likelihood
            log_likelihood = np.sum(
                y_val * np.log(pred + 1e-10),
                axis=1
            ).mean()

            likelihoods.append(np.exp(log_likelihood))

        # Normalize to get weights
        total = sum(likelihoods)
        self.weights = [l / total for l in likelihoods]

        logger.info(f"BMA weights fitted: {self.weights}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using BMA

        Args:
            x: Input data

        Returns:
            BMA predictions
        """
        if self.weights is None:
            raise ValueError("Weights not fitted. Call fit_weights first.")

        predictions = []

        for model in self.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)

        # Weighted average
        bma_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            bma_pred += pred * weight

        return bma_pred


def create_ensemble_from_checkpoints(
    checkpoint_dir: str,
    ensemble_type: str = "weighted"
) -> WeightedEnsemble:
    """
    Create ensemble from saved checkpoints

    Args:
        checkpoint_dir: Directory with model checkpoints
        ensemble_type: Type of ensemble to create

    Returns:
        Ensemble model
    """
    checkpoint_path = Path(checkpoint_dir)
    model_files = sorted(checkpoint_path.glob("*.h5"))

    if not model_files:
        raise ValueError(f"No model checkpoints found in {checkpoint_dir}")

    logger.info(f"Loading {len(model_files)} models from {checkpoint_dir}")

    models = []
    for model_file in model_files:
        model = keras.models.load_model(model_file)
        models.append(model)

    if ensemble_type == "weighted":
        return WeightedEnsemble(models)
    elif ensemble_type == "stacking":
        return StackingEnsemble(models)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def evaluate_ensemble(
    ensemble: WeightedEnsemble,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate ensemble performance

    Args:
        ensemble: Ensemble model
        x_test: Test data
        y_test: Test labels

    Returns:
        Evaluation metrics
    """
    predictions = ensemble.predict(x_test)

    # Accuracy
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    # Top-3 accuracy
    top3_pred = np.argsort(predictions, axis=1)[:, -3:]
    top3_accuracy = np.mean([
        true_classes[i] in top3_pred[i]
        for i in range(len(true_classes))
    ])

    metrics = {
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),
    }

    logger.info(f"Ensemble metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    print("Advanced Ensemble Methods Ready")
    print("Features:")
    print("- Weighted Ensemble: Simple averaging with weights")
    print("- Stacking Ensemble: Meta-learner on base predictions")
    print("- Snapshot Ensemble: Cyclic learning rate snapshots")
    print("- Bayesian Model Averaging: Probabilistic weighting")
