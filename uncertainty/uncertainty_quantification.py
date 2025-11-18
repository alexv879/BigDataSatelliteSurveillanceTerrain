"""
Uncertainty Quantification and Out-of-Distribution Detection
Critical for trustworthy AI in satellite terrain classification
Know when the model is uncertain and when inputs are anomalous
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
from loguru import logger
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


class MonteCarlo Dropout:
    """
    Estimate uncertainty using Monte Carlo Dropout
    """

    def __init__(self, model: keras.Model, n_iterations: int = 50):
        """
        Initialize MC Dropout

        Args:
            model: Model with dropout layers
            n_iterations: Number of MC samples
        """
        self.model = model
        self.n_iterations = n_iterations

    def predict_with_uncertainty(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates

        Args:
            x: Input data

        Returns:
            (mean_predictions, uncertainty, epistemic_uncertainty)
        """
        # Run multiple forward passes with dropout enabled
        predictions = []

        for _ in range(self.n_iterations):
            # Keep dropout active during inference
            pred = self.model(x, training=True).numpy()
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_iterations, batch, classes)

        # Mean prediction
        mean_pred = predictions.mean(axis=0)

        # Predictive entropy (total uncertainty)
        epsilon = 1e-10
        predictive_entropy = -np.sum(
            mean_pred * np.log(mean_pred + epsilon),
            axis=-1
        )

        # Expected entropy (aleatoric uncertainty)
        expected_entropy = -np.mean(
            np.sum(predictions * np.log(predictions + epsilon), axis=-1),
            axis=0
        )

        # Mutual information (epistemic uncertainty)
        epistemic_uncertainty = predictive_entropy - expected_entropy

        logger.info(f"MC Dropout: {self.n_iterations} samples, mean uncertainty: {predictive_entropy.mean():.4f}")

        return mean_pred, predictive_entropy, epistemic_uncertainty


class EnsembleUncertainty:
    """
    Uncertainty from ensemble of models
    """

    def __init__(self, models: List[keras.Model]):
        """
        Initialize ensemble

        Args:
            models: List of trained models
        """
        self.models = models

    def predict_with_uncertainty(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with ensemble uncertainty

        Args:
            x: Input data

        Returns:
            (mean_prediction, disagreement)
        """
        # Get predictions from all models
        predictions = []

        for model in self.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_models, batch, classes)

        # Mean prediction
        mean_pred = predictions.mean(axis=0)

        # Variance-based disagreement
        variance = predictions.var(axis=0).mean(axis=-1)

        logger.info(f"Ensemble: {len(self.models)} models, mean disagreement: {variance.mean():.4f}")

        return mean_pred, variance


class ConformalPrediction:
    """
    Conformal prediction for calibrated prediction sets
    Provides statistical guarantees on coverage
    """

    def __init__(
        self,
        model: keras.Model,
        alpha: float = 0.1
    ):
        """
        Initialize conformal predictor

        Args:
            model: Trained model
            alpha: Miscoverage rate (1 - alpha is target coverage)
        """
        self.model = model
        self.alpha = alpha
        self.calibrated = False
        self.quantile = None

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ):
        """
        Calibrate on validation set

        Args:
            X_cal: Calibration inputs
            y_cal: Calibration labels (one-hot or indices)
        """
        # Get predictions
        predictions = self.model.predict(X_cal, verbose=0)

        # Compute conformity scores (1 - probability of true class)
        if y_cal.ndim == 2:  # One-hot
            true_class_probs = (predictions * y_cal).sum(axis=1)
        else:  # Class indices
            true_class_probs = predictions[np.arange(len(predictions)), y_cal]

        conformity_scores = 1 - true_class_probs

        # Compute quantile
        n = len(conformity_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(conformity_scores, q_level)

        self.calibrated = True

        logger.info(f"Conformal prediction calibrated: quantile = {self.quantile:.4f}")

    def predict_set(
        self,
        x: np.ndarray
    ) -> List[List[int]]:
        """
        Predict prediction sets with coverage guarantee

        Args:
            x: Input data

        Returns:
            List of prediction sets (class indices)
        """
        if not self.calibrated:
            raise ValueError("Must calibrate first")

        # Get predictions
        predictions = self.model.predict(x, verbose=0)

        # Build prediction sets
        prediction_sets = []

        for pred in predictions:
            # Include all classes with probability >= 1 - quantile
            pred_set = np.where(pred >= 1 - self.quantile)[0].tolist()
            prediction_sets.append(pred_set)

        avg_set_size = np.mean([len(s) for s in prediction_sets])
        logger.info(f"Prediction sets: avg size = {avg_set_size:.2f}")

        return prediction_sets


class OutOfDistributionDetector:
    """
    Detect out-of-distribution (OOD) samples
    """

    def __init__(
        self,
        model: keras.Model,
        method: str = 'mahalanobis'
    ):
        """
        Initialize OOD detector

        Args:
            model: Feature extractor
            method: 'mahalanobis', 'isolation_forest', or 'energy'
        """
        self.model = model
        self.method = method
        self.fitted = False

        # Create feature extractor (penultimate layer)
        self.feature_extractor = keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )

    def fit(
        self,
        X_train: np.ndarray
    ):
        """
        Fit OOD detector on in-distribution data

        Args:
            X_train: In-distribution training data
        """
        # Extract features
        features = self.feature_extractor.predict(X_train, verbose=0)

        if self.method == 'mahalanobis':
            # Fit Mahalanobis distance
            self.mean = features.mean(axis=0)
            self.cov_inv = np.linalg.pinv(np.cov(features.T))

        elif self.method == 'isolation_forest':
            # Fit Isolation Forest
            self.detector = IsolationForest(contamination=0.1, random_state=42)
            self.detector.fit(features)

        elif self.method == 'elliptic_envelope':
            # Fit Elliptic Envelope
            self.detector = EllipticEnvelope(contamination=0.1, random_state=42)
            self.detector.fit(features)

        elif self.method == 'energy':
            # Energy-based (no fitting needed, uses logits)
            pass

        self.fitted = True

        logger.info(f"OOD detector fitted: {self.method}")

    def detect(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect OOD samples

        Args:
            X: Input data
            threshold: Detection threshold (None for auto)

        Returns:
            (ood_scores, is_ood)
        """
        if not self.fitted and self.method != 'energy':
            raise ValueError("Must fit detector first")

        if self.method == 'mahalanobis':
            # Extract features
            features = self.feature_extractor.predict(X, verbose=0)

            # Compute Mahalanobis distance
            diff = features - self.mean
            ood_scores = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))

            # Threshold
            if threshold is None:
                threshold = np.percentile(ood_scores, 95)

        elif self.method == 'isolation_forest':
            features = self.feature_extractor.predict(X, verbose=0)
            ood_scores = -self.detector.score_samples(features)
            if threshold is None:
                threshold = 0

        elif self.method == 'elliptic_envelope':
            features = self.feature_extractor.predict(X, verbose=0)
            ood_scores = -self.detector.score_samples(features)
            if threshold is None:
                threshold = 0

        elif self.method == 'energy':
            # Energy score: -log(sum(exp(logits)))
            logits = self.model.predict(X, verbose=0)
            ood_scores = -np.log(np.sum(np.exp(logits), axis=1))

            if threshold is None:
                threshold = np.percentile(ood_scores, 95)

        is_ood = ood_scores > threshold

        logger.info(f"OOD detection: {is_ood.sum()} / {len(is_ood)} flagged as OOD")

        return ood_scores, is_ood


class CalibrationMetrics:
    """
    Evaluate model calibration
    """

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE)

        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities
            n_bins: Number of bins

        Returns:
            ECE score
        """
        # Get predicted class and confidence
        pred_class = np.argmax(y_pred_probs, axis=1)
        confidence = np.max(y_pred_probs, axis=1)

        # Convert y_true to indices if one-hot
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        # Bin confidences
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidence, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute ECE
        ece = 0.0
        n_samples = len(y_true)

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_accuracy = (pred_class[mask] == y_true[mask]).mean()
                bin_confidence = confidence[mask].mean()
                bin_size = mask.sum()

                ece += (bin_size / n_samples) * np.abs(bin_accuracy - bin_confidence)

        logger.info(f"ECE: {ece:.4f}")

        return float(ece)

    @staticmethod
    def reliability_diagram_data(
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Get data for reliability diagram

        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities
            n_bins: Number of bins

        Returns:
            Dictionary with bin data
        """
        # Get confidence
        pred_class = np.argmax(y_pred_probs, axis=1)
        confidence = np.max(y_pred_probs, axis=1)

        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        # Bin confidences
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidence, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute per-bin statistics
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_accuracies.append((pred_class[mask] == y_true[mask]).mean())
                bin_confidences.append(confidence[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)

        return {
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts),
            'bin_edges': bin_edges
        }


class TemperatureScaling:
    """
    Post-hoc calibration using temperature scaling
    """

    def __init__(self):
        """Initialize temperature scaling"""
        self.temperature = 1.0
        self.calibrated = False

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        learning_rate: float = 0.01,
        max_iter: int = 100
    ):
        """
        Learn optimal temperature

        Args:
            logits: Model logits (before softmax)
            y_true: True labels
            learning_rate: Learning rate
            max_iter: Maximum iterations
        """
        import tensorflow as tf

        # Convert to tensors
        logits_tf = tf.constant(logits, dtype=tf.float32)

        if y_true.ndim == 1:
            y_true_tf = tf.one_hot(y_true, depth=logits.shape[1])
        else:
            y_true_tf = tf.constant(y_true, dtype=tf.float32)

        # Optimize temperature
        temperature = tf.Variable(1.0, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                # Apply temperature scaling
                scaled_logits = logits_tf / temperature

                # Compute loss (cross-entropy)
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        y_true_tf,
                        scaled_logits,
                        from_logits=True
                    )
                )

            # Update temperature
            gradients = tape.gradient(loss, [temperature])
            optimizer.apply_gradients(zip(gradients, [temperature]))

        self.temperature = float(temperature.numpy())
        self.calibrated = True

        logger.info(f"Temperature scaling: T = {self.temperature:.4f}")

    def calibrate_probabilities(
        self,
        logits: np.ndarray
    ) -> np.ndarray:
        """
        Apply temperature scaling to logits

        Args:
            logits: Model logits

        Returns:
            Calibrated probabilities
        """
        if not self.calibrated:
            logger.warning("Temperature not fitted, using T=1.0")

        # Scale and apply softmax
        scaled_logits = logits / self.temperature
        calibrated_probs = tf.nn.softmax(scaled_logits).numpy()

        return calibrated_probs


if __name__ == "__main__":
    print("Uncertainty Quantification Ready!")
    print("\n📊 Uncertainty Estimation:")
    print("  - Monte Carlo Dropout (epistemic + aleatoric)")
    print("  - Ensemble disagreement")
    print("  - Predictive entropy")
    print("\n🎯 Conformal Prediction:")
    print("  - Statistical coverage guarantees")
    print("  - Prediction sets instead of point predictions")
    print("  - Calibrated uncertainty")
    print("\n🚨 Out-of-Distribution Detection:")
    print("  - Mahalanobis distance")
    print("  - Isolation Forest")
    print("  - Energy-based detection")
    print("  - Elliptic Envelope")
    print("\n📈 Calibration:")
    print("  - Expected Calibration Error (ECE)")
    print("  - Temperature scaling")
    print("  - Reliability diagrams")
    print("\n✅ Trustworthy AI for satellite terrain classification!")
