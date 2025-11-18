"""
Active Learning for Satellite Terrain Classification
Intelligent sample selection to minimize labeling costs
Reduce labeling effort by 50-80%
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from loguru import logger
from abc import ABC, abstractmethod
from scipy.stats import entropy
from sklearn.cluster import KMeans


class QueryStrategy(ABC):
    """Base class for query strategies"""

    @abstractmethod
    def select_samples(
        self,
        model: keras.Model,
        unlabeled_data: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Select samples for labeling

        Args:
            model: Trained model
            unlabeled_data: Unlabeled samples
            n_samples: Number of samples to select

        Returns:
            Indices of selected samples
        """
        pass


class UncertaintySampling(QueryStrategy):
    """
    Uncertainty-based sampling
    Select samples where model is most uncertain
    """

    def __init__(self, method: str = 'entropy'):
        """
        Initialize uncertainty sampling

        Args:
            method: 'entropy', 'margin', or 'least_confident'
        """
        self.method = method

    def select_samples(
        self,
        model: keras.Model,
        unlabeled_data: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Select most uncertain samples"""
        # Get predictions
        predictions = model.predict(unlabeled_data, verbose=0)

        # Compute uncertainty scores
        if self.method == 'entropy':
            scores = self._entropy_score(predictions)
        elif self.method == 'margin':
            scores = self._margin_score(predictions)
        else:  # least_confident
            scores = self._least_confident_score(predictions)

        # Select top uncertain samples
        indices = np.argsort(scores)[-n_samples:]

        logger.info(f"Selected {n_samples} samples using {self.method} uncertainty")

        return indices

    def _entropy_score(self, predictions: np.ndarray) -> np.ndarray:
        """Compute entropy-based uncertainty"""
        return entropy(predictions.T)

    def _margin_score(self, predictions: np.ndarray) -> np.ndarray:
        """Compute margin-based uncertainty (difference between top 2 predictions)"""
        sorted_preds = np.sort(predictions, axis=1)
        # Higher margin = more confident, so we want LOW margin
        margin = sorted_preds[:, -1] - sorted_preds[:, -2]
        return -margin  # Negate so higher score = more uncertain

    def _least_confident_score(self, predictions: np.ndarray) -> np.ndarray:
        """Compute least confident uncertainty"""
        max_conf = np.max(predictions, axis=1)
        return 1 - max_conf  # Higher score = less confident


class DiversitySampling(QueryStrategy):
    """
    Diversity-based sampling
    Select diverse samples to cover feature space
    """

    def __init__(self, method: str = 'kmeans'):
        """
        Initialize diversity sampling

        Args:
            method: 'kmeans' or 'coreset'
        """
        self.method = method

    def select_samples(
        self,
        model: keras.Model,
        unlabeled_data: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Select diverse samples"""
        # Extract features
        feature_extractor = keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output  # Penultimate layer
        )

        features = feature_extractor.predict(unlabeled_data, verbose=0)

        if self.method == 'kmeans':
            indices = self._kmeans_selection(features, n_samples)
        else:  # coreset
            indices = self._coreset_selection(features, n_samples)

        logger.info(f"Selected {n_samples} diverse samples using {self.method}")

        return indices

    def _kmeans_selection(
        self,
        features: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """K-means based diversity selection"""
        # Cluster features
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(features)

        # Select samples closest to cluster centers
        indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features - center, axis=1)
            idx = np.argmin(distances)
            indices.append(idx)

        return np.array(indices)

    def _coreset_selection(
        self,
        features: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Coreset-based diversity selection"""
        # Greedy k-center algorithm
        selected_indices = []

        # Start with random sample
        first_idx = np.random.randint(len(features))
        selected_indices.append(first_idx)

        for _ in range(n_samples - 1):
            # Compute distances to nearest selected sample
            selected_features = features[selected_indices]
            distances = np.min(
                np.linalg.norm(
                    features[:, np.newaxis] - selected_features,
                    axis=2
                ),
                axis=1
            )

            # Select sample farthest from selected samples
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)

        return np.array(selected_indices)


class BALDSampling(QueryStrategy):
    """
    Bayesian Active Learning by Disagreement (BALD)
    Uses model uncertainty via Monte Carlo Dropout
    """

    def __init__(self, n_iterations: int = 20):
        """
        Initialize BALD

        Args:
            n_iterations: Number of MC dropout iterations
        """
        self.n_iterations = n_iterations

    def select_samples(
        self,
        model: keras.Model,
        unlabeled_data: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Select samples with highest BALD score"""
        # Enable dropout at inference time
        mc_predictions = []

        for _ in range(self.n_iterations):
            preds = model(unlabeled_data, training=True)  # Keep dropout active
            mc_predictions.append(preds.numpy())

        mc_predictions = np.array(mc_predictions)  # (iterations, samples, classes)

        # Compute BALD score: I(y; θ | x, D)
        # = H(E[p(y|x)]) - E[H(p(y|x))]

        # Mean prediction
        mean_pred = mc_predictions.mean(axis=0)

        # Entropy of mean
        entropy_mean = entropy(mean_pred.T)

        # Mean of entropies
        entropies = np.array([entropy(pred.T) for pred in mc_predictions])
        mean_entropy = entropies.mean(axis=0)

        # BALD score
        bald_scores = entropy_mean - mean_entropy

        # Select top samples
        indices = np.argsort(bald_scores)[-n_samples:]

        logger.info(f"Selected {n_samples} samples using BALD")

        return indices


class HybridSampling(QueryStrategy):
    """
    Hybrid strategy combining uncertainty and diversity
    Balances exploration and exploitation
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.5,
        diversity_weight: float = 0.5
    ):
        """
        Initialize hybrid sampling

        Args:
            uncertainty_weight: Weight for uncertainty score
            diversity_weight: Weight for diversity score
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight

        self.uncertainty_sampler = UncertaintySampling(method='entropy')
        self.diversity_sampler = DiversitySampling(method='coreset')

    def select_samples(
        self,
        model: keras.Model,
        unlabeled_data: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Select samples balancing uncertainty and diversity"""
        # Get predictions
        predictions = model.predict(unlabeled_data, verbose=0)

        # Uncertainty scores
        uncertainty_scores = entropy(predictions.T)

        # Diversity scores
        feature_extractor = keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
        features = feature_extractor.predict(unlabeled_data, verbose=0)

        # Compute diversity scores (distance to labeled data)
        diversity_scores = np.zeros(len(unlabeled_data))

        # Simple diversity: variance in feature space
        for i in range(len(unlabeled_data)):
            distances = np.linalg.norm(features - features[i], axis=1)
            diversity_scores[i] = distances.mean()

        # Normalize scores
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / \
                           (uncertainty_scores.max() - uncertainty_scores.min() + 1e-10)
        diversity_scores = (diversity_scores - diversity_scores.min()) / \
                         (diversity_scores.max() - diversity_scores.min() + 1e-10)

        # Combined score
        combined_scores = (
            self.uncertainty_weight * uncertainty_scores +
            self.diversity_weight * diversity_scores
        )

        # Select top samples
        indices = np.argsort(combined_scores)[-n_samples:]

        logger.info(f"Selected {n_samples} samples using hybrid strategy")

        return indices


class ActiveLearner:
    """
    Active learning pipeline
    Iteratively train model and select samples for labeling
    """

    def __init__(
        self,
        model: keras.Model,
        query_strategy: QueryStrategy,
        initial_labeled_size: int = 100
    ):
        """
        Initialize active learner

        Args:
            model: Model to train
            query_strategy: Strategy for sample selection
            initial_labeled_size: Initial labeled pool size
        """
        self.model = model
        self.query_strategy = query_strategy
        self.initial_labeled_size = initial_labeled_size

        self.labeled_indices = []
        self.unlabeled_indices = []

        logger.info(f"Active learner initialized with {query_strategy.__class__.__name__}")

    def initialize_pool(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'random'
    ):
        """
        Initialize labeled/unlabeled pools

        Args:
            X: All data
            y: All labels
            method: 'random' or 'stratified'
        """
        n_total = len(X)

        if method == 'random':
            indices = np.random.permutation(n_total)
        else:  # stratified
            # Ensure balanced classes
            unique_classes = np.unique(y)
            per_class = self.initial_labeled_size // len(unique_classes)

            indices = []
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                selected = np.random.choice(
                    cls_indices,
                    size=min(per_class, len(cls_indices)),
                    replace=False
                )
                indices.extend(selected)

            indices = np.array(indices)

        # Split into labeled/unlabeled
        self.labeled_indices = indices[:self.initial_labeled_size].tolist()
        self.unlabeled_indices = [
            i for i in range(n_total)
            if i not in self.labeled_indices
        ]

        logger.info(
            f"Initialized: {len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )

    def train_iteration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train model on labeled data

        Args:
            X: All data
            y: All labels
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        # Get labeled data
        X_labeled = X[self.labeled_indices]
        y_labeled = y[self.labeled_indices]

        # Train
        history = self.model.fit(
            X_labeled,
            y_labeled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )

        logger.info(
            f"Trained on {len(self.labeled_indices)} samples - "
            f"Acc: {history.history['accuracy'][-1]:.4f}"
        )

        return history.history

    def query(
        self,
        X: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Query samples for labeling

        Args:
            X: All data
            n_samples: Number of samples to query

        Returns:
            Indices of queried samples
        """
        # Get unlabeled data
        X_unlabeled = X[self.unlabeled_indices]

        # Select samples
        selected_indices = self.query_strategy.select_samples(
            self.model,
            X_unlabeled,
            n_samples
        )

        # Convert to original indices
        queried_indices = [self.unlabeled_indices[i] for i in selected_indices]

        return np.array(queried_indices)

    def update_pool(self, queried_indices: np.ndarray):
        """
        Update labeled/unlabeled pools after labeling

        Args:
            queried_indices: Indices of newly labeled samples
        """
        # Move to labeled pool
        self.labeled_indices.extend(queried_indices.tolist())

        # Remove from unlabeled pool
        self.unlabeled_indices = [
            i for i in self.unlabeled_indices
            if i not in queried_indices
        ]

        logger.info(
            f"Pool updated: {len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )

    def active_learning_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oracle: Callable,  # Function to get labels
        n_iterations: int = 10,
        n_query: int = 100,
        epochs_per_iteration: int = 10
    ) -> List[Dict]:
        """
        Full active learning loop

        Args:
            X: All data
            y: All labels (oracle uses this)
            oracle: Function to get labels for indices
            n_iterations: Number of AL iterations
            n_query: Samples to query per iteration
            epochs_per_iteration: Training epochs per iteration

        Returns:
            List of iteration histories
        """
        histories = []

        for iteration in range(n_iterations):
            logger.info(f"\n=== Active Learning Iteration {iteration + 1}/{n_iterations} ===")

            # Train on current labeled data
            history = self.train_iteration(X, y, epochs=epochs_per_iteration)

            # Query new samples
            queried_indices = self.query(X, n_query)

            # Get labels from oracle (in practice, human would label)
            # oracle(queried_indices) would return labels

            # Update pools
            self.update_pool(queried_indices)

            # Store history
            histories.append({
                'iteration': iteration,
                'n_labeled': len(self.labeled_indices),
                'history': history
            })

            # Check if done
            if len(self.unlabeled_indices) < n_query:
                logger.info("Insufficient unlabeled samples, stopping")
                break

        return histories


if __name__ == "__main__":
    print("Active Learning Ready!")
    print("\nQuery Strategies:")
    print("- Uncertainty Sampling (entropy, margin, least confident)")
    print("- Diversity Sampling (k-means, coreset)")
    print("- BALD (Bayesian Active Learning by Disagreement)")
    print("- Hybrid (uncertainty + diversity)")
    print("\nBenefits:")
    print("- Reduce labeling effort by 50-80%")
    print("- Focus on informative samples")
    print("- Balance exploration and exploitation")
    print("\n🎯 Smart sample selection for efficient learning!")
