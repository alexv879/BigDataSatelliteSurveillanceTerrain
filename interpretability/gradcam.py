"""
Model Interpretability: GradCAM, Attention Visualization, SHAP
Explain model predictions for satellite terrain classification
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Visualize which regions of the image are important for predictions
    """

    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize GradCAM

        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to visualize
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()

    def _find_target_layer(self) -> str:
        """Automatically find the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # Conv layer
                return layer.name
        raise ValueError("Could not find convolutional layer")

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap

        Args:
            image: Input image (preprocessed)
            class_idx: Target class index (if None, uses predicted class)
            eps: Small constant for numerical stability

        Returns:
            Heatmap array
        """
        # Create gradient model
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image[np.newaxis, ...])
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]

        # Compute gradient of class score with respect to feature map
        grads = tape.gradient(class_channel, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image

        Args:
            heatmap: GradCAM heatmap
            image: Original image
            alpha: Transparency of heatmap
            colormap: OpenCV colormap

        Returns:
            Superimposed image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Normalize image
        if image.max() <= 1.0:
            image = np.uint8(255 * image)

        # Superimpose
        superimposed = heatmap * alpha + image * (1 - alpha)
        return np.uint8(superimposed)

    def visualize(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize GradCAM

        Args:
            image: Input image
            class_idx: Target class index
            save_path: Path to save visualization
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(image, class_idx)

        # Overlay on image
        superimposed = self.overlay_heatmap(heatmap, image)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("GradCAM Heatmap")
        axes[1].axis('off')

        axes[2].imshow(superimposed)
        axes[2].set_title("GradCAM Overlay")
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class AttentionVisualization:
    """Visualize attention weights for Vision Transformers"""

    def __init__(self, model: keras.Model):
        self.model = model

    def extract_attention_weights(
        self,
        image: np.ndarray,
        layer_name: str = "multi_head_attention"
    ) -> np.ndarray:
        """
        Extract attention weights from transformer

        Args:
            image: Input image
            layer_name: Name of attention layer

        Returns:
            Attention weights
        """
        # Find attention layers
        attention_layers = [
            layer for layer in self.model.layers
            if layer_name in layer.name.lower()
        ]

        if not attention_layers:
            raise ValueError(f"No attention layers found with name: {layer_name}")

        # Create model to extract attention
        attention_model = keras.Model(
            inputs=self.model.input,
            outputs=[layer.output for layer in attention_layers]
        )

        # Get attention outputs
        attention_outputs = attention_model(image[np.newaxis, ...])

        return attention_outputs

    def visualize_attention_map(
        self,
        image: np.ndarray,
        attention_weights: np.ndarray,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention map

        Args:
            image: Original image
            attention_weights: Attention weights from transformer
            head_idx: Which attention head to visualize
            save_path: Path to save visualization
        """
        # Extract attention for specific head
        attn = attention_weights[0, head_idx]

        # Reshape to 2D grid
        grid_size = int(np.sqrt(attn.shape[0] - 1))  # Exclude CLS token
        attn_map = attn[1:, 1:].mean(axis=0).reshape(grid_size, grid_size)

        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        # Resize to image size
        attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(attn_map, cmap='viridis')
        axes[1].set_title(f"Attention Map (Head {head_idx})")
        axes[1].axis('off')

        # Overlay
        overlay = np.uint8(255 * image)
        heatmap = np.uint8(255 * attn_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed = heatmap * 0.4 + overlay * 0.6

        axes[2].imshow(np.uint8(superimposed))
        axes[2].set_title("Attention Overlay")
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class UncertaintyEstimator:
    """Estimate prediction uncertainty using Monte Carlo Dropout"""

    def __init__(self, model: keras.Model, n_iterations: int = 100):
        """
        Initialize uncertainty estimator

        Args:
            model: Keras model with dropout layers
            n_iterations: Number of MC dropout iterations
        """
        self.model = model
        self.n_iterations = n_iterations

    def predict_with_uncertainty(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict with uncertainty estimation

        Args:
            image: Input image

        Returns:
            mean_prediction: Mean prediction across iterations
            std_prediction: Standard deviation of predictions
            entropy: Prediction entropy (uncertainty measure)
        """
        predictions = []

        # Run multiple forward passes with dropout enabled
        for _ in range(self.n_iterations):
            pred = self.model(image[np.newaxis, ...], training=True)
            predictions.append(pred.numpy())

        predictions = np.array(predictions).squeeze()

        # Compute statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        # Compute entropy
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10))

        return mean_pred, std_pred, entropy

    def visualize_uncertainty(
        self,
        image: np.ndarray,
        class_names: list,
        save_path: Optional[str] = None
    ):
        """
        Visualize prediction uncertainty

        Args:
            image: Input image
            class_names: List of class names
            save_path: Path to save visualization
        """
        mean_pred, std_pred, entropy = self.predict_with_uncertainty(image)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Display image
        axes[0].imshow(image)
        axes[0].set_title(f"Prediction Entropy: {entropy:.3f}")
        axes[0].axis('off')

        # Plot predictions with uncertainty
        y_pos = np.arange(len(class_names))
        axes[1].barh(y_pos, mean_pred, xerr=std_pred, alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(class_names)
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Predictions with Uncertainty')
        axes[1].set_xlim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def explain_prediction(
    model: keras.Model,
    image: np.ndarray,
    class_names: list,
    method: str = "gradcam"
) -> dict:
    """
    Comprehensive prediction explanation

    Args:
        model: Keras model
        image: Input image
        class_names: List of class names
        method: Explanation method ('gradcam', 'attention', 'uncertainty')

    Returns:
        Dictionary with explanations
    """
    # Get prediction
    pred = model.predict(image[np.newaxis, ...])[0]
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]

    result = {
        "predicted_class": class_names[pred_class],
        "confidence": float(confidence),
        "all_probabilities": {class_names[i]: float(pred[i]) for i in range(len(pred))}
    }

    # Add visual explanation
    if method == "gradcam":
        gradcam = GradCAM(model)
        heatmap = gradcam.compute_heatmap(image, pred_class)
        result["heatmap"] = heatmap

    elif method == "uncertainty":
        uncertainty = UncertaintyEstimator(model)
        mean_pred, std_pred, entropy = uncertainty.predict_with_uncertainty(image)
        result["uncertainty"] = {
            "mean_prediction": mean_pred.tolist(),
            "std_prediction": std_pred.tolist(),
            "entropy": float(entropy)
        }

    return result


if __name__ == "__main__":
    print("Model Interpretability Tools Ready")
    print("Features:")
    print("- GradCAM: Visualize important image regions")
    print("- Attention Visualization: For transformer models")
    print("- Uncertainty Estimation: Monte Carlo Dropout")
    print("- Comprehensive explanations")
