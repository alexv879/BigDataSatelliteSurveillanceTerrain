"""
Temporal Change Detection for Satellite Imagery
Detect deforestation, urbanization, flooding, and other changes over time
Critical for real-world terrain surveillance
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Dict, Optional
from loguru import logger
from datetime import datetime


class SiameseChangeDetector(keras.Model):
    """
    Siamese network for bi-temporal change detection
    Detects changes between two time periods
    """

    def __init__(self, backbone: str = 'resnet50', num_classes: int = 2):
        super().__init__()
        self.backbone_name = backbone

        # Shared encoder
        if backbone == 'resnet50':
            self.encoder = keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3),
                pooling=None
            )
        else:
            self.encoder = self._build_custom_encoder()

        # Freeze encoder initially
        self.encoder.trainable = False

        # Decoder for segmentation
        self.decoder = self._build_decoder()

        # Change detection head
        self.change_head = keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(num_classes, 1, activation='softmax')
        ])

        logger.info(f"Siamese change detector created with {backbone}")

    def _build_custom_encoder(self):
        """Build custom encoder"""
        return keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
        ])

    def _build_decoder(self):
        """Build decoder for upsampling"""
        return keras.Sequential([
            layers.UpSampling2D(2),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.UpSampling2D(2),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.UpSampling2D(2),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
        ])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None):
        """
        Forward pass

        Args:
            inputs: Tuple of (t1_image, t2_image)
            training: Training mode

        Returns:
            Change map
        """
        t1_image, t2_image = inputs

        # Encode both images with shared weights
        t1_features = self.encoder(t1_image, training=training)
        t2_features = self.encoder(t2_image, training=training)

        # Compute difference
        diff = tf.abs(t1_features - t2_features)

        # Decode
        decoded = self.decoder(diff, training=training)

        # Change detection
        change_map = self.change_head(decoded, training=training)

        return change_map


class TimeSeriesAnalyzer:
    """
    Analyze time series of satellite images
    Detect trends, seasonal patterns, anomalies
    """

    def __init__(self, model: keras.Model):
        self.model = model

    def detect_trends(
        self,
        image_sequence: List[np.ndarray],
        timestamps: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """
        Detect temporal trends in imagery

        Args:
            image_sequence: List of images over time
            timestamps: Corresponding timestamps

        Returns:
            Trend analysis results
        """
        # Extract features for each image
        features = []
        for img in image_sequence:
            feat = self.model.predict(img[np.newaxis, ...], verbose=0)
            features.append(feat[0])

        features = np.array(features)

        # Compute trends
        results = {
            'mean_trend': np.mean(features, axis=0),
            'std_trend': np.std(features, axis=0),
            'linear_trend': self._fit_linear_trend(features, timestamps)
        }

        return results

    def _fit_linear_trend(
        self,
        features: np.ndarray,
        timestamps: List[datetime]
    ) -> np.ndarray:
        """Fit linear trend"""
        # Convert timestamps to numeric
        t0 = timestamps[0]
        time_numeric = np.array([(t - t0).days for t in timestamps])

        # Fit linear regression for each feature
        trends = []
        for feat_idx in range(features.shape[1]):
            slope = np.polyfit(time_numeric, features[:, feat_idx], 1)[0]
            trends.append(slope)

        return np.array(trends)

    def detect_anomalies(
        self,
        image_sequence: List[np.ndarray],
        threshold: float = 2.0
    ) -> List[int]:
        """
        Detect anomalous images in sequence

        Args:
            image_sequence: Sequence of images
            threshold: Anomaly threshold (std deviations)

        Returns:
            Indices of anomalous images
        """
        # Extract features
        features = []
        for img in image_sequence:
            feat = self.model.predict(img[np.newaxis, ...], verbose=0)
            features.append(feat[0])

        features = np.array(features)

        # Compute distances from mean
        mean_feat = features.mean(axis=0)
        distances = np.linalg.norm(features - mean_feat, axis=1)

        # Find anomalies
        mean_dist = distances.mean()
        std_dist = distances.std()

        anomaly_indices = np.where(distances > mean_dist + threshold * std_dist)[0]

        logger.info(f"Detected {len(anomaly_indices)} anomalies")

        return anomaly_indices.tolist()


class LSTMChangePredictor(keras.Model):
    """
    LSTM-based change predictor
    Predicts future changes based on historical patterns
    """

    def __init__(self, feature_dim: int = 512, sequence_length: int = 10):
        super().__init__()

        self.encoder = keras.Sequential([
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.GlobalAveragePooling2D()
        ])

        self.lstm = keras.Sequential([
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(128, return_sequences=False)
        ])

        self.predictor = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(feature_dim)
        ])

    def call(self, image_sequence: tf.Tensor, training=None):
        """
        Predict future state

        Args:
            image_sequence: Sequence of images (batch, time, H, W, C)

        Returns:
            Predicted future features
        """
        batch_size, time_steps = tf.shape(image_sequence)[0], tf.shape(image_sequence)[1]

        # Encode each frame
        encoded = []
        for t in range(time_steps):
            frame_features = self.encoder(image_sequence[:, t], training=training)
            encoded.append(frame_features)

        encoded = tf.stack(encoded, axis=1)  # (batch, time, features)

        # LSTM processing
        lstm_out = self.lstm(encoded, training=training)

        # Predict future
        prediction = self.predictor(lstm_out, training=training)

        return prediction


class ChangeMetrics:
    """Compute change detection metrics"""

    @staticmethod
    def compute_change_area(
        change_map: np.ndarray,
        pixel_size_m: float = 10.0
    ) -> Dict[str, float]:
        """
        Compute area of change

        Args:
            change_map: Binary change map
            pixel_size_m: Pixel size in meters

        Returns:
            Change area statistics
        """
        # Count changed pixels
        changed_pixels = (change_map == 1).sum()

        # Compute area
        pixel_area_m2 = pixel_size_m ** 2
        total_area_m2 = changed_pixels * pixel_area_m2

        # Convert to hectares
        area_hectares = total_area_m2 / 10000

        return {
            'changed_pixels': int(changed_pixels),
            'area_m2': float(total_area_m2),
            'area_hectares': float(area_hectares),
            'area_km2': float(total_area_m2 / 1e6)
        }

    @staticmethod
    def classify_change_type(
        t1_classes: np.ndarray,
        t2_classes: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, int]:
        """
        Classify types of changes

        Args:
            t1_classes: Classifications at time 1
            t2_classes: Classifications at time 2
            class_names: Class names

        Returns:
            Change type counts
        """
        changes = {}

        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if i != j:
                    # Count transitions from class1 to class2
                    mask = (t1_classes == i) & (t2_classes == j)
                    count = mask.sum()

                    if count > 0:
                        changes[f"{class1}_to_{class2}"] = int(count)

        return changes


def create_change_detection_dataset(
    t1_images: List[str],
    t2_images: List[str],
    change_masks: List[str],
    batch_size: int = 16
) -> tf.data.Dataset:
    """
    Create bi-temporal change detection dataset

    Args:
        t1_images: Time 1 image paths
        t2_images: Time 2 image paths
        change_masks: Change mask paths
        batch_size: Batch size

    Returns:
        Dataset
    """

    def load_pair(t1_path, t2_path, mask_path):
        # Load images
        t1 = tf.io.read_file(t1_path)
        t1 = tf.image.decode_jpeg(t1, channels=3)
        t1 = tf.image.resize(t1, [224, 224])
        t1 = tf.cast(t1, tf.float32) / 255.0

        t2 = tf.io.read_file(t2_path)
        t2 = tf.image.decode_jpeg(t2, channels=3)
        t2 = tf.image.resize(t2, [224, 224])
        t2 = tf.cast(t2, tf.float32) / 255.0

        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [224, 224])
        mask = tf.cast(mask, tf.int32)

        return (t1, t2), mask

    dataset = tf.data.Dataset.from_tensor_slices((t1_images, t2_images, change_masks))
    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    print("Temporal Change Detection Ready!")
    print("\nCapabilities:")
    print("- Bi-temporal change detection (Siamese network)")
    print("- Time-series trend analysis")
    print("- Anomaly detection in sequences")
    print("- LSTM-based change prediction")
    print("- Change area computation")
    print("- Change type classification")
    print("\n🌍 Detect deforestation, urbanization, flooding, and more!")
