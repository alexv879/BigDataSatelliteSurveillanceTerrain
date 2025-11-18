"""
Multi-Modal Satellite Data Fusion
Combines Optical + SAR + Multispectral + Hyperspectral data
Powerful for all-weather, all-time terrain classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger


class MultiModalFusionNetwork(keras.Model):
    """
    Multi-modal fusion network for satellite data
    Combines different sensor modalities for robust classification
    """

    def __init__(
        self,
        num_classes: int = 10,
        optical_channels: int = 3,
        sar_channels: int = 2,
        multispectral_channels: int = 8,
        fusion_method: str = 'late'
    ):
        """
        Initialize multi-modal fusion network

        Args:
            num_classes: Number of terrain classes
            optical_channels: RGB/optical channels
            sar_channels: SAR polarizations (VV, VH)
            multispectral_channels: Multispectral bands
            fusion_method: 'early', 'late', or 'hybrid'
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # Individual encoders for each modality
        self.optical_encoder = self._build_encoder(optical_channels, name='optical')
        self.sar_encoder = self._build_encoder(sar_channels, name='sar')
        self.multispectral_encoder = self._build_encoder(multispectral_channels, name='multispectral')

        # Fusion layers
        if fusion_method == 'early':
            self.fusion = self._build_early_fusion()
        elif fusion_method == 'late':
            self.fusion = self._build_late_fusion()
        else:  # hybrid
            self.fusion = self._build_hybrid_fusion()

        # Classification head
        self.classifier = keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        logger.info(f"Multi-modal fusion network created ({fusion_method} fusion)")

    def _build_encoder(self, channels: int, name: str) -> keras.Sequential:
        """Build encoder for single modality"""
        return keras.Sequential([
            # Conv block 1
            layers.Conv2D(64, 3, padding='same', activation='relu', name=f'{name}_conv1'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),

            # Conv block 2
            layers.Conv2D(128, 3, padding='same', activation='relu', name=f'{name}_conv2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),

            # Conv block 3
            layers.Conv2D(256, 3, padding='same', activation='relu', name=f'{name}_conv3'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),

            # Conv block 4
            layers.Conv2D(512, 3, padding='same', activation='relu', name=f'{name}_conv4'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
        ], name=f'{name}_encoder')

    def _build_early_fusion(self) -> keras.Model:
        """Early fusion: Concatenate raw inputs"""
        return keras.Sequential([
            layers.Concatenate(axis=-1),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization()
        ])

    def _build_late_fusion(self) -> keras.Model:
        """Late fusion: Concatenate encoded features"""
        return keras.Sequential([
            layers.Concatenate(axis=-1),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu')
        ])

    def _build_hybrid_fusion(self) -> keras.Model:
        """Hybrid fusion: Attention-based fusion"""
        return AttentionFusion(512)

    def call(self, inputs: Dict[str, tf.Tensor], training=None):
        """
        Forward pass

        Args:
            inputs: Dictionary with keys 'optical', 'sar', 'multispectral'
            training: Training mode

        Returns:
            Class predictions
        """
        # Encode each modality
        optical_features = self.optical_encoder(inputs['optical'], training=training)
        sar_features = self.sar_encoder(inputs['sar'], training=training)
        multispectral_features = self.multispectral_encoder(inputs['multispectral'], training=training)

        # Fuse features
        if self.fusion_method == 'early':
            fused = self.fusion([optical_features, sar_features, multispectral_features])
        else:
            fused = self.fusion([optical_features, sar_features, multispectral_features])

        # Classify
        output = self.classifier(fused, training=training)

        return output


class AttentionFusion(layers.Layer):
    """Attention-based fusion for multi-modal data"""

    def __init__(self, hidden_dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

        # Attention weights for each modality
        self.attention_weights = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(3, activation='softmax')  # 3 modalities
        ])

        self.fusion_layer = layers.Dense(hidden_dim, activation='relu')

    def call(self, features_list: List[tf.Tensor]):
        """
        Fuse features with learned attention

        Args:
            features_list: List of feature tensors from different modalities

        Returns:
            Fused features
        """
        # Stack features
        stacked = tf.stack(features_list, axis=1)  # (B, 3, D)

        # Compute attention weights
        pooled = tf.reduce_mean(stacked, axis=1)  # (B, D)
        attention = self.attention_weights(pooled)  # (B, 3)
        attention = tf.expand_dims(attention, axis=-1)  # (B, 3, 1)

        # Apply attention
        weighted = stacked * attention  # (B, 3, D)
        fused = tf.reduce_sum(weighted, axis=1)  # (B, D)

        # Transform
        output = self.fusion_layer(fused)

        return output


class SARPreprocessor:
    """SAR-specific preprocessing"""

    @staticmethod
    def lee_filter(sar_image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Lee filter for SAR speckle reduction

        Args:
            sar_image: SAR image
            window_size: Filter window size

        Returns:
            Filtered image
        """
        import cv2

        # Convert to dB
        sar_db = 10 * np.log10(sar_image + 1e-10)

        # Apply bilateral filter (approximation of Lee filter)
        filtered = cv2.bilateralFilter(
            sar_db.astype(np.float32),
            window_size,
            75,
            75
        )

        # Convert back
        return 10 ** (filtered / 10)

    @staticmethod
    def compute_texture_features(sar_image: np.ndarray) -> np.ndarray:
        """
        Compute texture features from SAR

        Args:
            sar_image: SAR image

        Returns:
            Texture feature stack
        """
        from skimage.feature import graycomatrix, graycoprops

        # Normalize to 0-255
        normalized = ((sar_image - sar_image.min()) /
                     (sar_image.max() - sar_image.min()) * 255).astype(np.uint8)

        # GLCM
        glcm = graycomatrix(
            normalized,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        # Texture properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()

        return np.array([contrast, dissimilarity, homogeneity, energy, correlation])


class MultispectralProcessor:
    """Multispectral/Hyperspectral processing"""

    @staticmethod
    def compute_vegetation_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute vegetation indices

        Args:
            bands: Dictionary with band names (Red, NIR, Green, etc.)

        Returns:
            Dictionary of vegetation indices
        """
        indices = {}

        # NDVI (Normalized Difference Vegetation Index)
        if 'NIR' in bands and 'Red' in bands:
            indices['NDVI'] = (bands['NIR'] - bands['Red']) / (bands['NIR'] + bands['Red'] + 1e-10)

        # EVI (Enhanced Vegetation Index)
        if all(b in bands for b in ['NIR', 'Red', 'Blue']):
            indices['EVI'] = 2.5 * ((bands['NIR'] - bands['Red']) /
                                   (bands['NIR'] + 6 * bands['Red'] - 7.5 * bands['Blue'] + 1))

        # NDWI (Normalized Difference Water Index)
        if 'Green' in bands and 'NIR' in bands:
            indices['NDWI'] = (bands['Green'] - bands['NIR']) / (bands['Green'] + bands['NIR'] + 1e-10)

        # NDBI (Normalized Difference Built-up Index)
        if 'SWIR' in bands and 'NIR' in bands:
            indices['NDBI'] = (bands['SWIR'] - bands['NIR']) / (bands['SWIR'] + bands['NIR'] + 1e-10)

        return indices

    @staticmethod
    def atmospheric_correction(image: np.ndarray, method: str = 'dos') -> np.ndarray:
        """
        Simple atmospheric correction

        Args:
            image: Multispectral image
            method: Correction method

        Returns:
            Corrected image
        """
        if method == 'dos':  # Dark Object Subtraction
            # Find dark object (min value per band)
            dark_object = image.min(axis=(0, 1))

            # Subtract
            corrected = image - dark_object

            # Clip
            corrected = np.clip(corrected, 0, None)

            return corrected

        return image


class CloudMaskGenerator:
    """Generate cloud masks for optical imagery"""

    def __init__(self):
        self.model = self._build_cloud_model()

    def _build_cloud_model(self) -> keras.Model:
        """Build U-Net for cloud segmentation"""
        inputs = layers.Input(shape=(None, None, 3))

        # Encoder
        c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
        p1 = layers.MaxPooling2D(2)(c1)

        c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
        c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
        p2 = layers.MaxPooling2D(2)(c2)

        # Bottleneck
        c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
        c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)

        # Decoder
        u2 = layers.UpSampling2D(2)(c3)
        u2 = layers.Concatenate()([u2, c2])
        c4 = layers.Conv2D(128, 3, padding='same', activation='relu')(u2)

        u1 = layers.UpSampling2D(2)(c4)
        u1 = layers.Concatenate()([u1, c1])
        c5 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)

        # Output
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

        return keras.Model(inputs, outputs)

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate cloud mask"""
        mask = self.model.predict(image[np.newaxis, ...])[0]
        return (mask > 0.5).astype(np.uint8)


def create_multimodal_dataset(
    optical_paths: List[str],
    sar_paths: List[str],
    multispectral_paths: List[str],
    labels: np.ndarray,
    batch_size: int = 32
) -> tf.data.Dataset:
    """
    Create multi-modal dataset

    Args:
        optical_paths: Paths to optical images
        sar_paths: Paths to SAR images
        multispectral_paths: Paths to multispectral images
        labels: Labels
        batch_size: Batch size

    Returns:
        Multi-modal dataset
    """

    def load_multimodal_sample(optical_path, sar_path, ms_path, label):
        # Load optical
        optical = tf.io.read_file(optical_path)
        optical = tf.image.decode_jpeg(optical, channels=3)
        optical = tf.image.resize(optical, [224, 224])
        optical = tf.cast(optical, tf.float32) / 255.0

        # Load SAR
        sar = tf.io.read_file(sar_path)
        sar = tf.image.decode_png(sar, channels=2)
        sar = tf.image.resize(sar, [224, 224])
        sar = tf.cast(sar, tf.float32) / 255.0

        # Load multispectral
        ms = tf.io.read_file(ms_path)
        ms = tf.image.decode_png(ms, channels=8)
        ms = tf.image.resize(ms, [224, 224])
        ms = tf.cast(ms, tf.float32) / 255.0

        return {
            'optical': optical,
            'sar': sar,
            'multispectral': ms
        }, label

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        optical_paths,
        sar_paths,
        multispectral_paths,
        labels
    ))

    dataset = dataset.map(load_multimodal_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    print("Multi-Modal Satellite Fusion Ready!")
    print("\nCapabilities:")
    print("- Optical + SAR + Multispectral fusion")
    print("- SAR speckle filtering (Lee filter)")
    print("- Texture feature extraction")
    print("- Vegetation indices (NDVI, EVI, NDWI, NDBI)")
    print("- Atmospheric correction")
    print("- Cloud masking")
    print("- Attention-based fusion")
    print("\n🚀 All-weather, all-time terrain classification!")
