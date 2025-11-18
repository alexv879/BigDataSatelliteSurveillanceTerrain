"""
Semantic Segmentation for Satellite Imagery
Pixel-level terrain classification with state-of-the-art architectures
U-Net, DeepLabV3+, PSPNet, and more
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Dict, Optional
from loguru import logger


class UNet(keras.Model):
    """
    U-Net architecture for semantic segmentation
    Excellent for satellite imagery segmentation
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        filters: List[int] = [64, 128, 256, 512, 1024]
    ):
        """
        Initialize U-Net

        Args:
            num_classes: Number of terrain classes
            input_shape: Input shape
            filters: Number of filters per level
        """
        super().__init__()
        self.num_classes = num_classes
        self.filters = filters

        # Build model
        inputs = layers.Input(shape=input_shape)
        x = self._build_unet(inputs, filters, num_classes)
        self.model = keras.Model(inputs, x)

        logger.info(f"U-Net created: {input_shape} -> {num_classes} classes")

    def _build_unet(self, inputs, filters, num_classes):
        """Build U-Net architecture"""
        # Encoder
        encoder_outputs = []

        x = inputs
        for i, f in enumerate(filters[:-1]):
            x = self._conv_block(x, f, name=f'encoder_{i}')
            encoder_outputs.append(x)
            x = layers.MaxPooling2D(2, name=f'pool_{i}')(x)

        # Bottleneck
        x = self._conv_block(x, filters[-1], name='bottleneck')

        # Decoder
        for i, f in enumerate(reversed(filters[:-1])):
            x = layers.UpSampling2D(2, name=f'upsample_{i}')(x)
            x = layers.Concatenate(name=f'concat_{i}')([x, encoder_outputs[-(i+1)]])
            x = self._conv_block(x, f, name=f'decoder_{i}')

        # Output
        outputs = layers.Conv2D(
            num_classes,
            1,
            activation='softmax',
            name='output'
        )(x)

        return outputs

    def _conv_block(self, x, filters, name):
        """Convolutional block"""
        x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu', name=f'{name}_relu1')(x)

        x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = layers.Activation('relu', name=f'{name}_relu2')(x)

        return x

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


class DeepLabV3Plus(keras.Model):
    """
    DeepLabV3+ with Atrous Spatial Pyramid Pooling
    State-of-the-art for semantic segmentation
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        backbone: str = 'resnet50',
        output_stride: int = 16
    ):
        """
        Initialize DeepLabV3+

        Args:
            num_classes: Number of classes
            input_shape: Input shape
            backbone: Backbone network
            output_stride: Output stride (8 or 16)
        """
        super().__init__()
        self.num_classes = num_classes

        inputs = layers.Input(shape=input_shape)
        x = self._build_deeplabv3plus(inputs, num_classes, backbone, output_stride)
        self.model = keras.Model(inputs, x)

        logger.info(f"DeepLabV3+ created with {backbone} backbone")

    def _build_deeplabv3plus(self, inputs, num_classes, backbone, output_stride):
        """Build DeepLabV3+ architecture"""
        # Encoder (backbone)
        if backbone == 'resnet50':
            encoder = keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs
            )
            # Get intermediate features
            low_level_features = encoder.get_layer('conv2_block3_out').output
            high_level_features = encoder.get_layer('conv5_block3_out').output
        else:
            # Simple encoder
            x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            low_level_features = x

            x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
            for filters in [64, 128, 256]:
                x = self._residual_block(x, filters)
            high_level_features = x

        # ASPP (Atrous Spatial Pyramid Pooling)
        aspp_features = self._aspp_block(high_level_features)

        # Decoder
        # Upsample ASPP features
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(aspp_features)

        # Process low-level features
        low_level = layers.Conv2D(48, 1, padding='same')(low_level_features)
        low_level = layers.BatchNormalization()(low_level)
        low_level = layers.Activation('relu')(low_level)

        # Concatenate
        x = layers.Concatenate()([x, low_level])

        # Refine
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Final upsampling
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        # Output
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)

        return outputs

    def _aspp_block(self, x):
        """Atrous Spatial Pyramid Pooling"""
        # Different dilation rates
        rates = [6, 12, 18]

        # 1x1 convolution
        conv1 = layers.Conv2D(256, 1, padding='same')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation('relu')(conv1)

        # Atrous convolutions
        atrous_convs = []
        for rate in rates:
            conv = layers.Conv2D(256, 3, padding='same', dilation_rate=rate)(x)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Activation('relu')(conv)
            atrous_convs.append(conv)

        # Global average pooling
        gap = layers.GlobalAveragePooling2D(keepdims=True)(x)
        gap = layers.Conv2D(256, 1, padding='same')(gap)
        gap = layers.BatchNormalization()(gap)
        gap = layers.Activation('relu')(gap)
        gap = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(gap)

        # Concatenate all
        concat = layers.Concatenate()([conv1] + atrous_convs + [gap])

        # Final 1x1 conv
        output = layers.Conv2D(256, 1, padding='same')(concat)
        output = layers.BatchNormalization()(output)
        output = layers.Activation('relu')(output)

        return output

    def _residual_block(self, x, filters):
        """Residual block"""
        shortcut = x

        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Match dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)

        return x

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


class PSPNet(keras.Model):
    """
    Pyramid Scene Parsing Network
    Captures context at multiple scales
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        pyramid_bins: List[int] = [1, 2, 3, 6]
    ):
        """
        Initialize PSPNet

        Args:
            num_classes: Number of classes
            input_shape: Input shape
            pyramid_bins: Pyramid pooling bin sizes
        """
        super().__init__()
        self.num_classes = num_classes
        self.pyramid_bins = pyramid_bins

        inputs = layers.Input(shape=input_shape)
        x = self._build_pspnet(inputs, num_classes, pyramid_bins)
        self.model = keras.Model(inputs, x)

        logger.info(f"PSPNet created with pyramid bins: {pyramid_bins}")

    def _build_pspnet(self, inputs, num_classes, pyramid_bins):
        """Build PSPNet architecture"""
        # Backbone
        backbone = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )

        features = backbone.get_layer('conv5_block3_out').output

        # Pyramid pooling module
        pyramid_features = self._pyramid_pooling_module(features, pyramid_bins)

        # Concatenate
        x = layers.Concatenate()([features, pyramid_features])

        # Final convolutions
        x = layers.Conv2D(512, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

        # Upsample to original size
        x = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)

        # Output
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)

        return outputs

    def _pyramid_pooling_module(self, x, bins):
        """Pyramid pooling module"""
        h, w = x.shape[1], x.shape[2]
        pyramid_outputs = []

        for bin_size in bins:
            # Adaptive pooling
            pooled = layers.AveragePooling2D(
                pool_size=(h // bin_size, w // bin_size),
                strides=(h // bin_size, w // bin_size)
            )(x)

            # 1x1 conv
            pooled = layers.Conv2D(512 // len(bins), 1, padding='same')(pooled)
            pooled = layers.BatchNormalization()(pooled)
            pooled = layers.Activation('relu')(pooled)

            # Upsample back
            pooled = layers.UpSampling2D(size=(bin_size, bin_size), interpolation='bilinear')(pooled)

            pyramid_outputs.append(pooled)

        # Concatenate all pyramid levels
        return layers.Concatenate()(pyramid_outputs)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


class SegmentationMetrics:
    """Metrics for semantic segmentation"""

    @staticmethod
    def compute_iou(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int
    ) -> Dict[str, float]:
        """
        Compute Intersection over Union (IoU)

        Args:
            y_true: Ground truth (H, W)
            y_pred: Predictions (H, W)
            num_classes: Number of classes

        Returns:
            IoU metrics
        """
        iou_per_class = []

        for cls in range(num_classes):
            true_mask = (y_true == cls)
            pred_mask = (y_pred == cls)

            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()

            if union == 0:
                iou = 1.0  # Perfect if class not present
            else:
                iou = intersection / union

            iou_per_class.append(iou)

        return {
            'mean_iou': float(np.mean(iou_per_class)),
            'per_class_iou': [float(iou) for iou in iou_per_class]
        }

    @staticmethod
    def compute_dice(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int
    ) -> float:
        """
        Compute Dice coefficient

        Args:
            y_true: Ground truth (H, W)
            y_pred: Predictions (H, W)
            num_classes: Number of classes

        Returns:
            Mean Dice coefficient
        """
        dice_per_class = []

        for cls in range(num_classes):
            true_mask = (y_true == cls)
            pred_mask = (y_pred == cls)

            intersection = np.logical_and(true_mask, pred_mask).sum()
            total = true_mask.sum() + pred_mask.sum()

            if total == 0:
                dice = 1.0
            else:
                dice = 2 * intersection / total

            dice_per_class.append(dice)

        return float(np.mean(dice_per_class))

    @staticmethod
    def compute_pixel_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute pixel accuracy

        Args:
            y_true: Ground truth (H, W)
            y_pred: Predictions (H, W)

        Returns:
            Pixel accuracy
        """
        correct = (y_true == y_pred).sum()
        total = y_true.size

        return float(correct / total)


def create_segmentation_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    img_size: Tuple[int, int] = (512, 512),
    num_classes: int = 10,
    batch_size: int = 8,
    augment: bool = True
) -> tf.data.Dataset:
    """
    Create segmentation dataset

    Args:
        image_paths: Paths to images
        mask_paths: Paths to segmentation masks
        img_size: Image size
        num_classes: Number of classes
        batch_size: Batch size
        augment: Apply augmentation

    Returns:
        tf.data.Dataset
    """

    def load_sample(img_path, mask_path):
        # Load image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0

        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, img_size, method='nearest')
        mask = tf.cast(mask, tf.int32)
        mask = tf.squeeze(mask, axis=-1)

        # One-hot encode
        mask = tf.one_hot(mask, num_classes)

        return img, mask

    def augment_sample(img, mask):
        """Apply augmentation"""
        # Random flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        # Random rotation (90, 180, 270 degrees)
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        mask = tf.image.rot90(mask, k=k)

        # Random brightness/contrast
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)

        img = tf.clip_by_value(img, 0.0, 1.0)

        return img, mask

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


class PostProcessor:
    """Post-processing for segmentation results"""

    @staticmethod
    def conditional_random_field(
        image: np.ndarray,
        predictions: np.ndarray,
        num_iterations: int = 5
    ) -> np.ndarray:
        """
        Apply CRF for smoothing (simplified version)

        Args:
            image: Original image
            predictions: Class probabilities (H, W, C)
            num_iterations: CRF iterations

        Returns:
            Refined predictions
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            logger.warning("pydensecrf not installed, skipping CRF")
            return np.argmax(predictions, axis=-1)

        h, w, num_classes = predictions.shape

        # Setup CRF
        d = dcrf.DenseCRF2D(w, h, num_classes)

        # Unary potentials
        U = unary_from_softmax(predictions.transpose(2, 0, 1))
        d.setUnaryEnergy(U)

        # Pairwise potentials
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

        # Inference
        Q = d.inference(num_iterations)
        result = np.argmax(Q, axis=0).reshape(h, w)

        return result

    @staticmethod
    def remove_small_objects(
        segmentation: np.ndarray,
        min_size: int = 100
    ) -> np.ndarray:
        """
        Remove small segmented regions

        Args:
            segmentation: Segmentation map
            min_size: Minimum region size in pixels

        Returns:
            Cleaned segmentation
        """
        from scipy import ndimage

        cleaned = segmentation.copy()
        unique_classes = np.unique(segmentation)

        for cls in unique_classes:
            mask = (segmentation == cls)

            # Label connected components
            labeled, num_features = ndimage.label(mask)

            # Remove small components
            for region_id in range(1, num_features + 1):
                region_mask = (labeled == region_id)
                if region_mask.sum() < min_size:
                    cleaned[region_mask] = 0  # Background

        return cleaned


if __name__ == "__main__":
    print("Semantic Segmentation Ready!")
    print("\nArchitectures:")
    print("- U-Net: Fast and accurate")
    print("- DeepLabV3+: State-of-the-art with ASPP")
    print("- PSPNet: Pyramid pooling for multi-scale context")
    print("\nMetrics:")
    print("- IoU (Intersection over Union)")
    print("- Dice coefficient")
    print("- Pixel accuracy")
    print("\nPost-processing:")
    print("- CRF smoothing")
    print("- Small object removal")
    print("\n🎯 Pixel-perfect terrain classification!")
