"""
Advanced Data Augmentation Pipeline for Satellite Terrain Classification
Implements state-of-the-art augmentation techniques: RandAugment, MixUp, CutMix
"""
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import albumentations as A


class MixUp:
    """
    MixUp: Beyond Empirical Risk Minimization
    Mixes two images and their labels
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, images, labels):
        batch_size = tf.shape(images)[0]

        # Sample lambda from Beta distribution
        lambda_value = tf.random.uniform([], 0, 1)
        if self.alpha > 0:
            lambda_value = tf.numpy_function(
                lambda: np.random.beta(self.alpha, self.alpha),
                [],
                tf.float32
            )

        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))

        # Mix images
        mixed_images = lambda_value * images + (1 - lambda_value) * tf.gather(images, indices)

        # Mix labels
        mixed_labels = lambda_value * labels + (1 - lambda_value) * tf.gather(labels, indices)

        return mixed_images, mixed_labels


class CutMix:
    """
    CutMix: Regularization Strategy to Train Strong Classifiers
    Cuts and pastes patches between images
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _get_box(self, lambda_value, image_shape):
        """Generate random bounding box"""
        image_h, image_w = image_shape[0], image_shape[1]

        cut_ratio = tf.math.sqrt(1.0 - lambda_value)
        cut_h = tf.cast(image_h * cut_ratio, tf.int32)
        cut_w = tf.cast(image_w * cut_ratio, tf.int32)

        # Random center
        cx = tf.random.uniform([], 0, image_w, dtype=tf.int32)
        cy = tf.random.uniform([], 0, image_h, dtype=tf.int32)

        # Get box coordinates
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_w)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_h)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_w)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_h)

        return x1, y1, x2, y2

    def __call__(self, images, labels):
        batch_size = tf.shape(images)[0]

        # Sample lambda
        lambda_value = tf.numpy_function(
            lambda: np.random.beta(self.alpha, self.alpha),
            [],
            tf.float32
        )

        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)

        # Get bounding box
        image_shape = tf.shape(images)[1:3]
        x1, y1, x2, y2 = self._get_box(lambda_value, image_shape)

        # Create mask
        mask = tf.ones_like(images)
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.where(
                (tf.range(tf.shape(mask)[1])[:, None] >= y1) &
                (tf.range(tf.shape(mask)[1])[:, None] < y2) &
                (tf.range(tf.shape(mask)[2])[None, :] >= x1) &
                (tf.range(tf.shape(mask)[2])[None, :] < x2)
            ),
            tf.zeros([tf.reduce_sum(tf.cast(
                (tf.range(tf.shape(mask)[1])[:, None] >= y1) &
                (tf.range(tf.shape(mask)[1])[:, None] < y2) &
                (tf.range(tf.shape(mask)[2])[None, :] >= x1) &
                (tf.range(tf.shape(mask)[2])[None, :] < x2),
                tf.int32
            ))])
        )

        # Apply CutMix
        mixed_images = images * (1 - mask) + shuffled_images * mask

        # Adjust lambda based on actual box size
        box_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
        total_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        lambda_adjusted = 1 - box_area / total_area

        # Mix labels
        mixed_labels = lambda_adjusted * labels + (1 - lambda_adjusted) * shuffled_labels

        return mixed_images, mixed_labels


def get_albumentations_transform(image_size: int = 224, augment: bool = True):
    """
    Get Albumentations transform pipeline for advanced augmentation

    Args:
        image_size: Target image size
        augment: Whether to apply augmentation

    Returns:
        Albumentations transform
    """
    if augment:
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.5),
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 8,
                max_width=image_size // 8,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class RandAugment:
    """
    RandAugment: Practical automated data augmentation
    Randomly applies N augmentations with magnitude M
    """

    def __init__(self, n: int = 2, m: int = 10):
        """
        Args:
            n: Number of augmentation operations to apply
            m: Magnitude of augmentation (0-10)
        """
        self.n = n
        self.m = m
        self.augmentations = [
            self._auto_contrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, image):
        """Apply N random augmentations with magnitude M"""
        for _ in range(self.n):
            op = np.random.choice(self.augmentations)
            image = op(image)
        return image

    def _auto_contrast(self, image):
        return tf.image.adjust_contrast(image, 1.5)

    def _equalize(self, image):
        return tf.image.per_image_standardization(image)

    def _rotate(self, image):
        angle = (self.m / 10) * 30  # Max 30 degrees
        return tf.keras.preprocessing.image.random_rotation(
            image, angle, row_axis=0, col_axis=1, channel_axis=2
        )

    def _solarize(self, image):
        threshold = 1.0 - (self.m / 10) * 0.5
        return tf.where(image < threshold, image, 1.0 - image)

    def _color(self, image):
        factor = 1.0 + (self.m / 10) * 0.9
        return tf.image.adjust_saturation(image, factor)

    def _posterize(self, image):
        shift = int((self.m / 10) * 4)
        return tf.bitwise.left_shift(tf.bitwise.right_shift(
            tf.cast(image * 255, tf.int32), shift
        ), shift) / 255.0

    def _contrast(self, image):
        factor = 1.0 + (self.m / 10) * 0.9
        return tf.image.adjust_contrast(image, factor)

    def _brightness(self, image):
        delta = (self.m / 10) * 0.9
        return tf.image.adjust_brightness(image, delta)

    def _sharpness(self, image):
        kernel = tf.constant([
            [-1, -1, -1],
            [-1, 8 + self.m, -1],
            [-1, -1, -1]
        ], dtype=tf.float32) / (self.m + 8)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        return tf.nn.conv2d(image[None, ...], kernel, strides=1, padding='SAME')[0]

    def _shear_x(self, image):
        level = (self.m / 10) * 0.3
        return tf.keras.preprocessing.image.apply_affine_transform(
            image, shear=level
        )

    def _shear_y(self, image):
        level = (self.m / 10) * 0.3
        return tf.keras.preprocessing.image.apply_affine_transform(
            image, shear=level
        )

    def _translate_x(self, image):
        pixels = int((self.m / 10) * image.shape[1] * 0.3)
        return tf.keras.preprocessing.image.apply_affine_transform(
            image, tx=pixels
        )

    def _translate_y(self, image):
        pixels = int((self.m / 10) * image.shape[0] * 0.3)
        return tf.keras.preprocessing.image.apply_affine_transform(
            image, ty=pixels
        )


def create_augmented_dataset(
    dataset: tf.data.Dataset,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.5
) -> tf.data.Dataset:
    """
    Create dataset with advanced augmentation techniques

    Args:
        dataset: Input TensorFlow dataset
        use_mixup: Apply MixUp augmentation
        use_cutmix: Apply CutMix augmentation
        mixup_alpha: MixUp alpha parameter
        cutmix_alpha: CutMix alpha parameter
        mixup_prob: Probability of applying MixUp vs CutMix

    Returns:
        Augmented dataset
    """
    mixup = MixUp(alpha=mixup_alpha)
    cutmix = CutMix(alpha=cutmix_alpha)

    def augment(images, labels):
        # Randomly choose between MixUp and CutMix
        if use_mixup and use_cutmix:
            if tf.random.uniform([]) < mixup_prob:
                return mixup(images, labels)
            else:
                return cutmix(images, labels)
        elif use_mixup:
            return mixup(images, labels)
        elif use_cutmix:
            return cutmix(images, labels)
        return images, labels

    return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == "__main__":
    print("Advanced Augmentation Pipeline Ready")
    print("Features:")
    print("- MixUp: Mixes images and labels")
    print("- CutMix: Cuts and pastes image regions")
    print("- RandAugment: Automated augmentation search")
    print("- Albumentations: 20+ advanced transformations")
