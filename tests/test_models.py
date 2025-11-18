"""
Comprehensive Test Suite for Satellite Terrain Classification
Tests for models, training, inference, and utilities
"""
import pytest
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.vision_transformer import create_vit_model, PatchExtractor, PatchEncoder
from models.efficientnet_modern import create_efficientnet_model
from augmentation.advanced_augmentation import MixUp, CutMix, RandAugment


class TestVisionTransformer:
    """Test Vision Transformer implementation"""

    def test_patch_extractor(self):
        """Test patch extraction"""
        patch_size = 16
        extractor = PatchExtractor(patch_size)

        # Create test image
        image = tf.random.normal((1, 224, 224, 3))

        # Extract patches
        patches = extractor(image)

        # Check output shape
        num_patches = (224 // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        assert patches.shape == (1, num_patches, patch_dim)

    def test_patch_encoder(self):
        """Test patch encoding"""
        num_patches = 196
        projection_dim = 768

        encoder = PatchEncoder(num_patches, projection_dim)

        # Create test patches
        patches = tf.random.normal((1, num_patches, 768))

        # Encode
        encoded = encoder(patches)

        # Check output shape
        assert encoded.shape == (1, num_patches, projection_dim)

    def test_vit_model_creation(self):
        """Test ViT model creation"""
        model = create_vit_model(
            input_shape=(224, 224, 3),
            num_classes=10,
            patch_size=16,
            projection_dim=256,
            num_heads=4,
            transformer_layers=4
        )

        # Check model can be compiled
        assert model is not None
        assert isinstance(model, keras.Model)

        # Check output shape
        x = tf.random.normal((2, 224, 224, 3))
        y = model(x, training=False)

        assert y.shape == (2, 10)

    def test_vit_forward_pass(self):
        """Test ViT forward pass"""
        model = create_vit_model(
            input_shape=(224, 224, 3),
            num_classes=5,
            patch_size=16,
            projection_dim=128,
            num_heads=4,
            transformer_layers=2
        )

        # Forward pass
        x = tf.random.normal((4, 224, 224, 3))
        y = model(x, training=True)

        # Check output
        assert y.shape == (4, 5)
        assert tf.reduce_sum(y[0]).numpy() == pytest.approx(1.0, abs=1e-5)


class TestEfficientNet:
    """Test EfficientNet implementation"""

    def test_efficientnet_creation(self):
        """Test EfficientNet model creation"""
        model = create_efficientnet_model(
            input_shape=(224, 224, 3),
            num_classes=10,
            model_variant="B0",
            pretrained=False
        )

        assert model is not None
        assert isinstance(model, keras.Model)

    def test_efficientnet_forward_pass(self):
        """Test EfficientNet forward pass"""
        model = create_efficientnet_model(
            input_shape=(224, 224, 3),
            num_classes=5,
            model_variant="B0",
            pretrained=False
        )

        x = tf.random.normal((2, 224, 224, 3))
        y = model(x, training=False)

        assert y.shape == (2, 5)

    def test_efficientnet_variants(self):
        """Test different EfficientNet variants"""
        for variant in ['B0', 'B1', 'B2']:
            model = create_efficientnet_model(
                input_shape=(224, 224, 3),
                num_classes=10,
                model_variant=variant,
                pretrained=False
            )

            assert model is not None


class TestAugmentation:
    """Test augmentation techniques"""

    def test_mixup(self):
        """Test MixUp augmentation"""
        mixup = MixUp(alpha=0.2)

        images = tf.random.normal((8, 224, 224, 3))
        labels = tf.one_hot(tf.random.uniform((8,), 0, 10, dtype=tf.int32), 10)

        mixed_images, mixed_labels = mixup(images, labels)

        assert mixed_images.shape == images.shape
        assert mixed_labels.shape == labels.shape

    def test_cutmix(self):
        """Test CutMix augmentation"""
        cutmix = CutMix(alpha=1.0)

        images = tf.random.normal((8, 224, 224, 3))
        labels = tf.one_hot(tf.random.uniform((8,), 0, 10, dtype=tf.int32), 10)

        mixed_images, mixed_labels = cutmix(images, labels)

        assert mixed_images.shape == images.shape
        assert mixed_labels.shape == labels.shape

    def test_randaugment(self):
        """Test RandAugment"""
        randaug = RandAugment(n=2, m=10)

        image = tf.random.normal((224, 224, 3))
        augmented = randaug(image)

        assert augmented.shape == image.shape


class TestTraining:
    """Test training pipeline"""

    def test_simple_training(self):
        """Test simple training loop"""
        # Create simple model
        model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy data
        x_train = np.random.rand(32, 64, 64, 3).astype(np.float32)
        y_train = tf.one_hot(np.random.randint(0, 10, 32), 10).numpy()

        # Train for 1 epoch
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=8,
            verbose=0
        )

        assert 'loss' in history.history
        assert 'accuracy' in history.history

    def test_mixed_precision(self):
        """Test mixed precision training"""
        # Set mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        # Create model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(100,)),
            keras.layers.Dense(10, activation='softmax', dtype='float32')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Check policy
        assert model.layers[0].dtype_policy.name == 'mixed_float16'

        # Reset policy
        tf.keras.mixed_precision.set_global_policy('float32')


class TestInference:
    """Test inference pipeline"""

    def test_batch_prediction(self):
        """Test batch prediction"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(100,)),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Predict
        x = np.random.rand(16, 100).astype(np.float32)
        predictions = model.predict(x, verbose=0)

        assert predictions.shape == (16, 10)
        assert np.allclose(predictions.sum(axis=1), 1.0)

    def test_single_prediction(self):
        """Test single image prediction"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Single prediction
        x = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(x, verbose=0)

        assert prediction.shape == (1, 10)


class TestEnsemble:
    """Test ensemble methods"""

    def test_model_averaging(self):
        """Test simple model averaging"""
        # Create multiple models
        models = []
        for _ in range(3):
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(100,)),
                keras.layers.Dense(10, activation='softmax')
            ])
            models.append(model)

        # Predict with each
        x = np.random.rand(8, 100).astype(np.float32)
        predictions = []

        for model in models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)

        # Average
        avg_pred = np.mean(predictions, axis=0)

        assert avg_pred.shape == (8, 10)


class TestDataPipeline:
    """Test data pipeline"""

    def test_tf_dataset_creation(self):
        """Test TensorFlow dataset creation"""
        # Create dataset
        x = np.random.rand(100, 64, 64, 3).astype(np.float32)
        y = tf.one_hot(np.random.randint(0, 10, 100), 10).numpy()

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(16)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Check
        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape[0] <= 16
            assert batch_y.shape[0] <= 16

    def test_data_augmentation_pipeline(self):
        """Test data augmentation in pipeline"""
        x = np.random.rand(100, 64, 64, 3).astype(np.float32)
        y = tf.one_hot(np.random.randint(0, 10, 100), 10).numpy()

        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        # Add augmentation
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            return image, label

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(16)

        # Check
        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape == (16, 64, 64, 3)


def run_all_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
