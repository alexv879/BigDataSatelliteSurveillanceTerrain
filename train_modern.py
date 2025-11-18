"""
Modern Training Pipeline for Satellite Terrain Classification
State-of-the-art training with all cutting-edge features
"""
import tensorflow as tf
from tensorflow import keras
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
from typing import Tuple, Optional
from loguru import logger
import sys
import os

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from models.vision_transformer import create_vit_with_mixed_precision
from models.efficientnet_modern import create_efficientnet_with_mixed_precision, get_training_callbacks
from augmentation.advanced_augmentation import (
    create_augmented_dataset,
    get_albumentations_transform,
    MixUp,
    CutMix
)
from mlops.experiment_tracking import ExperimentTracker, ModelVersioning
from interpretability.gradcam import GradCAM, UncertaintyEstimator
from optimization.model_optimizer import optimize_model_pipeline


def setup_gpu(config: DictConfig):
    """Configure GPU settings"""
    if config.hardware.gpu.enabled:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(
                        gpu, config.hardware.gpu.memory_growth
                    )
                logger.info(f"GPUs configured: {len(gpus)}")
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        else:
            logger.warning("No GPUs found, using CPU")
    else:
        logger.info("GPU disabled, using CPU")


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


def create_data_pipeline(
    data_dir: str,
    config: DictConfig,
    training: bool = True
) -> tf.data.Dataset:
    """
    Create modern data pipeline with advanced augmentation

    Args:
        data_dir: Data directory
        config: Configuration
        training: Whether this is training data

    Returns:
        TensorFlow dataset
    """
    # Create image data generator
    if training:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=config.model.input_shape[:2],
            batch_size=config.training.batch_size,
            class_mode='categorical',
            subset='training'
        )
    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=config.model.input_shape[:2],
            batch_size=config.training.batch_size,
            class_mode='categorical'
        )

    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *config.model.input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, config.model.num_classes), dtype=tf.float32)
        )
    )

    # Apply advanced augmentation
    if training and config.data.augmentation.use_advanced:
        dataset = create_augmented_dataset(
            dataset,
            use_mixup=config.data.augmentation.use_mixup,
            use_cutmix=config.data.augmentation.use_cutmix,
            mixup_alpha=config.data.augmentation.mixup_alpha,
            cutmix_alpha=config.data.augmentation.cutmix_alpha
        )

    # Performance optimization
    if config.data.cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_model(config: DictConfig) -> keras.Model:
    """
    Create model based on configuration

    Args:
        config: Model configuration

    Returns:
        Keras model
    """
    logger.info(f"Creating {config.model.architecture} model...")

    if config.model.architecture == "vit":
        model = create_vit_with_mixed_precision(
            input_shape=tuple(config.model.input_shape),
            num_classes=config.model.num_classes,
            patch_size=config.model.vit.patch_size,
            projection_dim=config.model.vit.projection_dim,
            num_heads=config.model.vit.num_heads,
            transformer_layers=config.model.vit.transformer_layers,
            learning_rate=config.training.initial_learning_rate
        )

    elif config.model.architecture == "efficientnetv2":
        model = create_efficientnet_with_mixed_precision(
            input_shape=tuple(config.model.input_shape),
            num_classes=config.model.num_classes,
            model_variant=config.model.variant,
            pretrained=config.model.pretrained,
            fine_tune_layers=config.model.fine_tune_layers,
            dropout_rate=config.model.dropout_rate,
            use_attention=config.model.use_attention,
            learning_rate=config.training.initial_learning_rate
        )

    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")

    logger.info(f"Model created with {model.count_params():,} parameters")
    return model


def get_callbacks(config: DictConfig, tracker: ExperimentTracker) -> list:
    """
    Get training callbacks

    Args:
        config: Configuration
        tracker: Experiment tracker

    Returns:
        List of callbacks
    """
    callbacks = []

    # Experiment tracking callbacks
    callbacks.extend(tracker.get_callbacks())

    # Model checkpoint
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath='models/checkpoints/best_model.h5',
        monitor=config.mlops.checkpointing.monitor,
        mode=config.mlops.checkpointing.mode,
        save_best_only=config.mlops.checkpointing.save_best_only,
        verbose=1
    ))

    # Early stopping
    if config.training.early_stopping.enabled:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=config.training.early_stopping.monitor,
            patience=config.training.early_stopping.patience,
            mode=config.training.early_stopping.mode,
            restore_best_weights=True,
            verbose=1
        ))

    # Learning rate scheduling
    if config.training.lr_schedule.type == "cosine_annealing":
        callbacks.append(keras.callbacks.CosineDecayRestarts(
            initial_learning_rate=config.training.initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=config.training.lr_schedule.min_lr
        ))

    # TensorBoard
    callbacks.append(keras.callbacks.TensorBoard(
        log_dir='logs/tensorboard',
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    ))

    return callbacks


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training function

    Args:
        cfg: Hydra configuration
    """
    logger.info("=== Starting Modern Training Pipeline ===")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup
    set_seed(cfg.seed)
    setup_gpu(cfg)

    # Initialize experiment tracking
    tracker = ExperimentTracker(
        project_name=cfg.mlops.experiment_tracking.project_name,
        experiment_name=f"{cfg.model.architecture}_{cfg.model.variant}",
        use_mlflow=cfg.mlops.experiment_tracking.use_mlflow,
        use_wandb=cfg.mlops.experiment_tracking.use_wandb,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Log hyperparameters
    tracker.log_params({
        "architecture": cfg.model.architecture,
        "variant": cfg.model.variant,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.initial_learning_rate,
        "epochs": cfg.training.epochs
    })

    # Create data pipelines
    logger.info("Creating data pipelines...")
    train_dataset = create_data_pipeline(cfg.data.train_dir, cfg, training=True)
    val_dataset = create_data_pipeline(cfg.data.train_dir, cfg, training=False)

    # Create model
    model = create_model(cfg)

    # Get callbacks
    callbacks = get_callbacks(cfg, tracker)

    # Training
    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=cfg.training.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Log final metrics
    final_metrics = {
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1])
    }
    tracker.log_metrics(final_metrics)

    # Save model
    logger.info("Saving model...")
    model_path = "models/final_model.h5"
    model.save(model_path)
    tracker.log_artifact(model_path)

    # Model versioning
    if cfg.mlops.model_versioning.enabled:
        logger.info("Registering model...")
        versioning = ModelVersioning(cfg.mlops.model_versioning.registry_path)
        versioning.register_model(
            model=model,
            model_name=f"{cfg.model.architecture}_{cfg.model.variant}",
            version=cfg.deployment.model_version,
            metrics=final_metrics,
            config=OmegaConf.to_container(cfg.model, resolve=True),
            description=f"Trained on {cfg.training.epochs} epochs"
        )

    # Model optimization
    if cfg.optimization.quantization.enabled or cfg.optimization.pruning.enabled:
        logger.info("Optimizing model...")
        methods = []
        if cfg.optimization.quantization.enabled:
            methods.append(cfg.optimization.quantization.method)
        if cfg.optimization.pruning.enabled:
            methods.append("pruning")

        optimized_models = optimize_model_pipeline(
            model,
            representative_dataset=train_dataset.take(100),
            methods=methods
        )
        logger.info(f"Optimized models created: {list(optimized_models.keys())}")

    # Model interpretability
    if cfg.interpretability.gradcam.enabled:
        logger.info("Creating interpretability visualizations...")
        gradcam = GradCAM(model)
        # Save sample visualizations
        # (Add code to visualize on test samples)

    # Finish tracking
    tracker.finish()

    logger.info("=== Training Completed Successfully ===")
    logger.info(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    train()
