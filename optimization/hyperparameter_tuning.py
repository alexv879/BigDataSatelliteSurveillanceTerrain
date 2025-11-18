"""
Hyperparameter Optimization with Optuna
Automated hyperparameter search for optimal model performance
"""
import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.vision_transformer import create_vit_model
from models.efficientnet_modern import create_efficientnet_model


class HyperparameterTuner:
    """Automated hyperparameter tuning with Optuna"""

    def __init__(
        self,
        model_builder: Callable,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        study_name: str = "terrain_classification",
        storage: Optional[str] = None
    ):
        """
        Initialize hyperparameter tuner

        Args:
            model_builder: Function to build model given hyperparameters
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            study_name: Name of Optuna study
            storage: Optional database URL for study persistence
        """
        self.model_builder = model_builder
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.study_name = study_name
        self.storage = storage or f"sqlite:///{study_name}.db"

        logger.info(f"Hyperparameter tuner initialized: {study_name}")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function

        Args:
            trial: Optuna trial

        Returns:
            Validation accuracy to maximize
        """
        # Sample hyperparameters
        params = self.suggest_hyperparameters(trial)

        # Build model
        model = self.model_builder(**params)

        # Training configuration
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = 50

        # Callbacks
        callbacks = [
            TFKerasPruningCallback(trial, 'val_accuracy'),
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]

        # Train
        history = model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )

        # Return best validation accuracy
        return max(history.history['val_accuracy'])

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for trial

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of hyperparameters
        """
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3),
        }

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> optuna.Study:
        """
        Run hyperparameter optimization

        Args:
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            n_jobs: Number of parallel jobs

        Returns:
            Optuna study
        """
        logger.info(f"Starting optimization with {n_trials} trials...")

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        # Log results
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Save study
        self.save_study(study)

        return study

    def save_study(self, study: optuna.Study):
        """Save study results"""
        save_dir = Path(f"optimization/studies/{self.study_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        with open(save_dir / "best_params.json", 'w') as f:
            json.dump(study.best_params, f, indent=2)

        # Save all trials
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })

        with open(save_dir / "all_trials.json", 'w') as f:
            json.dump(trials_data, f, indent=2)

        logger.info(f"Study saved to {save_dir}")


class VisionTransformerTuner(HyperparameterTuner):
    """Specialized tuner for Vision Transformer"""

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest ViT-specific hyperparameters"""
        return {
            'input_shape': (224, 224, 3),
            'num_classes': self.y_train.shape[1],
            'patch_size': trial.suggest_categorical('patch_size', [8, 16, 32]),
            'projection_dim': trial.suggest_categorical('projection_dim', [256, 512, 768, 1024]),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 12, 16]),
            'transformer_layers': trial.suggest_int('transformer_layers', 6, 24, step=2),
            'mlp_head_units': [
                trial.suggest_categorical('mlp_dim_1', [1024, 2048, 4096]),
                trial.suggest_categorical('mlp_dim_2', [512, 1024, 2048])
            ],
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.5)
        }


class EfficientNetTuner(HyperparameterTuner):
    """Specialized tuner for EfficientNet"""

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest EfficientNet-specific hyperparameters"""
        return {
            'input_shape': (224, 224, 3),
            'num_classes': self.y_train.shape[1],
            'model_variant': trial.suggest_categorical('variant', ['B0', 'B1', 'B2', 'B3']),
            'pretrained': True,
            'fine_tune_layers': trial.suggest_int('fine_tune_layers', 0, 100, step=10),
            'dropout_rate': trial.suggest_uniform('dropout_rate', 0.2, 0.5),
            'use_attention': trial.suggest_categorical('use_attention', [True, False])
        }


class DataAugmentationTuner:
    """Tune data augmentation hyperparameters"""

    def __init__(self, base_model: keras.Model):
        self.base_model = base_model

    def suggest_augmentation_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest data augmentation hyperparameters"""
        return {
            'use_mixup': trial.suggest_categorical('use_mixup', [True, False]),
            'use_cutmix': trial.suggest_categorical('use_cutmix', [True, False]),
            'mixup_alpha': trial.suggest_uniform('mixup_alpha', 0.1, 0.4),
            'cutmix_alpha': trial.suggest_uniform('cutmix_alpha', 0.5, 1.5),
            'randaugment_n': trial.suggest_int('randaugment_n', 1, 4),
            'randaugment_m': trial.suggest_int('randaugment_m', 5, 15),
            'horizontal_flip_prob': trial.suggest_uniform('h_flip', 0.3, 0.7),
            'rotation_limit': trial.suggest_int('rotation', 15, 45),
        }


class TrainingScheduleTuner:
    """Tune training schedule and learning rate"""

    @staticmethod
    def suggest_schedule_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest training schedule hyperparameters"""
        return {
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 10),
            'lr_schedule': trial.suggest_categorical('lr_schedule', [
                'cosine', 'exponential', 'step', 'reduce_on_plateau'
            ]),
            'min_lr': trial.suggest_loguniform('min_lr', 1e-8, 1e-5),
            'gradient_clip': trial.suggest_uniform('grad_clip', 0.5, 2.0),
        }


def run_full_hyperparameter_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'efficientnet',
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Run comprehensive hyperparameter search

    Args:
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        model_type: Type of model to tune
        n_trials: Number of trials

    Returns:
        Best hyperparameters
    """
    logger.info(f"Starting full hyperparameter search for {model_type}")

    # Select tuner
    if model_type == 'vit':
        tuner = VisionTransformerTuner(
            model_builder=create_vit_model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            study_name=f"vit_tuning"
        )
    elif model_type == 'efficientnet':
        tuner = EfficientNetTuner(
            model_builder=create_efficientnet_model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            study_name=f"efficientnet_tuning"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run optimization
    study = tuner.optimize(n_trials=n_trials)

    return study.best_params


def visualize_optimization(study_name: str):
    """
    Visualize optimization results

    Args:
        study_name: Name of the study
    """
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
            plot_contour
        )
        import plotly

        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db"
        )

        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(f"optimization/visualizations/{study_name}_history.html")

        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_html(f"optimization/visualizations/{study_name}_importances.html")

        # Parameter relationships
        fig = plot_slice(study)
        fig.write_html(f"optimization/visualizations/{study_name}_slice.html")

        logger.info(f"Visualizations saved to optimization/visualizations/")

    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


if __name__ == "__main__":
    print("Hyperparameter Optimization Ready")
    print("Features:")
    print("- Optuna-based automated tuning")
    print("- ViT and EfficientNet specialized tuners")
    print("- Data augmentation tuning")
    print("- Training schedule optimization")
    print("- Visualization of optimization results")
