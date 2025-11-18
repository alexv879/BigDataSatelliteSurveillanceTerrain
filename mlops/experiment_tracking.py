"""
MLOps Infrastructure: Experiment Tracking with MLflow and Weights & Biases
Track experiments, models, metrics, and artifacts for reproducibility
"""
import mlflow
import mlflow.tensorflow
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path
from loguru import logger


class ExperimentTracker:
    """Unified experiment tracking with MLflow and W&B"""

    def __init__(
        self,
        project_name: str = "satellite-terrain-classification",
        experiment_name: str = "default",
        use_mlflow: bool = True,
        use_wandb: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.config = config or {}

        # Initialize MLflow
        if self.use_mlflow:
            self._init_mlflow()

        # Initialize W&B
        if self.use_wandb:
            self._init_wandb()

    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()
        logger.info(f"MLflow tracking initialized: {mlflow.get_tracking_uri()}")

    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.config,
            reinit=True
        )
        logger.info(f"W&B tracking initialized: {wandb.run.name}")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.use_mlflow:
            mlflow.log_params(params)
        if self.use_wandb:
            wandb.config.update(params)
        logger.info(f"Logged parameters: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.use_mlflow:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_artifact(self, artifact_path: str):
        """Log artifact file"""
        if self.use_mlflow:
            mlflow.log_artifact(artifact_path)
        if self.use_wandb:
            wandb.save(artifact_path)
        logger.info(f"Logged artifact: {artifact_path}")

    def log_model(self, model: keras.Model, model_name: str = "model"):
        """Log trained model"""
        # Save model
        model_path = f"models/{model_name}"
        os.makedirs(model_path, exist_ok=True)

        if self.use_mlflow:
            mlflow.tensorflow.log_model(model, model_name)

        if self.use_wandb:
            model.save(f"{model_path}/model.h5")
            wandb.save(f"{model_path}/model.h5")

        logger.info(f"Logged model: {model_name}")

    def log_figure(self, figure, name: str):
        """Log matplotlib figure"""
        if self.use_mlflow:
            mlflow.log_figure(figure, f"{name}.png")
        if self.use_wandb:
            wandb.log({name: wandb.Image(figure)})

    def get_callbacks(self) -> list:
        """Get tracking callbacks for Keras training"""
        callbacks = []

        if self.use_mlflow:
            callbacks.append(MLflowCallback())

        if self.use_wandb:
            callbacks.append(WandbCallback(
                save_model=True,
                monitor='val_accuracy',
                mode='max',
                log_weights=True,
                log_gradients=True
            ))

        return callbacks

    def finish(self):
        """End tracking session"""
        if self.use_mlflow:
            mlflow.end_run()
        if self.use_wandb:
            wandb.finish()
        logger.info("Experiment tracking finished")


class MLflowCallback(keras.callbacks.Callback):
    """Custom MLflow callback for Keras"""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            mlflow.log_metric(key, value, step=epoch)


class ModelVersioning:
    """Model versioning and registry"""

    def __init__(self, model_registry_path: str = "models/registry"):
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load model registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}}

    def _save_metadata(self):
        """Save model registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(
        self,
        model: keras.Model,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        description: str = ""
    ):
        """
        Register a new model version

        Args:
            model: Keras model to register
            model_name: Name of the model
            version: Version string
            metrics: Model performance metrics
            config: Model configuration
            description: Model description
        """
        # Create version directory
        version_dir = self.registry_path / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_dir / "model.h5"
        model.save(model_path)

        # Save metadata
        model_metadata = {
            "version": version,
            "metrics": metrics,
            "config": config,
            "description": description,
            "path": str(model_path)
        }

        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {}

        self.metadata["models"][model_name][version] = model_metadata
        self._save_metadata()

        logger.info(f"Registered model: {model_name} v{version}")
        logger.info(f"Metrics: {metrics}")

    def load_model(self, model_name: str, version: str = "latest") -> keras.Model:
        """Load a model from registry"""
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        if version == "latest":
            version = max(self.metadata["models"][model_name].keys())

        model_info = self.metadata["models"][model_name][version]
        model_path = model_info["path"]

        logger.info(f"Loading model: {model_name} v{version}")
        return keras.models.load_model(model_path)

    def get_best_model(self, model_name: str, metric: str = "accuracy") -> tuple:
        """Get best model version based on metric"""
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        best_version = None
        best_metric = -float('inf')

        for version, info in self.metadata["models"][model_name].items():
            if metric in info["metrics"]:
                if info["metrics"][metric] > best_metric:
                    best_metric = info["metrics"][metric]
                    best_version = version

        logger.info(f"Best model: {model_name} v{best_version} ({metric}={best_metric})")
        return self.load_model(model_name, best_version), best_version


class PerformanceMonitor:
    """Monitor training performance and resource usage"""

    def __init__(self):
        self.metrics = {
            "training_time": [],
            "inference_time": [],
            "memory_usage": [],
            "gpu_usage": []
        }

    def log_training_time(self, duration: float):
        """Log training duration"""
        self.metrics["training_time"].append(duration)
        logger.info(f"Training time: {duration:.2f}s")

    def log_inference_time(self, duration: float):
        """Log inference duration"""
        self.metrics["inference_time"].append(duration)

    def log_memory_usage(self):
        """Log memory usage"""
        import psutil
        memory = psutil.virtual_memory()
        self.metrics["memory_usage"].append(memory.percent)

    def log_gpu_usage(self):
        """Log GPU usage if available"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Log GPU memory usage
            logger.info(f"GPUs available: {len(gpus)}")

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        import numpy as np
        summary = {}

        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }

        return summary


if __name__ == "__main__":
    print("MLOps Infrastructure Ready")
    print("Features:")
    print("- Experiment tracking with MLflow and W&B")
    print("- Model versioning and registry")
    print("- Performance monitoring")
    print("- Artifact logging")
    print("- Automated callbacks")
