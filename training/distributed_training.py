"""
Distributed Training Support
Multi-GPU and multi-worker training strategies
"""
import tensorflow as tf
from tensorflow import keras
import os
from typing import Optional, Callable, Dict, Any
from loguru import logger
import json


class DistributedTrainer:
    """Distributed training with multiple GPUs"""

    def __init__(
        self,
        strategy_type: str = 'mirrored',
        num_gpus: Optional[int] = None
    ):
        """
        Initialize distributed trainer

        Args:
            strategy_type: Type of distribution strategy
                - 'mirrored': Single machine, multiple GPUs
                - 'multi_worker': Multiple machines
                - 'tpu': TPU training
            num_gpus: Number of GPUs to use (None = all available)
        """
        self.strategy_type = strategy_type
        self.num_gpus = num_gpus
        self.strategy = self._create_strategy()

        logger.info(f"Distributed training strategy: {strategy_type}")
        logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

    def _create_strategy(self) -> tf.distribute.Strategy:
        """Create distribution strategy"""

        if self.strategy_type == 'mirrored':
            # Single machine, multiple GPUs
            if self.num_gpus:
                gpus = tf.config.list_physical_devices('GPU')[:self.num_gpus]
                return tf.distribute.MirroredStrategy(devices=[gpu.name for gpu in gpus])
            else:
                return tf.distribute.MirroredStrategy()

        elif self.strategy_type == 'multi_worker':
            # Multiple machines
            return tf.distribute.MultiWorkerMirroredStrategy()

        elif self.strategy_type == 'tpu':
            # TPU training
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.TPUStrategy(resolver)

        elif self.strategy_type == 'parameter_server':
            # Parameter server strategy
            return tf.distribute.ParameterServerStrategy()

        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

    def create_distributed_dataset(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        options: Optional[tf.distribute.InputOptions] = None
    ) -> tf.data.Dataset:
        """
        Create distributed dataset

        Args:
            dataset: Input dataset
            batch_size: Global batch size
            options: Optional input options

        Returns:
            Distributed dataset
        """
        # Adjust batch size for number of replicas
        per_replica_batch_size = batch_size // self.strategy.num_replicas_in_sync

        dataset = dataset.batch(per_replica_batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Distribute dataset
        if options is None:
            options = tf.distribute.InputOptions(
                experimental_fetch_to_device=True,
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA
            )

        return self.strategy.experimental_distribute_dataset(dataset, options)

    def train_distributed(
        self,
        model_builder: Callable,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int,
        callbacks: Optional[list] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> keras.Model:
        """
        Train model with distributed strategy

        Args:
            model_builder: Function to build model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            callbacks: Training callbacks
            model_config: Model configuration

        Returns:
            Trained model
        """
        logger.info("Starting distributed training...")

        with self.strategy.scope():
            # Build model within strategy scope
            model = model_builder(**(model_config or {}))

            logger.info(f"Model created with {model.count_params():,} parameters")

        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks or [],
            verbose=1
        )

        logger.info("Distributed training completed")
        return model, history


class GradientAccumulationTrainer:
    """Gradient accumulation for large batch training"""

    def __init__(
        self,
        model: keras.Model,
        accumulation_steps: int = 4
    ):
        """
        Initialize gradient accumulation trainer

        Args:
            model: Keras model
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in model.trainable_variables
        ]

        logger.info(f"Gradient accumulation with {accumulation_steps} steps")

    @tf.function
    def train_step(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        optimizer: keras.optimizers.Optimizer,
        loss_fn: Callable,
        step: tf.Variable
    ) -> Dict[str, tf.Tensor]:
        """
        Single training step with gradient accumulation

        Args:
            x: Input batch
            y: Target batch
            optimizer: Optimizer
            loss_fn: Loss function
            step: Current step counter

        Returns:
            Metrics dictionary
        """
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = loss_fn(y, predictions) / self.accumulation_steps

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Accumulate gradients
        for i, grad in enumerate(gradients):
            self.gradient_accumulation[i].assign_add(grad)

        # Apply gradients every N steps
        tf.cond(
            tf.equal(step % self.accumulation_steps, 0),
            lambda: self._apply_gradients(optimizer),
            lambda: None
        )

        return {'loss': loss * self.accumulation_steps}

    def _apply_gradients(self, optimizer):
        """Apply accumulated gradients"""
        optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.model.trainable_variables)
        )

        # Reset accumulation
        for var in self.gradient_accumulation:
            var.assign(tf.zeros_like(var))


class MixedPrecisionTrainer:
    """Advanced mixed precision training"""

    def __init__(
        self,
        policy: str = 'mixed_float16',
        loss_scale: Optional[float] = None
    ):
        """
        Initialize mixed precision trainer

        Args:
            policy: Mixed precision policy
            loss_scale: Optional custom loss scale
        """
        self.policy_name = policy
        self.policy = tf.keras.mixed_precision.Policy(policy)
        tf.keras.mixed_precision.set_global_policy(self.policy)

        self.loss_scale = loss_scale
        logger.info(f"Mixed precision training: {policy}")

    def get_optimizer(
        self,
        base_optimizer: keras.optimizers.Optimizer
    ) -> keras.optimizers.Optimizer:
        """
        Wrap optimizer for mixed precision

        Args:
            base_optimizer: Base optimizer

        Returns:
            Mixed precision optimizer
        """
        if self.policy_name == 'mixed_float16':
            if self.loss_scale:
                return tf.keras.mixed_precision.LossScaleOptimizer(
                    base_optimizer,
                    dynamic=False,
                    initial_scale=self.loss_scale
                )
            else:
                return tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        return base_optimizer


class DataParallelTrainer:
    """Data parallelism across multiple devices"""

    def __init__(self, devices: list):
        """
        Initialize data parallel trainer

        Args:
            devices: List of device names
        """
        self.devices = devices
        self.num_devices = len(devices)
        logger.info(f"Data parallel training on {self.num_devices} devices")

    def replicate_model(self, model: keras.Model) -> list:
        """Replicate model across devices"""
        replicas = []

        for device in self.devices:
            with tf.device(device):
                replica = keras.models.clone_model(model)
                replica.set_weights(model.get_weights())
                replicas.append(replica)

        return replicas

    @tf.function
    def parallel_train_step(
        self,
        model_replicas: list,
        x_batches: list,
        y_batches: list,
        optimizer: keras.optimizers.Optimizer,
        loss_fn: Callable
    ) -> tf.Tensor:
        """
        Parallel training step across devices

        Args:
            model_replicas: Model replicas
            x_batches: Input batches for each device
            y_batches: Target batches for each device
            optimizer: Optimizer
            loss_fn: Loss function

        Returns:
            Average loss
        """
        losses = []
        all_gradients = []

        for i, (model, x, y) in enumerate(zip(model_replicas, x_batches, y_batches)):
            with tf.device(self.devices[i]):
                with tf.GradientTape() as tape:
                    predictions = model(x, training=True)
                    loss = loss_fn(y, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                all_gradients.append(gradients)
                losses.append(loss)

        # Average gradients
        avg_gradients = []
        for grads in zip(*all_gradients):
            avg_grad = tf.reduce_mean(tf.stack(grads), axis=0)
            avg_gradients.append(avg_grad)

        # Apply gradients to all replicas
        for model in model_replicas:
            optimizer.apply_gradients(zip(avg_gradients, model.trainable_variables))

        return tf.reduce_mean(losses)


def setup_multi_worker_training():
    """Setup multi-worker training configuration"""

    # Get worker configuration from environment
    tf_config = {
        'cluster': {
            'worker': ['localhost:12345', 'localhost:23456']
        },
        'task': {'type': 'worker', 'index': 0}
    }

    # Set TF_CONFIG environment variable
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    logger.info("Multi-worker configuration set")
    logger.info(f"TF_CONFIG: {tf_config}")


def configure_gpu_memory_growth():
    """Configure GPUs for memory growth"""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"Configured memory growth for {len(gpus)} GPUs")

        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPUs found")


def create_distributed_callbacks(
    strategy: tf.distribute.Strategy
) -> list:
    """
    Create callbacks for distributed training

    Args:
        strategy: Distribution strategy

    Returns:
        List of callbacks
    """
    callbacks = []

    # Backup and restore for fault tolerance
    callbacks.append(keras.callbacks.BackupAndRestore(
        backup_dir='./training_checkpoints'
    ))

    # Model checkpoint (save only on chief worker)
    if strategy.cluster_resolver and strategy.cluster_resolver.task_type == 'chief':
        callbacks.append(keras.callbacks.ModelCheckpoint(
            'models/distributed_checkpoint.h5',
            save_best_only=True
        ))

    # TensorBoard with profile
    callbacks.append(keras.callbacks.TensorBoard(
        log_dir='logs/distributed',
        profile_batch='10,20',
        update_freq='epoch'
    ))

    return callbacks


if __name__ == "__main__":
    print("Distributed Training Ready")
    print("Features:")
    print("- Multi-GPU training (MirroredStrategy)")
    print("- Multi-worker training (MultiWorkerMirroredStrategy)")
    print("- TPU support (TPUStrategy)")
    print("- Gradient accumulation for large batches")
    print("- Advanced mixed precision training")
    print("- Data parallelism")
    print("- Fault tolerance with backup/restore")
