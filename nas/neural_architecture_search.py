"""
Neural Architecture Search (NAS) for Satellite Terrain Classification
Automatically discover optimal model architectures
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from loguru import logger
import json
from pathlib import Path


class SearchSpace:
    """Define architecture search space"""

    def __init__(self):
        self.conv_layers = [16, 32, 64, 128, 256]
        self.dense_layers = [64, 128, 256, 512, 1024]
        self.kernel_sizes = [3, 5, 7]
        self.activations = ['relu', 'gelu', 'swish']
        self.pooling_types = ['max', 'avg']
        self.dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.num_conv_blocks = [2, 3, 4, 5]
        self.num_dense_layers = [1, 2, 3]

    def sample_architecture(self) -> Dict:
        """Sample random architecture from search space"""
        return {
            'num_conv_blocks': random.choice(self.num_conv_blocks),
            'conv_filters': [random.choice(self.conv_layers) for _ in range(5)],
            'kernel_sizes': [random.choice(self.kernel_sizes) for _ in range(5)],
            'activation': random.choice(self.activations),
            'pooling': random.choice(self.pooling_types),
            'num_dense': random.choice(self.num_dense_layers),
            'dense_units': [random.choice(self.dense_layers) for _ in range(3)],
            'dropout': random.choice(self.dropout_rates)
        }


class ArchitectureBuilder:
    """Build model from architecture specification"""

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, arch_spec: Dict) -> keras.Model:
        """Build model from architecture specification"""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        # Convolutional blocks
        for i in range(arch_spec['num_conv_blocks']):
            filters = arch_spec['conv_filters'][i]
            kernel_size = arch_spec['kernel_sizes'][i]
            activation = arch_spec['activation']

            x = keras.layers.Conv2D(
                filters, kernel_size,
                padding='same',
                activation=activation
            )(x)
            x = keras.layers.BatchNormalization()(x)

            # Pooling
            if arch_spec['pooling'] == 'max':
                x = keras.layers.MaxPooling2D(2)(x)
            else:
                x = keras.layers.AveragePooling2D(2)(x)

        # Global pooling
        x = keras.layers.GlobalAveragePooling2D()(x)

        # Dense layers
        for i in range(arch_spec['num_dense']):
            units = arch_spec['dense_units'][i]
            x = keras.layers.Dense(units, activation=arch_spec['activation'])(x)
            x = keras.layers.Dropout(arch_spec['dropout'])(x)

        # Output
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        return model


class RandomSearch:
    """Random architecture search"""

    def __init__(
        self,
        search_space: SearchSpace,
        builder: ArchitectureBuilder,
        num_trials: int = 50,
        epochs_per_trial: int = 10
    ):
        self.search_space = search_space
        self.builder = builder
        self.num_trials = num_trials
        self.epochs_per_trial = epochs_per_trial
        self.results = []

    def search(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset
    ) -> Dict:
        """Run random search"""
        logger.info(f"Starting random search with {self.num_trials} trials...")

        best_accuracy = 0
        best_arch = None

        for trial in range(self.num_trials):
            logger.info(f"Trial {trial + 1}/{self.num_trials}")

            # Sample architecture
            arch_spec = self.search_space.sample_architecture()

            # Build model
            model = self.builder.build(arch_spec)

            # Compile
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Train briefly
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=self.epochs_per_trial,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
                ]
            )

            # Evaluate
            val_accuracy = max(history.history['val_accuracy'])

            # Record results
            result = {
                'trial': trial,
                'architecture': arch_spec,
                'val_accuracy': float(val_accuracy),
                'params': model.count_params()
            }
            self.results.append(result)

            logger.info(f"Val accuracy: {val_accuracy:.4f} | Params: {model.count_params():,}")

            # Track best
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_arch = arch_spec
                logger.info(f"New best architecture! Accuracy: {best_accuracy:.4f}")

        logger.info(f"Search completed. Best accuracy: {best_accuracy:.4f}")

        return {
            'best_architecture': best_arch,
            'best_accuracy': float(best_accuracy),
            'all_results': self.results
        }

    def save_results(self, path: str):
        """Save search results"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'best_architecture': self.results[0] if self.results else None,
                'all_results': self.results
            }, f, indent=2)


class EvolutionarySearch:
    """Evolutionary architecture search with mutations"""

    def __init__(
        self,
        search_space: SearchSpace,
        builder: ArchitectureBuilder,
        population_size: int = 20,
        num_generations: int = 10,
        mutation_rate: float = 0.2
    ):
        self.search_space = search_space
        self.builder = builder
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    def mutate(self, arch_spec: Dict) -> Dict:
        """Mutate architecture"""
        mutated = arch_spec.copy()

        # Randomly mutate one aspect
        mutation_type = random.choice([
            'conv_filters', 'kernel_sizes', 'activation',
            'pooling', 'dense_units', 'dropout'
        ])

        if mutation_type == 'conv_filters':
            idx = random.randint(0, len(mutated['conv_filters']) - 1)
            mutated['conv_filters'][idx] = random.choice(self.search_space.conv_layers)

        elif mutation_type == 'kernel_sizes':
            idx = random.randint(0, len(mutated['kernel_sizes']) - 1)
            mutated['kernel_sizes'][idx] = random.choice(self.search_space.kernel_sizes)

        elif mutation_type == 'activation':
            mutated['activation'] = random.choice(self.search_space.activations)

        elif mutation_type == 'pooling':
            mutated['pooling'] = random.choice(self.search_space.pooling_types)

        elif mutation_type == 'dense_units':
            idx = random.randint(0, len(mutated['dense_units']) - 1)
            mutated['dense_units'][idx] = random.choice(self.search_space.dense_layers)

        elif mutation_type == 'dropout':
            mutated['dropout'] = random.choice(self.search_space.dropout_rates)

        return mutated

    def search(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset
    ) -> Dict:
        """Run evolutionary search"""
        logger.info(f"Starting evolutionary search...")

        # Initialize population
        population = [
            self.search_space.sample_architecture()
            for _ in range(self.population_size)
        ]

        best_overall = None
        best_accuracy = 0

        for generation in range(self.num_generations):
            logger.info(f"Generation {generation + 1}/{self.num_generations}")

            # Evaluate population
            fitness_scores = []

            for arch_spec in population:
                model = self.builder.build(arch_spec)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Quick training
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=5,
                    verbose=0
                )

                accuracy = max(history.history['val_accuracy'])
                fitness_scores.append(accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_overall = arch_spec

            # Select top performers
            top_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
            survivors = [population[i] for i in top_indices]

            # Create new population through mutation
            new_population = survivors.copy()

            while len(new_population) < self.population_size:
                parent = random.choice(survivors)
                if random.random() < self.mutation_rate:
                    child = self.mutate(parent)
                else:
                    child = parent.copy()
                new_population.append(child)

            population = new_population

            logger.info(f"Best accuracy in generation: {max(fitness_scores):.4f}")

        logger.info(f"Evolution completed. Best accuracy: {best_accuracy:.4f}")

        return {
            'best_architecture': best_overall,
            'best_accuracy': float(best_accuracy)
        }


class DARTS:
    """Differentiable Architecture Search (simplified)"""

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_search_cell(self) -> keras.Model:
        """Create differentiable search cell"""
        inputs = keras.Input(shape=self.input_shape)

        # Mixed operations (learnable weights)
        ops = []

        # Convolutions with different kernels
        ops.append(keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs))
        ops.append(keras.layers.Conv2D(64, 5, padding='same', activation='relu')(inputs))
        ops.append(keras.layers.Conv2D(64, 7, padding='same', activation='relu')(inputs))

        # Pooling
        ops.append(keras.layers.MaxPooling2D(3, strides=1, padding='same')(inputs))
        ops.append(keras.layers.AveragePooling2D(3, strides=1, padding='same')(inputs))

        # Skip connection
        ops.append(inputs)

        # Learnable combination weights
        # (In full DARTS, these would be learned through bilevel optimization)
        weights = [1.0 / len(ops)] * len(ops)

        # Weighted sum
        output = tf.add_n([op * w for op, w in zip(ops, weights)])

        return keras.Model(inputs, output)


def run_nas(
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    method: str = 'random',
    num_trials: int = 50
) -> Dict:
    """
    Run Neural Architecture Search

    Args:
        train_data: Training dataset
        val_data: Validation dataset
        input_shape: Input image shape
        num_classes: Number of classes
        method: Search method ('random' or 'evolutionary')
        num_trials: Number of trials/generations

    Returns:
        Best architecture and results
    """
    search_space = SearchSpace()
    builder = ArchitectureBuilder(input_shape, num_classes)

    if method == 'random':
        searcher = RandomSearch(search_space, builder, num_trials=num_trials)
    elif method == 'evolutionary':
        searcher = EvolutionarySearch(search_space, builder, num_generations=num_trials)
    else:
        raise ValueError(f"Unknown method: {method}")

    results = searcher.search(train_data, val_data)

    # Save results
    searcher.save_results(f'nas/results/{method}_search_results.json')

    return results


if __name__ == "__main__":
    print("Neural Architecture Search Ready")
    print("Methods:")
    print("- Random Search: Sample random architectures")
    print("- Evolutionary Search: Evolve architectures over generations")
    print("- DARTS: Differentiable architecture search")
    print("\nAutomatically discovers optimal architectures for your dataset!")
