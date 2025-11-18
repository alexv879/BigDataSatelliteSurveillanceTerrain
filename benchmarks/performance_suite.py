"""
Comprehensive Performance Benchmarking Suite
Test inference speed, throughput, memory usage, and accuracy
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Benchmark results"""
    model_name: str
    avg_latency_ms: float
    throughput_fps: float
    memory_mb: float
    accuracy: float
    top5_accuracy: float
    model_size_mb: float
    params_count: int


class PerformanceBenchmark:
    """Comprehensive performance benchmarking"""

    def __init__(self):
        self.results = []

    def benchmark_latency(
        self,
        model: keras.Model,
        test_images: np.ndarray,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference latency

        Args:
            model: Model to benchmark
            test_images: Test images
            num_iterations: Number of iterations
            warmup: Warmup iterations

        Returns:
            Latency statistics
        """
        logger.info("Benchmarking latency...")

        # Warmup
        for _ in range(warmup):
            _ = model.predict(test_images[:1], verbose=0)

        # Benchmark
        latencies = []

        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model.predict(test_images[:1], verbose=0)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        }

    def benchmark_throughput(
        self,
        model: keras.Model,
        test_images: np.ndarray,
        batch_sizes: List[int] = [1, 8, 16, 32, 64],
        duration: int = 10
    ) -> Dict[int, float]:
        """
        Benchmark throughput at different batch sizes

        Args:
            model: Model to benchmark
            test_images: Test images
            batch_sizes: Batch sizes to test
            duration: Benchmark duration in seconds

        Returns:
            Throughput (images/sec) for each batch size
        """
        logger.info("Benchmarking throughput...")

        throughputs = {}

        for batch_size in batch_sizes:
            # Prepare batch
            batch = test_images[:batch_size]

            # Warmup
            for _ in range(10):
                _ = model.predict(batch, verbose=0)

            # Benchmark
            start = time.time()
            iterations = 0

            while time.time() - start < duration:
                _ = model.predict(batch, verbose=0)
                iterations += 1

            elapsed = time.time() - start
            throughput = (iterations * batch_size) / elapsed

            throughputs[batch_size] = throughput
            logger.info(f"Batch {batch_size}: {throughput:.2f} images/sec")

        return throughputs

    def benchmark_memory(
        self,
        model: keras.Model,
        test_images: np.ndarray
    ) -> Dict[str, float]:
        """
        Benchmark memory usage

        Args:
            model: Model to benchmark
            test_images: Test images

        Returns:
            Memory statistics
        """
        logger.info("Benchmarking memory...")

        process = psutil.Process()

        # Initial memory
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB

        # Run inference
        _ = model.predict(test_images[:32], verbose=0)

        # Peak memory
        peak_mem = process.memory_info().rss / 1024 / 1024  # MB

        return {
            'initial_mb': initial_mem,
            'peak_mb': peak_mem,
            'delta_mb': peak_mem - initial_mem
        }

    def benchmark_accuracy(
        self,
        model: keras.Model,
        test_images: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Benchmark accuracy

        Args:
            model: Model to benchmark
            test_images: Test images
            test_labels: Test labels (one-hot)

        Returns:
            Accuracy metrics
        """
        logger.info("Benchmarking accuracy...")

        predictions = model.predict(test_images, verbose=0)

        # Top-1 accuracy
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        accuracy = np.mean(pred_classes == true_classes)

        # Top-5 accuracy
        top5_pred = np.argsort(predictions, axis=1)[:, -5:]
        top5_accuracy = np.mean([
            true_classes[i] in top5_pred[i]
            for i in range(len(true_classes))
        ])

        return {
            'accuracy': float(accuracy),
            'top5_accuracy': float(top5_accuracy)
        }

    def run_full_benchmark(
        self,
        model: keras.Model,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        model_name: str
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark

        Args:
            model: Model to benchmark
            test_images: Test images
            test_labels: Test labels
            model_name: Model name

        Returns:
            Complete benchmark results
        """
        logger.info(f"Running full benchmark for {model_name}...")

        # Latency
        latency_stats = self.benchmark_latency(model, test_images)

        # Throughput
        throughput_stats = self.benchmark_throughput(model, test_images)

        # Memory
        memory_stats = self.benchmark_memory(model, test_images)

        # Accuracy
        accuracy_stats = self.benchmark_accuracy(model, test_images, test_labels)

        # Model size
        model_size_mb = sum([
            np.prod(w.shape) * 4 / 1024 / 1024  # Assuming float32
            for w in model.get_weights()
        ])

        result = BenchmarkResult(
            model_name=model_name,
            avg_latency_ms=latency_stats['mean_ms'],
            throughput_fps=throughput_stats[1],  # Batch size 1
            memory_mb=memory_stats['peak_mb'],
            accuracy=accuracy_stats['accuracy'],
            top5_accuracy=accuracy_stats['top5_accuracy'],
            model_size_mb=model_size_mb,
            params_count=model.count_params()
        )

        self.results.append(result)
        logger.info(f"Benchmark completed for {model_name}")

        return result

    def save_results(self, path: str):
        """Save benchmark results"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            'results': [asdict(r) for r in self.results],
            'summary': self._generate_summary()
        }

        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {path}")

    def _generate_summary(self) -> Dict:
        """Generate benchmark summary"""
        if not self.results:
            return {}

        return {
            'fastest_model': min(
                self.results,
                key=lambda x: x.avg_latency_ms
            ).model_name,
            'most_accurate': max(
                self.results,
                key=lambda x: x.accuracy
            ).model_name,
            'smallest_model': min(
                self.results,
                key=lambda x: x.model_size_mb
            ).model_name,
            'highest_throughput': max(
                self.results,
                key=lambda x: x.throughput_fps
            ).model_name
        }

    def compare_models(self):
        """Print model comparison"""
        if not self.results:
            logger.warning("No results to compare")
            return

        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)

        # Header
        print(f"{'Model':<20} {'Latency (ms)':<15} {'FPS':<10} {'Accuracy':<12} {'Size (MB)':<12}")
        print("-"*80)

        # Results
        for result in sorted(self.results, key=lambda x: x.avg_latency_ms):
            print(
                f"{result.model_name:<20} "
                f"{result.avg_latency_ms:<15.2f} "
                f"{result.throughput_fps:<10.2f} "
                f"{result.accuracy*100:<12.2f} "
                f"{result.model_size_mb:<12.2f}"
            )

        print("="*80)

        # Summary
        summary = self._generate_summary()
        print("\nSUMMARY:")
        print(f"  Fastest: {summary['fastest_model']}")
        print(f"  Most Accurate: {summary['most_accurate']}")
        print(f"  Smallest: {summary['smallest_model']}")
        print(f"  Highest Throughput: {summary['highest_throughput']}")
        print()


if __name__ == "__main__":
    print("Performance Benchmarking Suite Ready")
    print("Metrics:")
    print("- Inference latency (mean, p50, p95, p99)")
    print("- Throughput at various batch sizes")
    print("- Memory usage")
    print("- Accuracy (top-1, top-5)")
    print("- Model size and parameters")
