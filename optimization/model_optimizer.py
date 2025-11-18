"""
Model Optimization: Quantization, Pruning, and TensorRT Conversion
Optimize models for faster inference and smaller size
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger
import time


class ModelQuantizer:
    """Quantize models for faster inference"""

    def __init__(self, model: keras.Model):
        self.model = model

    def quantize_int8(
        self,
        representative_dataset: tf.data.Dataset,
        save_path: str = "models/optimized/quantized_int8.tflite"
    ) -> str:
        """
        Quantize model to INT8 using post-training quantization

        Args:
            representative_dataset: Representative dataset for calibration
            save_path: Path to save quantized model

        Returns:
            Path to saved model
        """
        logger.info("Starting INT8 quantization...")

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for calibration
        def representative_data_gen():
            for data in representative_dataset.take(100):
                if isinstance(data, tuple):
                    images = data[0]
                else:
                    images = data
                yield [images.numpy()]

        converter.representative_dataset = representative_data_gen

        # Ensure INT8 for all ops
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        # Convert
        tflite_model = converter.convert()

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(tflite_model)

        model_size = len(tflite_model) / 1024 / 1024
        logger.info(f"INT8 quantized model saved to {save_path}")
        logger.info(f"Model size: {model_size:.2f} MB")

        return save_path

    def quantize_float16(
        self,
        save_path: str = "models/optimized/quantized_float16.tflite"
    ) -> str:
        """
        Quantize model to FLOAT16

        Args:
            save_path: Path to save quantized model

        Returns:
            Path to saved model
        """
        logger.info("Starting FLOAT16 quantization...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(tflite_model)

        model_size = len(tflite_model) / 1024 / 1024
        logger.info(f"FLOAT16 quantized model saved to {save_path}")
        logger.info(f"Model size: {model_size:.2f} MB")

        return save_path

    def dynamic_range_quantization(
        self,
        save_path: str = "models/optimized/quantized_dynamic.tflite"
    ) -> str:
        """
        Dynamic range quantization (weights only)

        Args:
            save_path: Path to save quantized model

        Returns:
            Path to saved model
        """
        logger.info("Starting dynamic range quantization...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(tflite_model)

        model_size = len(tflite_model) / 1024 / 1024
        logger.info(f"Dynamic range quantized model saved to {save_path}")
        logger.info(f"Model size: {model_size:.2f} MB")

        return save_path


class ModelPruner:
    """Prune models to reduce size and improve efficiency"""

    def __init__(self, model: keras.Model):
        self.model = model

    def create_pruned_model(
        self,
        target_sparsity: float = 0.5,
        begin_step: int = 0,
        end_step: int = 1000,
        frequency: int = 100
    ) -> keras.Model:
        """
        Create pruned model using magnitude-based pruning

        Args:
            target_sparsity: Target sparsity (0-1)
            begin_step: Step to start pruning
            end_step: Step to end pruning
            frequency: Pruning frequency

        Returns:
            Pruned model
        """
        logger.info(f"Creating pruned model (target sparsity: {target_sparsity})")

        # Define pruning schedule
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=begin_step,
            end_step=end_step,
            frequency=frequency
        )

        # Apply pruning
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            pruning_schedule=pruning_schedule
        )

        logger.info("Pruned model created successfully")
        return pruned_model

    def remove_pruning(self, pruned_model: keras.Model) -> keras.Model:
        """
        Remove pruning wrappers from model

        Args:
            pruned_model: Pruned model with wrappers

        Returns:
            Model without pruning wrappers
        """
        return tfmot.sparsity.keras.strip_pruning(pruned_model)


class TensorRTConverter:
    """Convert models to TensorRT for maximum performance"""

    def __init__(self, model: keras.Model):
        self.model = model

    def convert_to_tensorrt(
        self,
        save_path: str = "models/optimized/tensorrt_model",
        precision_mode: str = "FP16",
        max_workspace_size_bytes: int = 1 << 32
    ) -> str:
        """
        Convert model to TensorRT

        Args:
            save_path: Path to save TensorRT model
            precision_mode: Precision mode (FP32, FP16, INT8)
            max_workspace_size_bytes: Maximum workspace size

        Returns:
            Path to saved model
        """
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt

            logger.info(f"Converting to TensorRT ({precision_mode})...")

            # Save model
            temp_save_path = "models/temp_saved_model"
            self.model.save(temp_save_path)

            # Create TensorRT converter
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=precision_mode,
                max_workspace_size_bytes=max_workspace_size_bytes
            )

            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=temp_save_path,
                conversion_params=conversion_params
            )

            # Convert
            converter.convert()

            # Save
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            converter.save(save_path)

            logger.info(f"TensorRT model saved to {save_path}")
            return save_path

        except ImportError:
            logger.error("TensorRT not available. Install TensorFlow with TensorRT support.")
            raise


class ONNXConverter:
    """Convert models to ONNX format for cross-platform deployment"""

    def __init__(self, model: keras.Model):
        self.model = model

    def convert_to_onnx(
        self,
        save_path: str = "models/optimized/model.onnx",
        opset_version: int = 13
    ) -> str:
        """
        Convert Keras model to ONNX

        Args:
            save_path: Path to save ONNX model
            opset_version: ONNX opset version

        Returns:
            Path to saved model
        """
        try:
            import tf2onnx

            logger.info("Converting to ONNX...")

            # Convert
            spec = (tf.TensorSpec(self.model.input_shape, tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=spec,
                opset=opset_version
            )

            # Save
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"ONNX model saved to {save_path}")
            return save_path

        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
            raise


class BenchmarkOptimization:
    """Benchmark different optimization strategies"""

    def __init__(self):
        self.results = {}

    def benchmark_model(
        self,
        model_path: str,
        test_images: np.ndarray,
        num_iterations: int = 100
    ) -> dict:
        """
        Benchmark model performance

        Args:
            model_path: Path to model
            test_images: Test images
            num_iterations: Number of iterations

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking {model_path}...")

        # Load model
        if model_path.endswith('.tflite'):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Warmup
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], test_images[:1])
                interpreter.invoke()

            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                interpreter.set_tensor(input_details[0]['index'], test_images[:1])
                interpreter.invoke()
            end_time = time.time()

        else:
            model = keras.models.load_model(model_path)

            # Warmup
            model.predict(test_images[:1])

            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                model.predict(test_images[:1])
            end_time = time.time()

        avg_latency = (end_time - start_time) / num_iterations * 1000  # ms
        throughput = num_iterations / (end_time - start_time)  # inferences/sec

        results = {
            "average_latency_ms": avg_latency,
            "throughput_fps": throughput
        }

        logger.info(f"Average latency: {avg_latency:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} FPS")

        return results


def optimize_model_pipeline(
    model: keras.Model,
    representative_dataset: Optional[tf.data.Dataset] = None,
    methods: list = ["float16", "int8", "pruning"]
) -> dict:
    """
    Complete optimization pipeline

    Args:
        model: Keras model to optimize
        representative_dataset: Representative dataset for calibration
        methods: List of optimization methods to apply

    Returns:
        Dictionary of optimized model paths
    """
    results = {}

    # Quantization
    quantizer = ModelQuantizer(model)

    if "float16" in methods:
        results["float16"] = quantizer.quantize_float16()

    if "int8" in methods and representative_dataset:
        results["int8"] = quantizer.quantize_int8(representative_dataset)

    if "dynamic" in methods:
        results["dynamic"] = quantizer.dynamic_range_quantization()

    # Pruning
    if "pruning" in methods:
        pruner = ModelPruner(model)
        pruned_model = pruner.create_pruned_model()
        results["pruning"] = "models/optimized/pruned_model.h5"
        # Note: Pruned model needs to be fine-tuned before saving

    # ONNX
    if "onnx" in methods:
        onnx_converter = ONNXConverter(model)
        results["onnx"] = onnx_converter.convert_to_onnx()

    logger.info("Optimization pipeline completed!")
    logger.info(f"Optimized models: {list(results.keys())}")

    return results


if __name__ == "__main__":
    print("Model Optimization Tools Ready")
    print("Features:")
    print("- INT8/FLOAT16 Quantization")
    print("- Magnitude-based Pruning")
    print("- TensorRT Conversion")
    print("- ONNX Export")
    print("- Performance Benchmarking")
