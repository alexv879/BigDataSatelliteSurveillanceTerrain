"""
Resource Management
Manages system resources (memory, GPU, file handles) for safety and efficiency
"""
import os
import psutil
import gc
from typing import Optional, Callable
from contextlib import contextmanager
from loguru import logger
import tensorflow as tf


class ResourceManager:
    """
    System resource management
    Monitors and controls memory, GPU, and other resources
    """

    def __init__(
        self,
        max_memory_percent: float = 0.8,
        enable_gpu_growth: bool = True,
        gpu_memory_fraction: float = 0.9
    ):
        """
        Initialize resource manager

        Args:
            max_memory_percent: Maximum RAM usage (0-1)
            enable_gpu_growth: Enable GPU memory growth
            gpu_memory_fraction: Maximum GPU memory fraction
        """
        self.max_memory_percent = max_memory_percent
        self.enable_gpu_growth = enable_gpu_growth
        self.gpu_memory_fraction = gpu_memory_fraction

        # Configure GPU
        self._configure_gpu()

    def _configure_gpu(self):
        """Configure GPU settings for safety"""
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    if self.enable_gpu_growth:
                        # Enable memory growth
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for {gpu}")
                    else:
                        # Set fixed memory limit
                        memory_limit = int(
                            self.gpu_memory_fraction * get_gpu_memory(gpu)
                        )
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=memory_limit
                            )]
                        )
                        logger.info(f"Set memory limit to {memory_limit}MB for {gpu}")

                logger.success(f"Configured {len(gpus)} GPU(s)")

            except RuntimeError as e:
                logger.error(f"GPU configuration failed: {e}")
        else:
            logger.warning("No GPUs available")

    def check_memory(self) -> dict:
        """
        Check current memory usage

        Returns:
            Dictionary with memory statistics
        """
        memory = psutil.virtual_memory()

        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'within_limit': memory.percent < (self.max_memory_percent * 100)
        }

    def enforce_memory_limit(self):
        """
        Enforce memory limit

        Raises:
            MemoryError: If memory usage exceeds limit
        """
        memory_stats = self.check_memory()

        if not memory_stats['within_limit']:
            # Force garbage collection
            gc.collect()

            # Check again
            memory_stats = self.check_memory()

            if not memory_stats['within_limit']:
                raise MemoryError(
                    f"Memory usage {memory_stats['percent']:.1f}% exceeds "
                    f"limit {self.max_memory_percent * 100}%"
                )

    @contextmanager
    def managed_memory(self):
        """
        Context manager for memory-safe operations

        Usage:
            with resource_manager.managed_memory():
                # Memory-intensive operation
                pass
        """
        # Check before
        self.enforce_memory_limit()

        try:
            yield
        finally:
            # Cleanup after
            gc.collect()

    def get_gpu_info(self) -> list:
        """
        Get GPU information

        Returns:
            List of GPU info dictionaries
        """
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info = []

        for i, gpu in enumerate(gpus):
            info = {
                'index': i,
                'name': gpu.name,
                'memory_limit_set': self.enable_gpu_growth is False
            }
            gpu_info.append(info)

        return gpu_info

    def cleanup_gpu_memory(self):
        """Cleanup GPU memory"""
        try:
            from tensorflow.python.keras import backend as K
            K.clear_session()
            logger.info("Cleared GPU memory")
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")


def get_gpu_memory(gpu) -> int:
    """
    Get total GPU memory in MB

    Args:
        gpu: GPU device

    Returns:
        Memory in MB
    """
    try:
        # Try to get from nvidia-smi
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass

    # Default to 8GB if can't determine
    return 8192


@contextmanager
def limit_cpu_usage(max_threads: Optional[int] = None):
    """
    Context manager to limit CPU usage

    Args:
        max_threads: Maximum number of threads

    Usage:
        with limit_cpu_usage(4):
            # CPU-intensive operation
            pass
    """
    if max_threads is None:
        max_threads = max(1, psutil.cpu_count() // 2)

    # Save original settings
    original_threads = tf.config.threading.get_intra_op_parallelism_threads()

    try:
        # Set thread limit
        tf.config.threading.set_intra_op_parallelism_threads(max_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max_threads)

        logger.debug(f"Limited CPU threads to {max_threads}")
        yield

    finally:
        # Restore original settings
        tf.config.threading.set_intra_op_parallelism_threads(original_threads)


@contextmanager
def timeout(seconds: int):
    """
    Context manager for operation timeout

    Args:
        seconds: Timeout in seconds

    Usage:
        with timeout(30):
            # Operation with timeout
            pass
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds} seconds")

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class FileHandleManager:
    """
    Manage file handles to prevent resource leaks
    """

    def __init__(self, max_open_files: int = 100):
        """
        Initialize file handle manager

        Args:
            max_open_files: Maximum number of simultaneously open files
        """
        self.max_open_files = max_open_files
        self.open_files = []

    @contextmanager
    def open_file(self, path: str, mode: str = 'r'):
        """
        Safely open file with automatic cleanup

        Args:
            path: File path
            mode: Open mode

        Usage:
            with manager.open_file('data.txt') as f:
                data = f.read()
        """
        # Check limit
        if len(self.open_files) >= self.max_open_files:
            raise RuntimeError(
                f"Too many open files: {len(self.open_files)} "
                f"(max: {self.max_open_files})"
            )

        # Open file
        f = open(path, mode)
        self.open_files.append(f)

        try:
            yield f
        finally:
            # Always close
            f.close()
            self.open_files.remove(f)

    def close_all(self):
        """Close all open files"""
        for f in self.open_files[:]:
            try:
                f.close()
                self.open_files.remove(f)
            except Exception as e:
                logger.warning(f"Failed to close file: {e}")

        logger.info(f"Closed all file handles")

    def __del__(self):
        """Cleanup on deletion"""
        self.close_all()


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """
    Get global resource manager instance

    Returns:
        ResourceManager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def log_resource_usage():
    """Log current resource usage"""
    manager = get_resource_manager()

    # Memory
    memory = manager.check_memory()
    logger.info(
        f"Memory: {memory['used_gb']:.2f}GB / {memory['total_gb']:.2f}GB "
        f"({memory['percent']:.1f}%)"
    )

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU: {cpu_percent}%")

    # GPU
    gpus = manager.get_gpu_info()
    if gpus:
        for gpu in gpus:
            logger.info(f"GPU {gpu['index']}: {gpu['name']}")
