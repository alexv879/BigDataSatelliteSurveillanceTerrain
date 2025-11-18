"""
Continual/Lifelong Learning
Update models without catastrophic forgetting
"""
from .lifelong_learning import (
    EWC,
    PackNet,
    LwF,
    ProgressiveNeuralNetwork,
    MemoryReplay
)

__all__ = [
    'EWC',
    'PackNet',
    'LwF',
    'ProgressiveNeuralNetwork',
    'MemoryReplay'
]
