"""
Active Learning for Satellite Terrain Classification
Intelligent sample selection to minimize labeling costs
"""
from .query_strategies import (
    QueryStrategy,
    UncertaintySampling,
    DiversitySampling,
    BALDSampling,
    HybridSampling,
    ActiveLearner
)

__all__ = [
    'QueryStrategy',
    'UncertaintySampling',
    'DiversitySampling',
    'BALDSampling',
    'HybridSampling',
    'ActiveLearner'
]
