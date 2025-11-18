"""
Foundation Model Integration
SAM (Segment Anything) + CLIP for zero-shot classification
"""
from .sam_clip_integration import (
    SAMSegmentationAdapter,
    CLIPTerrainClassifier,
    SAMCLIPPipeline,
    FoundationModelEnsemble
)

__all__ = [
    'SAMSegmentationAdapter',
    'CLIPTerrainClassifier',
    'SAMCLIPPipeline',
    'FoundationModelEnsemble'
]
