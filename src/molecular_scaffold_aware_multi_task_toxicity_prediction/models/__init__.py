"""Graph neural network models for molecular property prediction."""

from .model import (
    ScaffoldAwareGCN,
    ScaffoldAwareGAT,
    ScaffoldAwareGraphSAGE,
    AttentionSubstructurePooling,
    MultiTaskToxicityPredictor,
)

__all__ = [
    "ScaffoldAwareGCN",
    "ScaffoldAwareGAT",
    "ScaffoldAwareGraphSAGE",
    "AttentionSubstructurePooling",
    "MultiTaskToxicityPredictor",
]