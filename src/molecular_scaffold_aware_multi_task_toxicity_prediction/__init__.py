"""Molecular Scaffold-Aware Multi-Task Toxicity Prediction.

A comprehensive study comparing scaffold-aware graph neural network architectures
for simultaneous prediction of multiple toxicity endpoints.
"""

__version__ = "1.0.0"
__author__ = "Molecular ML Research Team"

from .data import loader, preprocessing
from .models import model
from .training import trainer
from .evaluation import metrics
from .utils import config

__all__ = [
    "loader",
    "preprocessing",
    "model",
    "trainer",
    "metrics",
    "config",
]